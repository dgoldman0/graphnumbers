import mpmath
from fractions import Fraction as Q
from graphlocal import *
from graphlocal.interaction_bounds import GeometricInteraction
import oracle
mpmath.mp.dps = 60
def line_heat(t):  # return probability on Z with unit edge rates
    t = oracle.mpq(t); return mpmath.e ** (-2 * t) * mpmath.besseli(0, 2 * t)
def cut_heat(t):
    t = oracle.mpq(t); return (1 - mpmath.e ** (-4 * t)) / 2

x = EdgeInteraction(path(5), [(0, 1, -1), (3, 4, -1)])          # bridge cuts
y = EdgeInteraction(cycle(5), [(0, 1, -1), (2, 3, -1), (0, 2, 1)])  # mixed
z = EdgeInteraction(star(3), [(0, 1, -1), (0, 2, -1), (0, 3, -1)])
def hx(e, t):
    return oracle.heat(e.before.rows, [(a, b, s) for a, b, s in e.edits], t) * oracle.mpq(e.scale)

fails = 0
for t in (Q(1, 4), Q(1)):
    cases = [
        ("x*L", x * Line(), hx(x, t) * line_heat(t)),
        ("G(x)*L", GeometricInteraction(x) * Line(), hx(x, t) * line_heat(t)),
        ("y*E", y * CutLineDefect(), hx(y, t) * cut_heat(t)),
        ("G(y)*G(z)", GeometricInteraction(y) * GeometricInteraction(z), hx(y, t) * hx(z, t)),
        ("3x-2z+L", 3 * x - 2 * z + Line(), 3 * hx(x, t) - 2 * hx(z, t) + line_heat(t)),
        ("(x+E)*(z-E)", (x + CutLineDefect()) * (z - CutLineDefect()), (hx(x, t) + cut_heat(t)) * (hx(z, t) - cut_heat(t))),
        ("E^2*L", CutLineDefect() ** 2 * Line(), cut_heat(t) ** 2 * line_heat(t)),
        ("G(z)/3*E", GeometricInteraction(z) / 3 * CutLineDefect(), hx(z, t) / 3 * cut_heat(t)),
    ]
    for name, X, true in cases:
        try:
            c = controlled_heat(X, t, "1e-7", max_steps=400)
        except Exception as e:
            print(name, t, "EXC", type(e).__name__, e); continue
        lo, hi = oracle.mpq(c.interval.lower), oracle.mpq(c.interval.upper)
        ok = lo - mpmath.mpf(10)**-45 <= true <= hi + mpmath.mpf(10)**-45 and c.interval.radius <= Q("1e-7")
        if not ok: fails += 1
        print(name, t, "ok" if ok else "FAIL", float(true), float(lo), float(hi), "steps", c.steps, "radius", c.radius)
print("fails", fails)
