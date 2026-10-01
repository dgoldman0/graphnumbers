import mpmath
from fractions import Fraction as Q
from graphlocal import *
import oracle
mpmath.mp.dps = 50
def h_line(t): t = oracle.mpq(t); return mpmath.e ** (-2 * t) * mpmath.besseli(0, 2 * t)
def h_E(t): t = oracle.mpq(t); return (1 - mpmath.e ** (-4 * t)) / 2
def h_path(n, t): return oracle.graph_heat(path(n).rows, t)
def h_I(ell, t): return h_path(ell, t) - ell * h_line(t) - h_E(t)

fails = 0
for t in (Q(1, 4), Q(1, 2), Q(3, 2)):
    cases = [("I3", CutInteraction(3), h_I(3, t)),
             ("I1", CutInteraction(1), h_I(1, t)),
             ("TwoCut2", TwoCutLineDefect(2), 2 * h_E(t) + h_I(2, t)),
             ("LCD[0,2,7]", LineCutDefect([0, 2, 7]), 3 * h_E(t) + h_I(2, t) + h_I(5, t)),
             ("I2*E", CutInteraction(2) * CutLineDefect(), h_I(2, t) * h_E(t)),
             ("I2*L", CutInteraction(2) * Line(), h_I(2, t) * h_line(t)),
             ]
    for name, X, true in cases:
        for fn in (controlled_heat, relative_heat):
            try:
                c = fn(X, t, "1e-8")
            except Exception as e:
                print(name, fn.__name__, t, "EXC", type(e).__name__, e); continue
            lo, hi = oracle.mpq(c.interval.lower), oracle.mpq(c.interval.upper)
            ok = lo - mpmath.mpf(10) ** -40 <= true <= hi + mpmath.mpf(10) ** -40
            if not ok: fails += 1
            print(name, fn.__name__, t, "ok" if ok else "FAIL", mpmath.nstr(true, 15), float(lo), float(hi))
print("fails", fails)
