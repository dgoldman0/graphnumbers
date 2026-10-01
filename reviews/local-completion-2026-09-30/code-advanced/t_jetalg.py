import random
from fractions import Fraction as Q
import sympy as sp
from graphlocal.nonspectral import MomentJet, ROOT_DEGREE, rooted_cliques, multi_indices
random.seed(3)
a, b = sp.symbols('a b')
axes = (ROOT_DEGREE, rooted_cliques(3))
def to_poly(j):
    return sum(sp.Rational(c.numerator, c.denominator) * a**al[0] * b**al[1] for al, c in j.coefficients.items())
def trunc(expr, N):
    p = sp.Poly(sp.expand(expr), a, b)
    return sum(c * a**m[0] * b**m[1] for m, c in zip(p.monoms(), p.coeffs()) if sum(m) <= N)
def series_trunc(f, N):
    # expand f(t*a, t*b) in t up to order N
    t = sp.symbols('t')
    s = sp.series(f.subs({a: t*a, b: t*b}), t, 0, N + 1).removeO()
    return sp.expand(s.subs(t, 1))
fails = 0
for trial in range(30):
    N = random.randint(1, 4)
    coeffs = {al: Q(random.randint(-5, 5), random.randint(1, 4)) for al in multi_indices(2, N) if random.random() < .7}
    j = MomentJet(axes, N, coeffs)
    P = to_poly(j)
    m = j.mass
    if m:
        rec = j.reciprocal()
        if sp.expand(to_poly(rec) - series_trunc(1 / P, N)) != 0:
            fails += 1; print("RECIP FAIL", coeffs)
    if m == 1:
        lg = j.log()
        if sp.expand(to_poly(lg) - series_trunc(sp.log(P), N)) != 0:
            fails += 1; print("LOG FAIL", coeffs)
    j0 = j - m
    ex = j0.exp()
    if sp.expand(to_poly(ex) - series_trunc(sp.exp(to_poly(j0)), N)) != 0:
        fails += 1; print("EXP FAIL", coeffs)
    j2 = MomentJet(axes, N, {al: Q(random.randint(-3, 3)) for al in multi_indices(2, N)})
    if sp.expand(to_poly(j * j2) - trunc(P * to_poly(j2), N)) != 0:
        fails += 1; print("MUL FAIL")
    if N >= 2 and sp.expand(to_poly(j.truncate(N - 1)) - trunc(P, N - 1)) != 0:
        fails += 1; print("TRUNC FAIL")
    if sp.expand(to_poly(j ** 3) - trunc(P**3, N)) != 0:
        fails += 1; print("POW FAIL")
print("fails", fails)
