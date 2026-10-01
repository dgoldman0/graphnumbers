from fractions import Fraction as Q
from math import factorial
import sympy as sp
from graphlocal import *
from graphlocal.nonspectral import certified_jet, ROOT_DEGREE, rooted_cliques, link_components

z = sp.symbols('z')
def moments_from_gf(F, order):
    out = []
    G = F
    for j in range(order + 1):
        out.append(sp.nsimplify(sp.simplify(G.subs(z, 1))))
        G = z * sp.diff(G, z)
    return out

fails = 0
# NeumannInverse(H/c): degree law sum_n c^-n delta_n  -> F(z) = 1/(1 - z/c)
H = Finite.from_graph(path(2), normalize=True)
for c in (4, 3):
    inv = NeumannInverse(H / c)
    for order in (1, 2, 3):
        cert = certified_jet(inv, (ROOT_DEGREE,), order, "1e-6")
        M = moments_from_gf(1 / (1 - z / c), order)
        for j in range(order + 1):
            exact = Q(str(M[j])) / factorial(j)
            if not cert.interval((j,)).contains(exact):
                fails += 1; print("FAIL neumann", c, order, j, exact, cert.interval((j,)))
# CutLineExponential at radius 1: degree law exp(2t z - 2t z^2)
for t in (Q(1, 10), Q(-1, 5), Q(1, 2)):
    X = CutLineExponential(t)
    ts = sp.Rational(t.numerator, t.denominator)
    for order in (1, 2, 3):
        cert = certified_jet(X, (ROOT_DEGREE,), order, "1e-5")
        M = moments_from_gf(sp.exp(2 * ts * z - 2 * ts * z**2), order)
        for j in range(order + 1):
            exact = M[j]
            iv = cert.interval((j,))
            lo, hi = iv.lower, iv.upper
            ex = sp.N(exact / factorial(j), 50)
            if not (sp.Rational(lo.numerator, lo.denominator) <= ex <= sp.Rational(hi.numerator, hi.denominator)):
                fails += 1; print("FAIL cutexp", t, order, j, ex, float(lo), float(hi))
print("fails", fails)
