from fractions import Fraction as Q
from math import comb
from graphlocal import *
from graphlocal.local_inverse import refine_local_inverse, local_inverse_certificate

H = Finite.from_graph(path(2), normalize=True)
def qn_ball(n, r): return sum(comb(n, j) for j in range(min(n, r) + 1))

class Adversarial(Element):
    """Exact H*c plus in-support perturbation of size delta (weighted), sign s."""
    def __init__(self, c, sign=1, frac=Q(1)):
        self.c, self.sign, self.frac = Q(c), sign, frac
        self.degree_bound, self.variation_bound, self.positive = 1, Q(c), True
    def approximate(self, radius, k=1, epsilon="1e-6"):
        eps = Q(epsilon) * self.frac
        loc = (H * self.c).local(radius)
        if radius:
            pert = LocalHistogram(radius, [(path(2), self.sign * eps / 2 ** k)])
        else:
            pert = LocalHistogram(0, [(graph(1), self.sign * eps)])
        return LocalApproximation(loc + pert, k, eps)

def exact_inverse_error(hist, c, r, k, extra=4000):
    # exact T_r Z = sum c^n [Q_n]; types identified by root degree n
    got = {}
    for key, v in hist.values.items():
        got[key.graph.rows[0].bit_count()] = got.get(key.graph.rows[0].bit_count(), 0) + v
    err = Q(0)
    nmax = max(got) if got else 0
    for n in range(0, nmax + extra):
        err += abs(got.get(n, 0) - c ** n) * qn_ball(n, r) ** k
    return err

fails = 0
for c in (Q(1, 4), Q(1, 2)):
    for r, k in ((1, 1), (1, 2), (2, 1)):
        for sign in (1, -1):
            for eps in ("1e-3", "1e-6"):
                src = Adversarial(c, sign)
                cert = NeumannInverse(src).approximation_certificate(r, k, eps)
                actual = exact_inverse_error(cert.approximation.histogram, c, r, k)
                ok = actual <= cert.approximation.error <= Q(eps)
                fails += not ok
                print(f"Neumann c={c} r={r} k={k} sign={sign} eps={eps} actual={float(actual):.3e} bound={float(cert.approximation.error):.3e} stab={float(cert.stability_bound):.3e} ok={ok}")
# refine_local_inverse on 1 - H*c with adversarial noise; exact inverse = sum c^n [Q_n]
class AdvOneMinus(Adversarial):
    def approximate(self, radius, k=1, epsilon="1e-6"):
        a = super().approximate(radius, k, epsilon)
        return LocalApproximation(Finite.scalar(1).local(radius) - a.histogram, k, a.error)
for c in (Q(1, 4), Q(1, 3)):
    for r, k in ((1, 1), (1, 2)):
        for sign in (1, -1):
            src = AdvOneMinus(c, sign)
            cand = (1 + H * c).local(r)
            try:
                res = refine_local_inverse(src, cand, k=k, epsilon="1e-6", source_epsilon="1e-3")
            except Exception as e:
                print("refine EXC", c, r, k, sign, type(e).__name__, e); continue
            actual = exact_inverse_error(res.approximation.histogram, c, r, k)
            ok = actual <= res.approximation.error <= Q("1e-6")
            fails += not ok
            print(f"Refine c={c} r={r} k={k} sign={sign} actual={float(actual):.3e} bound={float(res.approximation.error):.3e} ok={ok}")
            init = local_inverse_certificate(src, cand, k=k, source_epsilon="1e-3")
            actual0 = exact_inverse_error(cand, c, r, k)
            ok0 = actual0 <= init.approximation.error
            fails += not ok0
            print(f"  initial actual={float(actual0):.3e} bound={float(init.approximation.error):.3e} ok={ok0}")
print("fails", fails)
