from fractions import Fraction as Q
from math import factorial
from graphlocal import *
from graphlocal.defect_exponential import CutLineExponential

def series(t, r, N):
    src = CutLineDefect().local(r).scale(t)
    term = total = LocalHistogram(r, [(graph(1), Q(1))])
    for n in range(1, N + 1):
        term = term.multiply(src).scale(Q(1, n))
        total = total + term
    return total

for t in (Q(1, 10), Q(-1, 3), Q(1), Q(3, 2)):
    for r in (1, 2):
        for k in (1, 2):
            for eps in ("1e-3", "1e-6"):
                X = CutLineExponential(t)
                try:
                    cert = X.approximation_certificate(r, k, eps)
                except Exception as e:
                    print(t, r, k, eps, "EXC", type(e).__name__, e); continue
                N = cert.truncation_degree
                ref = series(t, r, N + 12)
                diff = (ref - cert.approximation.histogram).norm(k)
                ok = diff <= cert.approximation.error <= Q(eps)
                # exp(4r|t|) variation claim for r>=2
                var = ref.norm(0)
                print(f"t={t} r={r} k={k} eps={eps} N={N} err={float(cert.approximation.error):.3e} diff12={float(diff):.3e} ok={ok} var={float(var):.6f} exp4rt={float(__import__('mpmath').e**(4*r*abs(t))):.6f}")
