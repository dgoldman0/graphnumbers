from fractions import Fraction as Q
import mpmath, time
from graphlocal import *
from graphlocal.defect_exponential import CutLineExponential

def series(t, r, N):
    src = CutLineDefect().local(r).scale(t)
    term = total = LocalHistogram(r, [(graph(1), Q(1))])
    for n in range(1, N + 1):
        term = term.multiply(src).scale(Q(1, n))
        total = total + term
    return total

for t, r, k, eps in [(Q(1, 10), 2, 1, "1e-3"), (Q(-1, 3), 2, 1, "1e-2"), (Q(1, 10), 2, 2, "1e-2"), (Q(1, 20), 3, 1, "1e-2")]:
    t0 = time.time()
    X = CutLineExponential(t)
    cert = X.approximation_certificate(r, k, eps)
    N = cert.truncation_degree
    ref = series(t, r, N + 5)
    diff = (ref - cert.approximation.histogram).norm(k)
    ok = diff <= cert.approximation.error <= Q(eps)
    var = ref.norm(0)
    print(f"t={t} r={r} k={k} eps={eps} N={N} err={float(cert.approximation.error):.3e} diff5={float(diff):.3e} ok={ok} var={float(var):.8f} exp4rt={float(mpmath.e**(4*r*abs(t))):.8f} norm_k={float(ref.norm(k)):.4g} nb={float(X.norm_bound(r,k)):.4g} {time.time()-t0:.1f}s", flush=True)
