from fractions import Fraction as Q
from math import comb, factorial
import mpmath as mp
from graphlocal import Finite, path, exp, star, graph
from graphlocal.elements import exp_bracket
mp.mp.dps = 80

def mpq(x): return mp.mpf(x.numerator) / x.denominator

# exp_bracket vs mpmath
bad = 0
for x in [Q(0), Q(1, 1000), Q(1, 3), Q(1), Q(5, 2), Q(7), Q(20), Q(50), Q(123, 2)]:
    lo, hi = exp_bracket(x)
    v = mp.e ** mpq(x)
    if not (mpq(lo) <= v <= mpq(hi)):
        bad += 1; print("exp_bracket FAIL", x)
print("exp_bracket ok" if not bad else "exp_bracket bad")

def star_coeffs_exp_sP3(s, shift=Q(0), N=200):
    """True T_1 coefficients of exp(s*P3 - shift*K1) by star size m (degree m)."""
    out = {}
    for n in range(N):
        for a in range(n + 1):
            m = n + a
            out[m] = out.get(m, mp.mpf(0)) + mpq(s) ** n / factorial(n) * comb(n, a) * 2 ** (n - a)
    f = mp.e ** (-mpq(shift))
    return {m: v * f for m, v in out.items()}

def check(s, shift, k, eps):
    X = Finite.from_graph(path(3)) * s - Finite.scalar(shift)
    approx = exp(X, max_terms=400).approximate(1, k, eps)
    truth = star_coeffs_exp_sP3(s, shift)
    got = {}
    for key, c in approx.histogram.values.items():
        g = key.graph
        deg = g.rows[0].bit_count()
        assert g.n == deg + 1 and g.edges == deg, "non-star type"
        got[deg] = c
    # norm of difference over degrees present in truth up to a cut where tail is negligible
    diff = mp.mpf(0)
    for m in set(truth) | set(got):
        diff += abs(truth.get(m, 0) - mpq(got.get(m, Q(0)))) * (m + 1) ** k
    ok = diff <= mpq(approx.error) * (1 + mp.mpf(10) ** -30)
    print(f"s={s} shift={shift} k={k} eps={eps}: true_err={mp.nstr(diff, 8)} claimed={mp.nstr(mpq(approx.error), 8)} {'OK' if ok else 'FAIL'}")

for s, shift in [(Q(1, 10), Q(0)), (Q(1, 2), Q(0)), (Q(1), Q(0)), (Q(-1, 2), Q(1)), (Q(1, 3), Q(-2))]:
    for k in (1, 2, 3):
        for eps in ("1e-3", "1e-9"):
            check(s, shift, k, eps)
