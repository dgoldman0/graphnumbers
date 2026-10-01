from fractions import Fraction as Q
from math import factorial
from itertools import combinations_with_replacement
from collections import Counter
import mpmath
from graphlocal import *
from t_cutexp3 import atoms, ball_size

def power_norms(r, n, ks):
    at = atoms(r)
    tot = {k: Q(0) for k in ks}
    for ms in combinations_with_replacement(range(len(at)), n):
        cnt = Counter(ms)
        mult = factorial(n)
        for v in cnt.values(): mult //= factorial(v)
        coef = Q(mult)
        for i in ms: coef *= abs(at[i][2])
        b = ball_size([at[i][:2] for i in ms], r)
        for k in ks: tot[k] += coef * b ** k
    return tot

ks = (0, 1, 2, 3)
bad = 0
for r, nmax in ((2, 34), (3, 26), (4, 18)):
    norms = [power_norms(r, n, ks) for n in range(nmax)]
    for t in (Q(1, 50), Q(1, 8), Q(1, 2)):
        x = 4 * r * t
        var = sum(norms[n][0] * t ** n / factorial(n) for n in range(nmax))
        print(f"r={r} t={t} variation(partial,{nmax} terms)={float(var):.12f} exp(4rt)={float(mpmath.e ** (4*r*t)):.12f}", flush=True)
        for k in (1, 2, 3):
            terms = [norms[n][k] * t ** n / factorial(n) for n in range(nmax)]
            for N in range(0, nmax - 8):
                exact_tail_lower = sum(terms[N + 1:])   # partial (lower bound of the exact tail)
                majorant = (r + 1) ** k * sum(Q(1 + 2 * n) ** (r * k) * x ** n / factorial(n) for n in range(N + 1, 160))
                if exact_tail_lower > majorant:
                    bad += 1; print("VIOLATION", r, k, t, N)
                # termwise check: each power's exact weighted norm vs its majorant term
            for n in range(nmax):
                if norms[n][k] > (r + 1) ** k * (1 + 2 * n) ** (r * k) * (4 * r) ** n:
                    bad += 1; print("TERM VIOLATION", r, k, n)
print("violations", bad)
