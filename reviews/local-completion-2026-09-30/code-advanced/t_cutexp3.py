from fractions import Fraction as Q
from math import factorial
from itertools import combinations_with_replacement
from collections import Counter
import mpmath
from graphlocal import *


def atoms(r):  # (left arm, right arm, coefficient) of T_r(E)
    return [(j, r, Q(2)) for j in range(r)] + [(r, r, Q(-2 * r))]


def ball_size(arms, r):
    poly = [1] + [0] * r
    for a, b in arms:
        f = [0] * (r + 1)
        for x in range(-a, b + 1):
            if abs(x) <= r:
                f[abs(x)] += 1
        new = [0] * (r + 1)
        for i, p in enumerate(poly):
            if p:
                for j, c in enumerate(f):
                    if i + j <= r:
                        new[i + j] += p * c
        poly = new
    return sum(poly)


_cache = {}


def power_norm(r, n, k):
    key = (r, n, k)
    if key in _cache:
        return _cache[key]
    at = atoms(r)
    total = Q(0)
    for ms in combinations_with_replacement(range(len(at)), n):
        cnt = Counter(ms)
        mult = factorial(n)
        for v in cnt.values():
            mult //= factorial(v)
        coef = Q(mult)
        for i in ms:
            coef *= abs(at[i][2])
        total += coef * ball_size([at[i][:2] for i in ms], r) ** k
    _cache[key] = total
    return total


if __name__ == "__main__":
    # validate the independent formula against library histograms (r>=2: no cancellation)
    for r in (2, 3):
        E = CutLineDefect().local(r)
        p = LocalHistogram(r, [(graph(1), Q(1))])
        for n in range(0, 4 if r < 3 else 3):
            for k in (0, 1, 2):
                assert p.norm(k) == power_norm(r, n, k), (r, n, k, p.norm(k), power_norm(r, n, k))
            p = p.multiply(E)
    print("independent power norms agree with library histograms (r=2,3)", flush=True)

    bad = 0
    for r in (2, 3, 4):
        for k in (1, 2, 3):
            for t in (Q(1, 50), Q(1, 8), Q(1, 2)):
                x = 4 * r * t
                nmax = 40
                norms = [power_norm(r, n, k) * t ** n / factorial(n) for n in range(nmax)]
                for N in range(0, 25):
                    exact_tail = sum(norms[N + 1:])
                    majorant = (r + 1) ** k * sum(Q(1 + 2 * n) ** (r * k) * x ** n / factorial(n)
                                                  for n in range(N + 1, 150))
                    if exact_tail > majorant:
                        bad += 1
                        print("VIOLATION", r, k, t, N, float(exact_tail), float(majorant))
                if k == 1:
                    var = sum(power_norm(r, n, 0) * t ** n / factorial(n) for n in range(nmax))
                    print(f"r={r} t={t} variation~{float(var):.10f} exp(4rt)={float(mpmath.e ** (4 * r * t)):.10f}",
                          flush=True)
    print("violations", bad)
