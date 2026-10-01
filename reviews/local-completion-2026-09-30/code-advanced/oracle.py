"""Independent brute-force oracles: subset sums of exact matrix traces, mpmath heat."""
from fractions import Fraction as Q
from itertools import combinations
import mpmath
import numpy as np

def adj_from_rows(rows):
    n = len(rows)
    return [[1 if rows[u] >> v & 1 else 0 for v in range(n)] for u in range(n)]

def apply(rows, edits):
    rows = list(rows)
    for u, v, s in edits:
        rows[u] ^= 1 << v
        rows[v] ^= 1 << u
    return rows

def lap(rows):
    n = len(rows)
    A = adj_from_rows(rows)
    return [[(sum(A[u]) if u == v else -A[u][v]) for v in range(n)] for u in range(n)]

def matmul(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

def trace_powers(M, order):
    n = len(M)
    P = [[Q(int(i == j)) for j in range(n)] for i in range(n)]
    out = [sum(P[i][i] for i in range(n))]
    for _ in range(order):
        P = matmul(P, M)
        out.append(sum(P[i][i] for i in range(n)))
    return out

def subset_iter(edits):
    k = len(edits)
    for size in range(k + 1):
        for S in combinations(edits, size):
            yield (-1) ** (k - size), S

def laplacian_moments(rows, edits, order):
    res = [Q(0)] * (order + 1)
    for sign, S in subset_iter(edits):
        L = lap(apply(rows, S))
        L = [[Q(x) for x in row] for row in L]
        for j, m in enumerate(trace_powers(L, order)):
            res[j] += sign * m
    return res

def lazy_moments(rows, edits, D, order):
    res = [Q(0)] * (order + 1)
    for sign, S in subset_iter(edits):
        L = lap(apply(rows, S))
        n = len(L)
        P = [[Q(int(i == j)) - Q(L[i][j], D) for j in range(n)] for i in range(n)]
        for j, m in enumerate(trace_powers(P, order)):
            res[j] += sign * m
    return res

def heat(rows, edits, t, dps=60):
    """Exact-ish interaction heat sum_S (-1)^{k-|S|} Tr exp(-t L_S) using mpmath eigh."""
    mpmath.mp.dps = dps
    t = mpmath.mpf(t.numerator) / t.denominator if isinstance(t, Q) else mpmath.mpf(t)
    total = mpmath.mpf(0)
    for sign, S in subset_iter(edits):
        L = lap(apply(rows, S))
        M = mpmath.matrix(L)
        if len(L) == 0:
            continue
        E = mpmath.eigsy(M, eigvals_only=True)
        total += sign * sum(mpmath.exp(-t * e) for e in E)
    return total

def graph_heat(rows, t, dps=60):
    mpmath.mp.dps = dps
    t = mpmath.mpf(t.numerator) / t.denominator if isinstance(t, Q) else mpmath.mpf(t)
    L = lap(rows)
    if not L:
        return mpmath.mpf(0)
    E = mpmath.eigsy(mpmath.matrix(L), eigvals_only=True)
    return sum(mpmath.exp(-t * e) for e in E)

def mpq(x):
    return mpmath.mpf(x.numerator) / x.denominator
