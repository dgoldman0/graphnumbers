"""Independent referee code: exact mixed moments of edge-edit interactions.

Nothing here imports the repo package. All arithmetic is exact (Python ints /
Fractions) unless explicitly noted.
"""
from fractions import Fraction as Q
from itertools import product, combinations, permutations
from math import comb, factorial


def lap(n, edges):
    M = [[0] * n for _ in range(n)]
    for u, v in edges:
        M[u][u] += 1
        M[v][v] += 1
        M[u][v] -= 1
        M[v][u] -= 1
    return M


def matmul(A, B):
    n = len(A)
    m = len(B[0])
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def trace_powers(M, order):
    n = len(M)
    P = [[int(i == j) for j in range(n)] for i in range(n)]
    out = []
    for _ in range(order + 1):
        out.append(sum(P[i][i] for i in range(n)))
        P = matmul(P, M)
    return out


def norm_edge(e):
    u, v = e
    return (min(u, v), max(u, v))


def brute_moments(n, edges, edits, order):
    """edits: list of (u,v,sign) with sign=-1 deletion (edge present), +1 insertion.
    Returns I_0..I_order with I_n = sum_S (-1)^{k-|S|} Tr L(G_S)^n."""
    base = set(norm_edge(e) for e in edges)
    k = len(edits)
    res = [0] * (order + 1)
    for mask in range(1 << k):
        E = set(base)
        for i, (u, v, s) in enumerate(edits):
            if mask >> i & 1:
                e = norm_edge((u, v))
                if s == -1:
                    assert e in E
                    E.remove(e)
                else:
                    assert e not in E
                    E.add(e)
        tr = trace_powers(lap(n, sorted(E)), order)
        sign = (-1) ** (k - bin(mask).count("1"))
        for j in range(order + 1):
            res[j] += sign * tr[j]
    return res


def incidence(n, e):
    u, v = e
    b = [0] * n
    b[u] = 1
    b[v] = -1
    return b


def cross_moments(n, edges, edits, order):
    """c[a][i][j] = b_i^T L^a b_j, L of the ORIGINAL graph, a=0..order."""
    L = lap(n, [norm_edge(e) for e in edges])
    bs = [incidence(n, (u, v)) for u, v, _ in edits]
    k = len(bs)
    out = []
    vecs = [b[:] for b in bs]  # L^a b_j
    for a in range(order + 1):
        out.append([[sum(x * y for x, y in zip(bs[i], vecs[j])) for j in range(k)] for i in range(k)])
        vecs = [[sum(L[r][c] * v[c] for c in range(n)) for r in range(n)] for v in vecs]
    return out


def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def cyclic_formula_bruteforce(n, edges, edits, order):
    """Formula (1) of HIGHER_INTERACTION_GEOMETRY by literal enumeration of
    label words and gap compositions. sigma_i = edit sign."""
    k = len(edits)
    c = cross_moments(n, edges, edits, order)
    sig = [s for _, _, s in edits]
    res = [Q(0)] * (order + 1)
    for N in range(1, order + 1):
        tot = Q(0)
        for m in range(k, N + 1):
            sub = 0
            for labels in product(range(k), repeat=m):
                if len(set(labels)) != k:
                    continue
                sg = 1
                for l in labels:
                    sg *= sig[l]
                for gaps in compositions(N - m, m):
                    p = 1
                    for j in range(m):
                        p *= c[gaps[j]][labels[j]][labels[(j + 1) % m]]
                        if p == 0:
                            break
                    sub += sg * p
            tot += Q(N, m) * sub
        res[N] = tot
    return res


def cyclic_formula_transfer(n, edges, edits, order, weights=None):
    """Same formula, computed by a transfer recursion over (mask,last,degree),
    written independently of the repo's DP. weights: optional list of rational
    multipliers x_i (perturbation L + sum x_i B_i); default sigma_i."""
    k = len(edits)
    c = cross_moments(n, edges, edits, order)
    x = weights if weights is not None else [s for _, _, s in edits]
    full = (1 << k) - 1
    res = [Q(0)] * (order + 1)
    # state: (first, mask, last, deg) where deg = letters + inner gaps so far
    for first in range(k):
        states = {(1 << first, first, 1): Q(x[first])}
        m = 1
        while states and m <= order:
            # close
            for (mask, last, deg), val in states.items():
                if mask == full:
                    for a in range(0, order - deg + 1):
                        N = deg + a
                        res[N] += Q(N, m) * val * c[a][last][first]
            # extend
            new = {}
            for (mask, last, deg), val in states.items():
                for nxt in range(k):
                    for a in range(0, order - deg):
                        cc = c[a][last][nxt]
                        if cc == 0:
                            continue
                        key = (mask | (1 << nxt), nxt, deg + a + 1)
                        new[key] = new.get(key, Q(0)) + val * cc * x[nxt]
            states = {kk: v for kk, v in new.items() if v != 0}
            m += 1
    return res


def weighted_brute(n, edges, edits, order, weights):
    """sum over S of (-1)^{k-|S|} Tr (L + sum_{i in S} x_i B_i)^n with rational x."""
    L = lap(n, [norm_edge(e) for e in edges])
    k = len(edits)
    res = [Q(0)] * (order + 1)
    for mask in range(1 << k):
        M = [[Q(L[i][j]) for j in range(n)] for i in range(n)]
        for i, (u, v, _) in enumerate(edits):
            if mask >> i & 1:
                b = incidence(n, (u, v))
                for r in range(n):
                    for cc in range(n):
                        M[r][cc] += weights[i] * b[r] * b[cc]
        tr = trace_powers(M, order)
        sign = (-1) ** (k - bin(mask).count("1"))
        for j in range(order + 1):
            res[j] += sign * tr[j]
    return res
