import sys, time
import numpy as np
import networkx as nx
from fractions import Fraction as Q
from math import factorial

def lap_np(n, edges):
    M = np.zeros((n, n), dtype=object)
    for u, v in edges:
        M[u, u] += 1; M[v, v] += 1; M[u, v] -= 1; M[v, u] -= 1
    return M

def traces(M, order):
    n = M.shape[0]
    P = np.eye(n, dtype=int).astype(object)
    out = []
    for _ in range(order + 1):
        out.append(int(sum(P[i, i] for i in range(n))))
        P = P.dot(M)
    return out

def minimal_subtree(T, cuts):
    # minimal subtree containing all selected edges: union of tree paths between endpoints
    verts = set(x for e in cuts for x in e)
    verts = sorted(verts)
    E = set()
    r = verts[0]
    for v in verts[1:]:
        p = nx.shortest_path(T, r, v)
        for a, b in zip(p, p[1:]):
            E.add((min(a, b), max(a, b)))
    for a, b in cuts:
        E.add((min(a, b), max(a, b)))
    return E

def predicted(T, cuts):
    k = len(cuts)
    if k == 1:
        return 1, Q(2)
    E = minimal_subtree(T, cuts)
    deg = {}
    for a, b in E:
        deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    s = len(E); ell = sum(1 for d in deg.values() if d == 1)
    p = 1
    for d in deg.values():
        if d >= 2: p *= factorial(d - 1)
    nu = 2 * s - ell
    return nu, Q((-1) ** (k - ell) * p, factorial(nu - 1))

maxn = int(sys.argv[1])
tot = 0; single = 0; bad = 0; pos = 0; neg = 0; ntrees = 0
t0 = time.time()
for n in range(2, maxn + 1):
    for T in nx.nonisomorphic_trees(n):
        ntrees += 1
        edges = sorted((min(a, b), max(a, b)) for a, b in T.edges())
        m = len(edges)
        order = 2 * m  # nu <= 2s - 2 <= 2m-2 ; use 2m to see beyond
        # all subset traces of G \ S
        table = []
        for mask in range(1 << m):
            kept = [e for i, e in enumerate(edges) if not (mask >> i & 1)]
            table.append(traces(lap_np(n, kept), order))
        # Mobius: mixed[mask] = sum_{S subset mask} (-1)^{|mask|-|S|} table[S]
        mixed = [list(r) for r in table]
        for bit in range(m):
            for mask in range(1 << m):
                if mask >> bit & 1:
                    lo = mixed[mask ^ (1 << bit)]
                    mixed[mask] = [a - b for a, b in zip(mixed[mask], lo)]
        for mask in range(1, 1 << m):
            cuts = [e for i, e in enumerate(edges) if mask >> i & 1]
            I = mixed[mask]
            nu, coef = predicted(T, cuts)
            first = next((j for j, x in enumerate(I) if x), None)
            lead = Q((-1) ** first * I[first], factorial(first)) if first is not None else None
            tot += 1
            if len(cuts) == 1: single += 1
            if first != nu or lead != coef:
                bad += 1
                print("MISMATCH", n, edges, cuts, first, lead, nu, coef)
            if coef > 0: pos += 1
            else: neg += 1
    print(f"n<={n}: trees {ntrees}, cutsets {tot}, single {single}, bad {bad}, pos {pos}, neg {neg}, {time.time()-t0:.1f}s", flush=True)
