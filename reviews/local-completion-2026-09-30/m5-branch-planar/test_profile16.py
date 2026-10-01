import itertools, random
import numpy as np
import networkx as nx
from math import perm

rng = random.Random(3)
worst = 0.0; cases = 0
for trial in range(400):
    n = rng.randint(4, 9)
    G = nx.gnp_random_graph(n, rng.uniform(0.2, 0.8), seed=rng.randint(0, 10**6))
    allpairs = list(itertools.combinations(range(n), 2))
    rng.shuffle(allpairs)
    k = rng.randint(1, 4)
    groups = []; used = 0
    for i in range(k):
        q = rng.randint(1, 3)
        grp = allpairs[used:used + q]; used += q
        groups.append(grp)
    if used > len(allpairs): continue
    mixed = rng.random() < 0.5
    # an edit toggles presence: deletion if present, insertion if absent (only deletions when not mixed)
    if not mixed:
        groups = [[e for e in g if G.has_edge(*e)] for g in groups]
        if any(len(g) == 0 for g in groups): continue
    Gmax = G.copy()
    for g in groups:
        for e in g:
            if not G.has_edge(*e): Gmax.add_edge(*e)
    D = max(dict(Gmax.degree()).values())
    if D == 0: continue
    for Dp in (D, D + 1, D + 3):
        mats = []
        for m in range(k + 1):
            for S in itertools.combinations(range(k), m):
                H = G.copy()
                for i in S:
                    for e in groups[i]:
                        if H.has_edge(*e): H.remove_edge(*e)
                        else: H.add_edge(*e)
                L = nx.laplacian_matrix(H, nodelist=range(n)).toarray().astype(float)
                mats.append(((-1) ** (k - m), np.eye(n) - L / Dp))
        bound_c = 2 ** k * np.prod([len(g) for g in groups]) / Dp ** k
        for j in range(0, 16):
            dj = sum(s * np.trace(np.linalg.matrix_power(P, j)) for s, P in mats)
            b = bound_c * perm(j, k) if j >= k else 0.0
            ratio = abs(dj) / b if b > 0 else (0.0 if abs(dj) < 1e-9 else float('inf'))
            worst = max(worst, ratio)
        cases += 1
print("cases:", cases, "max |d_j| / bound(16) =", worst)
