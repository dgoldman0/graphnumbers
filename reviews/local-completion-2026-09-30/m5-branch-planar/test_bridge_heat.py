import itertools, random
import numpy as np, networkx as nx
from gcore import quotient_leaf_edges
from test_bridge_blocks_lib import block_tree_graph

def heat(G, t):
    ev = np.linalg.eigvalsh(nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(float))
    return np.exp(-t * ev).sum()
def inter(G, F, t):
    k = len(F); s = 0.0
    for m in range(k + 1):
        for S in itertools.combinations(F, m):
            H = G.copy(); H.remove_edges_from(S); s += (-1) ** (k - m) * heat(H, t)
    return s
rng = random.Random(21); worst = 0; cnt = 0; proper = 0
for trial in range(120):
    G, _ = block_tree_graph(rng, rng.randint(3, 7))
    br = list(nx.bridges(G))
    if len(br) < 2: continue
    F = rng.sample(br, rng.randint(2, min(7, len(br))))
    _, L, _ = quotient_leaf_edges(G, F)
    sign = (-1) ** (len(F) - len(L)); proper += len(L) < len(F)
    for t in (0.3, 1.0, 3.0):
        a, b = inter(G, F, t), sign * inter(G, L, t)
        worst = max(worst, abs(a - b) / (1 + abs(a))); cnt += 1
print("heat-trace check of (2):", cnt, "evaluations,", proper, "proper reductions; max rel. discrepancy", worst)
