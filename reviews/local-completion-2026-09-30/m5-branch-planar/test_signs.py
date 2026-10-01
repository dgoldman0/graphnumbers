import itertools
import numpy as np
import networkx as nx

def heat(G, t):
    if G.number_of_nodes() == 0: return 0.0
    ev = np.linalg.eigvalsh(nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(float))
    return float(np.exp(-t * ev).sum())

def heat_interaction(G, F, t):
    k = len(F); tot = 0.0
    for m in range(k + 1):
        for S in itertools.combinations(F, m):
            H = G.copy(); H.remove_edges_from(S)
            tot += (-1) ** (k - m) * heat(H, t)
    return tot

ts = [0.01, 0.1, 0.5, 1, 2, 5, 10]
for k in range(2, 7):
    G = nx.star_graph(k); F = list(G.edges())
    vals = [heat_interaction(G, F, t) for t in ts]
    exact = [np.exp(-t) * (1 - np.exp(-t)) ** k for t in ts]
    print(f"star k={k}: max |num-(7)| = {max(abs(a-b) for a,b in zip(vals,exact)):.2e}; all positive: {all(v>0 for v in vals)}")
G = nx.cycle_graph(3); F = list(G.edges())
vals = [heat_interaction(G, F, t) for t in ts]
print("triangle: max |num-(10)| =", max(abs(a + (1-np.exp(-t))**3) for a, t in zip(vals, ts)), "all negative:", all(v < 0 for v in vals))
# three ordered cuts on a long path (approximating the line)
N = 81
for gaps in ((1, 1), (1, 2), (2, 3), (3, 3)):
    G = nx.path_graph(N); x0 = 40
    pos = [x0, x0 + gaps[0], x0 + gaps[0] + gaps[1]]
    F = [(p - 1, p) for p in pos]
    vals = [heat_interaction(G, F, t) for t in ts]
    print(f"path N={N} 3 cuts gaps {gaps}: heat interaction", ["%.3e" % v for v in vals])
