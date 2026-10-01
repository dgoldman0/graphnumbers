from fractions import Fraction as Q
import random, itertools
import networkx as nx
from common import *
from graphlocal import SparseEdgeDifference
random.seed(9)
bad = 0
for trial in range(120):
    n = random.randint(4, 22)
    G = random.choice([nx.cycle_graph(n), nx.gnp_random_graph(n, 3.0 / n, seed=random.randint(0, 10**9)),
                       nx.random_labeled_tree(n, seed=random.randint(0, 10**9)) if hasattr(nx, "random_labeled_tree") else nx.random_tree(n, seed=1)])
    G = nx.Graph(G)
    g = from_nx(G)
    pairs = list(itertools.combinations(range(n), 2))
    chosen = random.sample(pairs, random.randint(1, 4))
    edits = [(u, v, -1 if G.has_edge(u, v) else 1) for u, v in chosen]
    H = G.copy()
    for u, v, s in edits:
        (H.remove_edge if s < 0 else H.add_edge)(u, v)
    normalize = random.random() < 0.3
    X = SparseEdgeDifference(g, edits, normalize)
    s = Q(1, n) if normalize else Q(1)
    for r in range(0, 4):
        oracle = merge_hists(rooted_hist(H, r, s), rooted_hist(G, r, -s))
        ok, d = lib_vs_oracle(X.local(r), oracle)
        if not ok:
            bad += 1; print("FAIL", trial, r)
print("SED local bad:", bad)
