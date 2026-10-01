from fractions import Fraction as Q
import random, itertools
import networkx as nx
from common import *
from graphlocal import *
random.seed(4)
def keys_distinct(hist, rooted=True):
    ks = []
    for key in (hist.values if rooted else [IsoGraph for IsoGraph in hist]):
        H = to_nx(key.graph)
        nx.set_node_attributes(H, {u: int(u == 0) for u in H}, "r")
        ks.append(H)
    for A, B in itertools.combinations(ks, 2):
        if A.number_of_nodes() == B.number_of_nodes() and nx.is_isomorphic(A, B, node_match=lambda x, y: x["r"] == y["r"]):
            return False
    return True
bad = 0; n_hist = 0
for trial in range(300):
    n = random.randint(2, 12)
    G = nx.gnp_random_graph(n, random.uniform(0.2, 0.8), seed=random.randint(0, 10**9))
    if random.random() < 0.3:
        G = nx.random_regular_graph(random.choice([2, 3, 4]), 10, seed=random.randint(0, 10**9))
    X = Finite.from_graph(from_nx(G))
    for r in range(0, 4):
        h = X.local(r); n_hist += 1
        if not keys_distinct(h): bad += 1; print("split", trial, r)
for r in range(0, 4):
    for X in [Line() * Line(), CutLineDefect() * Line(), CutLineDefect() * CutLineDefect(), LineCutDefect([0, 1, 3]) * Line(),
              Finite.from_graph(cycle(5)) * Finite.from_graph(path(3)) * Line()]:
        h = X.local(r); n_hist += 1
        if not keys_distinct(h): bad += 1; print("split special", r)
# Finite unrooted terms distinct
for trial in range(200):
    gs = [from_nx(nx.gnp_random_graph(random.randint(1, 7), 0.5, seed=random.randint(0, 10**9))) for _ in range(5)]
    F = Finite([(Q(random.randint(-3, 3)), g) for g in gs])
    ks = [to_nx(g) for _, g in F.terms]
    for A, B in itertools.combinations(ks, 2):
        if nx.is_isomorphic(A, B): bad += 1; print("Finite split", trial)
print("histograms checked", n_hist, "bad", bad)
