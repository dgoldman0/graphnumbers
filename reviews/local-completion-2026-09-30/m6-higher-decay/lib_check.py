import random
from fractions import Fraction as Q
from math import factorial
import networkx as nx
from core import brute_moments, norm_edge
from graphlocal import EdgeInteraction, graph
from graphlocal.interaction_moments import interaction_moments, tree_interaction_leading
from graphlocal.interaction_bounds import interaction_geometry
random.seed(99)
bad = 0; tests = 0
for trial in range(250):
    n = random.randint(2, 8)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    mode = random.random()
    if mode < 0.3:
        T = nx.random_labeled_tree(n, seed=random.randint(0, 10**6)) if n > 1 else None
        edges = sorted(norm_edge(e) for e in T.edges())
    else:
        edges = [e for e in pairs if random.random() < 0.45]
    k = random.randint(1, min(4, len(pairs)))
    chosen = random.sample(pairs, k)
    kind = random.choice(["del", "mixed"])
    edits = []
    for u, v in chosen:
        if (u, v) in edges: edits.append((u, v, -1))
        elif kind == "mixed": edits.append((u, v, 1))
    if not edits: continue
    g = graph(n, edges)
    order = 9
    bf = brute_moments(n, edges, edits, order)
    for red in (False, True):
        val = EdgeInteraction(g, edits, reduce_bridges=red)
        im = interaction_moments(val, order)
        tests += 1
        if list(im.laplacian) != [Q(x) for x in bf]:
            bad += 1; print("MOMENT MISMATCH", red, n, edges, edits, im.laplacian, bf)
    # tree leading
    if mode < 0.3 and all(s == -1 for *_, s in edits):
        val = EdgeInteraction(g, edits)
        lead = tree_interaction_leading(val)
        I = brute_moments(n, edges, edits, max(lead.order, 1) + 1)
        first = next((j for j, x in enumerate(I) if x), None)
        c = Q((-1)**first * I[first], factorial(first))
        if first != lead.order or c != lead.heat_coefficient:
            bad += 1; print("TREE MISMATCH", n, edges, edits, first, c, lead)
print("tests", tests, "bad", bad)
