import random
from fractions import Fraction as Q
from itertools import combinations
from graphlocal import *
random.seed(5)

def raw(before, edits):
    terms = []
    for size in range(len(edits) + 1):
        for S in combinations(edits, size):
            after, _ = apply_edge_edits(before, S)
            terms.append(((-1) ** (len(edits) - size), after))
    return Finite(terms)

def is_bridge(g, u, v):
    after, _ = apply_edge_edits(g, [(u, v, -1)])
    from graphlocal.graphs import distances
    return v not in distances(after, u)

checked = applied = 0
for trial in range(120):
    n = random.randint(3, 9)
    # random tree plus a few extra edges to create cycle blocks
    edges = {(min(i, j), max(i, j)) for i in range(1, n) for j in [random.randrange(i)]}
    for _ in range(random.randint(0, 3)):
        u, v = random.sample(range(n), 2)
        edges.add((min(u, v), max(u, v)))
    g = graph(n, sorted(edges))
    bridges = [(u, v) for u, v in sorted(edges) if is_bridge(g, u, v)]
    if len(bridges) < 2: continue
    k = random.randint(2, min(5, len(bridges)))
    sel = [(u, v, -1) for u, v in random.sample(bridges, k)]
    x = EdgeInteraction(g, sel)
    ref = raw(g, sel)
    checked += 1
    applied += x.bridge_reduction_applies
    assert x.finite() == ref, (g.rows, sel)
    for r in (0, 1, 2, 3):
        assert x.local(r) == ref.local(r), (g.rows, sel, r)
    # metadata vs explicit
    assert ref.local(3).norm(0) <= x.variation_bound
print("checked", checked, "reduction applied", applied)
