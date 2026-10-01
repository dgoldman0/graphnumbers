import itertools
import networkx as nx
from collections import defaultdict
from gcore import Registry, rooted_ball, clean, interaction, histogram

reg = Registry()
def spider(arm):
    G = nx.Graph(); G.add_node(0); c = 1
    for a in range(3):
        prev = 0
        for i in range(arm):
            G.add_edge(prev, c); prev = c; c += 1
    return G
def hist(G, r):
    out = defaultdict(int)
    for o in G.nodes():
        out[reg.key(rooted_ball(G, o, r), o)] += 1
    return out
for r in (1, 2, 3):
    prev = None
    for arm in range(1, 2 * r + 4):
        G = spider(arm); F = [(0, n) for n in G.neighbors(0)]
        h = interaction(G, F, lambda H: hist(H, r))
        var = sum(abs(v) for v in h.values())
        stab = (h == prev)
        prev = h
        print(f"r={r} arm={arm}: variation={var}, types={len(h)}, equal to previous arm: {stab}")
