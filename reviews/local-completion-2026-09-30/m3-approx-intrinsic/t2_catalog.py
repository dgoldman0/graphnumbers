import sys, networkx as nx, indep
sys.path.insert(0, sys.argv[1])
import reconstruct_local as RL
# catalog completeness vs atlas
for maxn in range(1, 7):
    for maxdeg in (None, 0, 1, 2, 3, 4):
        mine = indep.connected_graphs(maxn, maxdeg)
        theirs = RL.catalog(maxn, maxdeg)
        if len(mine) != len(theirs):
            print("MISMATCH", maxn, maxdeg, len(mine), len(theirs))
print("catalog sizes checked")
# also check pairwise non-isomorphism & membership
theirs = RL.catalog(6)
G2 = [nx.from_dict_of_lists({u: g.neighbors(u) for u in range(g.n)}) for g in theirs]
for g in G2: g.add_nodes_from(range(len(g)))
mine = indep.connected_graphs(6)
used = set()
for h in mine:
    hits = [i for i, g in enumerate(G2) if g.number_of_nodes()==h.number_of_nodes() and g.number_of_edges()==h.number_of_edges() and nx.is_isomorphic(g, h)]
    assert len(hits) == 1, hits
    used.add(hits[0])
print("bijection ok", len(used), len(G2))
