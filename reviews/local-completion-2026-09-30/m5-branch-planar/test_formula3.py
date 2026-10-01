import itertools, random
import networkx as nx
from collections import defaultdict
from gcore import REG, interaction, component_vector, clean, quotient_leaf_edges
from test_bridge_blocks_lib import block_tree_graph

def formula3(G, F):
    H = G.copy(); H.remove_edges_from(F)
    blocks = [frozenset(c) for c in nx.connected_components(H)]
    lab = {v: i for i, b in enumerate(blocks) for v in b}
    deg = defaultdict(int)
    for u, v in F: deg[lab[u]] += 1; deg[lab[v]] += 1
    leaves = [i for i in range(len(blocks)) if deg[i] == 1]
    k = len(F); acc = defaultdict(int)
    for m in range(len(leaves) + 1):
        for J in itertools.combinations(leaves, m):
            keep = set().union(*(blocks[i] for i in range(len(blocks)) if i not in J))
            K = G.subgraph(keep)
            assert nx.is_connected(K)
            acc[REG.key(K)] += (-1) ** (k - m)
    return clean(acc)

rng = random.Random(11); n = 0
for trial in range(300):
    G, _ = block_tree_graph(rng, rng.randint(3, 7))
    br = list(nx.bridges(G))
    if len(br) < 2: continue
    F = rng.sample(br, rng.randint(2, min(6, len(br))))
    assert interaction(G, F, component_vector) == formula3(G, F)
    n += 1
for nv in range(3, 9):
    for T in nx.nonisomorphic_trees(nv):
        E = list(T.edges())
        for m in range(2, len(E) + 1):
            for F in itertools.combinations(E, m):
                assert interaction(T, list(F), component_vector) == formula3(T, list(F)); n += 1
print("formula (3) verified on", n, "cases (k>=2)")
