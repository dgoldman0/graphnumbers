import itertools, random, time
import networkx as nx
from gcore import *
from bcheck import check

rng = random.Random(12345)

def random_block(rng):
    kind = rng.choice(["K1", "cycle", "complete", "random", "tree", "K1", "edge"])
    if kind == "K1": return nx.empty_graph(1)
    if kind == "edge": return nx.path_graph(2)
    if kind == "cycle": return nx.cycle_graph(rng.randint(3, 5))
    if kind == "complete": return nx.complete_graph(rng.randint(3, 4))
    if kind == "tree": return nx.random_labeled_tree(rng.randint(2, 4), seed=rng.randint(0, 10**6)) if hasattr(nx, 'random_labeled_tree') else nx.path_graph(3)
    while True:
        g = nx.gnp_random_graph(rng.randint(3, 5), 0.6, seed=rng.randint(0, 10**6))
        if nx.is_connected(g): return g

def block_tree_graph(rng, nblocks):
    """Blocks joined by bridges along a random tree; returns G and list of bridges."""
    G = nx.Graph(); offs = []; base = 0
    for i in range(nblocks):
        B = random_block(rng)
        mapping = {v: base + j for j, v in enumerate(B.nodes())}
        G.add_nodes_from(mapping.values())
        G.add_edges_from((mapping[u], mapping[v]) for u, v in B.edges())
        offs.append(list(mapping.values())); base += B.number_of_nodes()
    bridges = []
    for i in range(1, nblocks):
        j = rng.randrange(i)
        u = rng.choice(offs[i]); v = rng.choice(offs[j])
        G.add_edge(u, v); bridges.append((u, v))
    return G, bridges

t0 = time.time(); count = 0; proper = 0; withcycles = 0
for trial in range(400):
    nb = rng.randint(2, 7)
    G, br = block_tree_graph(rng, nb)
    allbridges = list(nx.bridges(G))
    # choose random nonempty subset of all bridges (including bridges inside blocks)
    m = rng.randint(1, min(len(allbridges), 6))
    F = rng.sample(allbridges, m)
    is_tree, L, _ = quotient_leaf_edges(G, F)
    assert is_tree
    radii = [1, 2, 3] if G.number_of_nodes() <= 18 else [1]
    ok = check(G, F, radii, f"trial {trial}")
    assert ok
    count += 1; proper += len(L) < len(F); withcycles += (not nx.is_forest(G))
print("random block-tree graphs OK:", count, "cases;", proper, "proper reductions;", withcycles, "with cycles; time", round(time.time()-t0,1))

# random larger trees with random subsets
for trial in range(150):
    n = rng.randint(9, 13)
    T = nx.random_labeled_tree(n, seed=rng.randint(0, 10**6))
    E = list(T.edges())
    m = rng.randint(2, min(7, len(E)))
    F = rng.sample(E, m)
    assert check(T, F, [1, 2], f"tree trial {trial}")
print("random larger trees OK")
