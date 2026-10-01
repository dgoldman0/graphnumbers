import random
import networkx as nx
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

