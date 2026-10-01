import itertools, random
import networkx as nx
from collections import defaultdict
from gcore import *

def formula9(G, F):
    H = G.copy(); H.remove_edges_from(F)
    blocks = [frozenset(c) for c in nx.connected_components(H)]
    lab = {v: i for i, b in enumerate(blocks) for v in b}
    nb = len(blocks)
    acc = defaultdict(int)
    for size in range(1, nb + 1):
        for U in itertools.combinations(range(nb), size):
            Us = set(U)
            ext = [e for e in F if lab[e[0]] not in Us and lab[e[1]] not in Us]
            if ext: continue
            Fint = [e for e in F if lab[e[0]] in Us and lab[e[1]] in Us]
            verts = set().union(*(blocks[i] for i in U))
            base = H.subgraph(verts).copy()
            for a in range(len(Fint) + 1):
                for A in itertools.combinations(Fint, a):
                    K = base.copy(); K.add_edges_from(A)
                    if nx.is_connected(K):
                        acc[REG.key(K)] += (-1) ** a
    return clean(acc)

rng = random.Random(7)
n_ok = 0; n_cyc = 0
for trial in range(600):
    n = rng.randint(3, 7)
    while True:
        G = nx.gnp_random_graph(n, rng.uniform(0.35, 0.9), seed=rng.randint(0, 10**6))
        if nx.is_connected(G): break
    E = list(G.edges())
    F = rng.sample(E, rng.randint(1, min(len(E), 6)))
    lhs = interaction(G, F, component_vector)
    rhs = formula9(G, F)
    assert lhs == rhs, (list(G.edges()), F)
    n_ok += 1
    is_tree, L, _ = quotient_leaf_edges(G, F)
    n_cyc += (not is_tree)
print("formula (9) verified on", n_ok, "random cases;", n_cyc, "with non-tree quotients")

# triangle example and C_F != leaf-only
G = nx.cycle_graph(3)
F = list(G.edges())
v = interaction(G, F, component_vector)
names = {REG.key(nx.cycle_graph(3)): 'C3', REG.key(nx.path_graph(3)): 'P3', REG.key(nx.path_graph(2)): 'K2', REG.key(nx.empty_graph(1)): 'K1'}
print("triangle C_F:", {names.get(k, k): c for k, c in v.items()})
