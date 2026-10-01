import random, networkx as nx, refl
from collections import Counter
rng = random.Random(11)
def Dtri(G):
    c = Counter()
    tri = nx.triangles(G)
    for v in G.nodes: c[G.degree(v)] += (-1) ** tri[v]
    return {k: v for k, v in c.items() if v}
def pmul(a, b):
    c = Counter()
    for i, x in a.items():
        for j, y in b.items(): c[i + j] += x * y
    return {k: v for k, v in c.items() if v}
bad = 0
for _ in range(200):
    G = nx.gnp_random_graph(rng.randint(1, 7), rng.random(), seed=rng.randint(0, 10**9))
    H = nx.gnp_random_graph(rng.randint(1, 7), rng.random(), seed=rng.randint(0, 10**9))
    if Dtri(refl.cart(G, H)) != pmul(Dtri(G), Dtri(H)): bad += 1
print("D_triangle multiplicativity failures:", bad)
print("D(K3), Dtri(K3):", {2: 3}, Dtri(nx.complete_graph(3)))
