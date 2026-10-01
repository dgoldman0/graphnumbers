import random, itertools, sys
sys.path.insert(0, '.')
from glib import *
import networkx as nx
from networkx.algorithms import isomorphism as iso
rng = random.Random(1)
def to_nx(g, root=None):
    G = nx.Graph(); G.add_nodes_from(g)
    for u,v in edges_of(g): G.add_edge(u,v)
    for v in G: G.nodes[v]['r'] = (v==root)
    return G
bad = 0; tested = 0
for trial in range(3000):
    n = rng.randint(1, 8)
    p = rng.random()
    g = mk(n, [(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<p])
    # random relabel
    perm = list(range(n)); rng.shuffle(perm)
    h = {perm[v]: frozenset(perm[w] for w in g[v]) for v in g}
    root = rng.randrange(n)
    assert canon(g, root) == canon(h, perm[root])
    # compare with a different random graph
    g2 = mk(n, [(i,j) for i in range(n) for j in range(i+1,n) if rng.random()<p])
    root2 = rng.randrange(n)
    same = canon(g, root) == canon(g2, root2)
    nm = iso.categorical_node_match('r', False)
    truth = nx.is_isomorphic(to_nx(g, root), to_nx(g2, root2), node_match=nm)
    tested += 1
    if same != truth: bad += 1
print('canon mismatches', bad, 'of', tested)
# inj_rooted vs brute force
def brute_inj(F, u, G, o):
    nF = len(F); cnt = 0
    others = [x for x in F if x != u]
    for img in itertools.permutations([y for y in G if y != o], len(others)):
        m = dict(zip(others, img)); m[u] = o
        if all(m[b] in G[m[a]] for a in F for b in F[a]):
            cnt += 1
    return cnt
bad = 0
for trial in range(400):
    nF = rng.randint(1, 4); nG = rng.randint(1, 7)
    while True:
        F = mk(nF, [(i,j) for i in range(nF) for j in range(i+1,nF) if rng.random()<0.6])
        if is_connected(F): break
    G = mk(nG, [(i,j) for i in range(nG) for j in range(i+1,nG) if rng.random()<0.5])
    u = rng.randrange(nF); o = rng.randrange(nG)
    if inj_rooted(F,u,G,o) != brute_inj(F,u,G,o): bad += 1
print('inj mismatches', bad)
