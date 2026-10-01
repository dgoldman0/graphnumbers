import random, itertools
import networkx as nx
from common import *
from graphlocal.graphs import isomorphic, graph, cartesian, complete, cycle

random.seed(1)
bad = []
budget = 0
def check(a, b, rooted):
    global budget
    try:
        got = isomorphic(a, b, rooted)
    except BudgetExceeded:
        budget += 1
        return
    exp = nx_iso(a, b, rooted)
    if got != exp:
        bad.append((a.rows, b.rows, rooted, got, exp))

def relabel(g, perm):
    inv = {p: i for i, p in enumerate(perm)}
    return graph(g.n, [(inv[u], inv[v]) for u in range(g.n) for v in g.neighbors(u) if u < v])

# random small graphs, random relabelings and near-misses
for trial in range(3000):
    n = random.randint(1, 9)
    p = random.random()
    G = nx.gnp_random_graph(n, p, seed=random.randint(0, 10**9))
    a = from_nx(G)
    perm = list(range(n)); random.shuffle(perm)
    b = relabel(a, perm)
    check(a, b, False); check(a, b, True)
    # perm fixing root
    perm2 = [0] + random.sample(range(1, n), n - 1) if n > 1 else [0]
    b2 = relabel(a, perm2)
    check(a, b2, True)
    # degree-preserving double edge swap as near miss
    H = G.copy()
    if H.number_of_edges() >= 2:
        try:
            nx.double_edge_swap(H, nswap=1, max_tries=100, seed=random.randint(0, 10**9))
        except Exception:
            pass
    c = from_nx(H)
    check(a, c, False); check(a, c, True)
print("random done; bad:", len(bad), "budget:", budget)

# Tricky pairs
def rook(n):
    return nx.cartesian_product(nx.complete_graph(n), nx.complete_graph(n))
R = nx.convert_node_labels_to_integers(rook(4))

# Shrikhande graph: Cayley graph on Z4xZ4 with connection set {±(1,0), ±(0,1), ±(1,1)}
Sh = nx.Graph()
for x in range(4):
    for y in range(4):
        for dx, dy in [(1,0),(0,1),(1,1),(3,0),(0,3),(3,3)]:
            Sh.add_edge((x,y), ((x+dx)%4, (y+dy)%4))
Sh = nx.convert_node_labels_to_integers(Sh)
a, b = from_nx(R), from_nx(Sh)
print("rook vs shrikhande unrooted:", isomorphic(a, b), nx.is_isomorphic(R, Sh))
for r1 in range(16):
    for r2 in range(16):
        a1, b1 = from_nx(R, r1), from_nx(Sh, r2)
        check(a1, b1, True)
        check(a1, from_nx(R, r2), True)
        check(b1, from_nx(Sh, r1), True)
print("rook/shrikhande done; bad:", len(bad), "budget:", budget)

# Strongly regular / highly symmetric families
fams = [nx.petersen_graph(), nx.heawood_graph(), nx.pappus_graph(), nx.desargues_graph(), nx.dodecahedral_graph(),
        nx.hypercube_graph(4), nx.moebius_kantor_graph(), nx.truncated_tetrahedron_graph(), nx.circulant_graph(12, [1, 5]),
        nx.circulant_graph(12, [1, 3]), nx.circulant_graph(13, [1, 5]), nx.circulant_graph(13, [1, 3]),
        nx.paley_graph(13).to_undirected(), nx.complete_bipartite_graph(4, 4), nx.cycle_graph(12),
        nx.disjoint_union(nx.cycle_graph(6), nx.cycle_graph(6))]
fams = [nx.convert_node_labels_to_integers(nx.Graph(G)) for G in fams]
for G in fams:
    for H in fams:
        if G.number_of_nodes() != H.number_of_nodes():
            continue
        for r in range(0, G.number_of_nodes(), 3):
            for s in range(0, H.number_of_nodes(), 5):
                check(from_nx(G, r), from_nx(H, s), True)
        check(from_nx(G), from_nx(H), False)
# cycles 12 vs 2xC6, C_{2n} vs 2 C_n, regular same-degree graphs
for n in range(3, 9):
    check(from_nx(nx.cycle_graph(2 * n)), from_nx(nx.disjoint_union(nx.cycle_graph(n), nx.cycle_graph(n))), False)
# random regular pairs
for trial in range(400):
    n = random.choice([8, 10, 12])
    d = random.choice([3, 4])
    G = nx.random_regular_graph(d, n, seed=random.randint(0, 10**9))
    H = nx.random_regular_graph(d, n, seed=random.randint(0, 10**9))
    check(from_nx(G), from_nx(H), False)
    check(from_nx(G, 0), from_nx(H, random.randrange(n)), True)
print("families done; bad:", len(bad), "budget:", budget)
for x in bad[:10]:
    print(x)
