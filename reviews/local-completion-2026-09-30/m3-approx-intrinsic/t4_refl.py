import random, networkx as nx, indep, refl
from fractions import Fraction as Fr
from collections import defaultdict
rng = random.Random(7)
def rand_conn(n, p):
    while True:
        G = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10**9))
        if nx.is_connected(G): return G
# 1. multiplicativity on random connected graphs up to 7 vertices
bad = 0
for _ in range(150):
    G = rand_conn(rng.randint(1, 6), rng.random()); H = rand_conn(rng.randint(1, 6), rng.random())
    lhs = refl.coeffs(refl.F([(1, refl.cart(G, H))]))
    rhs_terms = [(a * b, refl.cart(X, Y)) for a, X in refl.F([(1, G)]) for b, Y in refl.F([(1, H)])]
    if lhs != refl.coeffs(rhs_terms): bad += 1
print("F multiplicativity failures:", bad)
# 2. FR = id, FO = -id, F(P3+K1)=0, F(K2)=-K2
bad = 0
for _ in range(150):
    G = rand_conn(rng.randint(1, 8), rng.random())
    if refl.coeffs(refl.F([(1, refl.R_graph(G))])) != refl.coeffs([(1, G)]): bad += 1
    if refl.coeffs(refl.F([(1, refl.O_graph(G))])) != refl.coeffs([(-1, G)]): bad += 1
print("section failures:", bad)
print("F(P3+K1) =", refl.coeffs(refl.F([(1, nx.path_graph(3)), (1, nx.empty_graph(1))])))
print("F(K2) == -K2:", refl.coeffs(refl.F([(1, nx.path_graph(2))])) == refl.coeffs([(-1, nx.path_graph(2))]))
# 3. locality: (r+1)-ball determines (sign, r-ball of QG); r-ball determines R/O contributions
def check_locality(G, r):
    Q = refl.parity_filter(G)
    seen = {}
    for v in G.nodes:
        key = indep.REG.register(indep.ball(G, v, r + 1))
        val = ((-1) ** (G.degree(v) % 2), indep.REG.register(indep.ball(Q, v, r)))
        if seen.setdefault(key, val) != val: return False
    return True
def contributions(G, r, kind):
    H = nx.convert_node_labels_to_integers(G)
    n = H.number_of_nodes()
    X = refl.R_graph(H) if kind == 'R' else refl.O_graph(H)
    # map each new vertex to its original root
    owner = {v: v for v in range(n)}
    if kind == 'R':
        odd = [v for v in range(n) if H.degree(v) % 2]
        for i, v in enumerate(odd):
            owner[n + i] = v; owner[n + len(odd) + i] = v
    else:
        even = [v for v in range(n) if H.degree(v) % 2 == 0]
        for i, v in enumerate(even):
            owner[n + 2 * i] = v; owner[n + 2 * i + 1] = v
    per = defaultdict(list)
    for x in X.nodes:
        per[owner[x]].append(indep.REG.register(indep.ball(X, x, r)))
    return {v: tuple(sorted(per[v])) for v in range(n)}, X
def check_section_locality(G, r, kind):
    contrib, X = contributions(G, r, kind)
    seen = {}
    for v in G.nodes:
        key = indep.REG.register(indep.ball(G, v, r))
        if seen.setdefault(key, contrib[v]) != contrib[v]: return False
    return True
bad = defaultdict(int)
for _ in range(120):
    G = nx.convert_node_labels_to_integers(rand_conn(rng.randint(2, 10), rng.random() * 0.6 + 0.1))
    for r in (0, 1, 2, 3):
        if not check_locality(G, r): bad[('F', r)] += 1
        if r >= 1:
            for kind in 'RO':
                if not check_section_locality(G, r, kind): bad[(kind, r)] += 1
print("locality failures:", dict(bad))
# 4. weighted bounds on random signed inputs
def rand_terms():
    return [(Fr(rng.randint(-4, 4) or 1, rng.randint(1, 3)), rand_conn(rng.randint(1, 7), rng.random())) for _ in range(rng.randint(1, 4))]
bad = defaultdict(int); cnt = 0
for _ in range(80):
    X = rand_terms()
    FX = refl.F(X); RX = [(c, refl.R_graph(G)) for c, G in X]; OX = [(c, refl.O_graph(G)) for c, G in X]
    for r in (0, 1, 2, 3):
        for k in (1, 2, 3):
            cnt += 1
            if indep.wnorm(indep.lin_hist(FX, r), k) > indep.wnorm(indep.lin_hist(X, r + 1), k): bad[('F', r, k)] += 1
            if r >= 1:
                if indep.wnorm(indep.lin_hist(RX, r), k) > (2 ** (k + 1) + 1) * indep.wnorm(indep.lin_hist(X, r), k): bad[('R', r, k)] += 1
                if indep.wnorm(indep.lin_hist(OX, r), k) > 3 ** (k + 1) * indep.wnorm(indep.lin_hist(X, r), k): bad[('O', r, k)] += 1
print("bound checks", cnt, "failures:", dict(bad))
