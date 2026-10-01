import networkx as nx, indep, refl
from fractions import Fraction as Fr
reg = indep.Registry()
def U(G): return (Fr(1, G.number_of_nodes()), G)
def prism(n): return refl.cart(nx.path_graph(2), nx.cycle_graph(n))
def add_leaf(G, v=0):
    H = nx.convert_node_labels_to_integers(G); m = H.number_of_nodes(); H.add_edge(v, m); return H
# (6): p_{1,k}(Z_n) = (2^k+5^k+2*4^k)/(2n+1), Z_n = U(Q_n) - H U(C_n), H U(C_n) = U(K2 x C_n)
for n in (5, 6, 7, 9, 12):
    Qn = add_leaf(prism(n)); P = prism(n)
    h = indep.lin_hist([U(Qn), (-Fr(1, P.number_of_nodes()), P)], 1, reg)
    print(n, [indep.wnorm(h, k, reg) == Fr(2**k + 5**k + 2*4**k, 2*n+1) for k in (1, 2, 3, 4)])
# n = 3, 4 (outside stated range) for curiosity
for n in (3, 4):
    Qn = add_leaf(prism(n)); P = prism(n)
    h = indep.lin_hist([U(Qn), (-Fr(1, P.number_of_nodes()), P)], 1, reg)
    print("n=%d" % n, [indep.wnorm(h, k, reg) == Fr(2**k + 5**k + 2*4**k, 2*n+1) for k in (1, 2, 3)])
# filtration example
for r in range(1, 6):
    a, b = nx.cycle_graph(2*r+2), nx.cycle_graph(2*r+3)
    h0 = indep.lin_hist([U(a), (-Fr(1, b.number_of_nodes()), b)], r, reg)
    h1 = indep.lin_hist([U(a), (-Fr(1, b.number_of_nodes()), b)], r+1, reg)
    print("filtration r=%d" % r, h0 == {}, h1 != {})
# (2) and (3) bounds on random graphs
import random
rng = random.Random(3)
bad2 = bad3 = 0; tot = 0
for _ in range(60):
    n = rng.randint(4, 12)
    G = nx.gnp_random_graph(n, rng.random()*0.5+0.1, seed=rng.randint(0, 10**9))
    non = [(u, v) for u in G for v in G if u < v and not G.has_edge(u, v)]
    if non:
        u, v = rng.choice(non); G2 = G.copy(); G2.add_edge(u, v)
        D = max(max(dict(G2.degree).values()), 1)
        for r in (1, 2, 3):
            B = sum(D**j for j in range(r+1))
            h = indep.lin_hist([(Fr(1, n), G2), (-Fr(1, n), G)], r, reg)
            for k in (1, 2):
                tot += 1
                if indep.wnorm(h, k, reg) > Fr(4*B**(k+1), n): bad2 += 1
    v = rng.randrange(n); Gp = G.copy(); Gp.add_edge(v, n)
    D = max(max(dict(Gp.degree).values()), 1)
    for r in (1, 2, 3):
        B = sum(D**j for j in range(r+1))
        h = indep.lin_hist([(Fr(1, n+1), Gp), (-Fr(1, n), G)], r, reg)
        for k in (1, 2):
            tot += 1
            if indep.wnorm(h, k, reg) > Fr(2*(B+1)*B**k, n+1): bad3 += 1
print("sparse bounds checked", tot, "fail(2)", bad2, "fail(3)", bad3)
# (1-K2)^2 and chi_{-1}
sq = [(1, nx.empty_graph(1)), (-2, nx.path_graph(2)), (1, nx.cycle_graph(4))]
h = indep.lin_hist(sq, 1, reg)
print("(1-K2)^2 radius-1 hist sizes/coeffs:", sorted((reg.reps[t].number_of_nodes(), v) for t, v in h.items()))
