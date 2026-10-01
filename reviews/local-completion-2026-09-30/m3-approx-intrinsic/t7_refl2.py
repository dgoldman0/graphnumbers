import networkx as nx, indep, refl
from fractions import Fraction as Fr
reg = indep.Registry()
def prism(n): return refl.cart(nx.path_graph(2), nx.cycle_graph(n))
def join(G, H):
    G = nx.convert_node_labels_to_integers(G); m = G.number_of_nodes()
    H = nx.convert_node_labels_to_integers(H, first_label=m)
    J = nx.union(G, H); J.add_edge(0, m); return J
def star(k): return indep.rooted(nx.star_graph(k), 0)
for n in range(3, 10):
    P = prism(n); J = join(P, nx.cycle_graph(2*n))
    FJ = refl.F([(1, J)])
    punct = P.copy(); punct.remove_node(0)
    expected = refl.coeffs([(1, nx.path_graph(2*n-1)), (-1, punct)])
    formula = refl.coeffs(FJ) == expected
    h = indep.lin_hist([(Fr(c, J.number_of_nodes()), G) for c, G in FJ], 1, reg)
    st = {k: h.get(reg.register(star(k)), 0) for k in (1, 2, 3)}
    exp_st = {1: Fr(1, 2*n), 2: Fr(1, 2) - Fr(3, 2*n), 3: -Fr(1, 2) + Fr(1, n)}
    others = {t: v for t, v in h.items() if t not in [reg.register(star(k)) for k in (1, 2, 3)]}
    print(n, "formula(10):", formula, "stars:", st == exp_st, "other radius-1 types:", len(others), "V=", sum(c*G.number_of_nodes() for c, G in FJ))
# (15): theta(X_n) radius-2 coefficient on the 7-vertex cube ball; X_n = U(Q_n) - U(prism)
cube = refl.cart(refl.cart(nx.path_graph(2), nx.path_graph(2)), nx.path_graph(2))
cube_ball = reg.register(indep.ball(cube, 0, 2))
def Hpow(k):  # H^k = Q_k / 2^k
    G = nx.empty_graph(1)
    for _ in range(k): G = refl.cart(G, nx.path_graph(2))
    return (Fr(1, 2**k), G)
for n in range(4, 10):
    P = prism(n); Qn = P.copy(); m = Qn.number_of_nodes(); Qn.add_edge(0, m)
    N = Qn.number_of_nodes()
    # theta(P) = P + S(D(P)(-z) - D(P)(z)) = P - 2 sum_{odd v} H^{deg v}
    terms = [(Fr(1, N), Qn)]
    for v in Qn.nodes:
        d = Qn.degree(v)
        if d % 2:
            c, G = Hpow(d); terms.append((Fr(-2, N) * c, G))
    terms.append((Fr(1, P.number_of_nodes()), P))  # -theta(H U(C_n)) = + H U(C_n) = + U(prism)
    h = indep.lin_hist(terms, 2, reg)
    sizes = sorted(reg.reps[indep.REG.register(indep.ball(Qn, v, 2))].number_of_nodes() if False else indep.ball(Qn, v, 2).number_of_nodes() for v in range(m))
    print(n, "cube-ball coeff:", h.get(cube_ball, 0), "expected", -Fr(2*(2*n-1), 2*n+1), "min old-root ball size", sizes[0])
