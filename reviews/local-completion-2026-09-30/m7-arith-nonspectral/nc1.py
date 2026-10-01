from fractions import Fraction as Q
from collections import Counter
import itertools
import networkx as nx
import sympy as sp
from gtools import *

def joint(G):
    c = Counter()
    for o in G.nodes:
        L = link(G, o)
        c[(G.degree(o), L.number_of_edges())] += 1
    return c

paw = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3)])
G = nx.disjoint_union(nx.complete_graph(3), nx.star_graph(3))
J = nx.disjoint_union(paw, nx.path_graph(3))
jG, jJ = joint(G), joint(J)
print("G joint", sorted(jG.items())); print("J joint", sorted(jJ.items()))
print("deg marginals", sorted(Counter(d for (d, t), c in jG.items() for _ in range(c)).items()), sorted(Counter(d for (d, t), c in jJ.items() for _ in range(c)).items()))
print("tri marginals", sorted(Counter(t for (d, t), c in jG.items() for _ in range(c)).items()), sorted(Counter(t for (d, t), c in jJ.items() for _ in range(c)).items()))
z, w = sp.symbols("z w")
F = sum(c * z**d * w**t for (d, t), c in jG.items()) - sum(c * z**d * w**t for (d, t), c in jJ.items())
print("F_{G-J} =", sp.factor(F), " matches z^2(w-1)(1-z)?", sp.expand(F - z**2 * (w - 1) * (1 - z)) == 0)
M11 = lambda j: sum(c * d * t for (d, t), c in j.items())
print("M11", M11(jG), M11(jJ))
def kappa11(j):
    V = Q(sum(j.values()), 7); M10 = Q(sum(c * d for (d, t), c in j.items()), 7)
    M01 = Q(sum(c * t for (d, t), c in j.items()), 7); M11_ = Q(M11(j), 7)
    return M11_ / V - M10 * M01 / V**2
print("normalized covariance difference", kappa11(jG) - kappa11(jJ))
# Mixed moment with a background: check M11((U(G)-U(J)) Y) for some finite normalized Y via product graphs
for Yg in (nx.path_graph(3), nx.complete_graph(4), paw, nx.cycle_graph(5)):
    def M11norm(H):
        return Q(sum(H.degree(o) * link(H, o).number_of_edges() for o in H.nodes), H.number_of_nodes())
    val = M11norm(nx.cartesian_product(G, Yg)) - M11norm(nx.cartesian_product(J, Yg))
    print("  background", Yg.number_of_nodes(), "vertices: M11((U(G)-U(J))U(Y)) =", val)

# Cone independence: matrix delta_F(cone(F')) for connected F up to 5 vertices
def connected_graphs(maxn):
    out = []
    for n in range(1, maxn + 1):
        reps = []
        for G_ in nx.graph_atlas_g():
            if G_.number_of_nodes() == n and nx.is_connected(G_):
                reps.append(G_)
        out.extend(reps)
    return out
Fs = connected_graphs(5)
print("#connected graphs up to 5 vertices:", len(Fs))
def cone_graph(F):
    H = nx.Graph(); H.add_nodes_from(F.nodes); H.add_edges_from(F.edges)
    H.add_node("apex")
    for v in F.nodes: H.add_edge("apex", v)
    return H
def deltaF(F, H):
    s = 0
    for o in H.nodes:
        for c in link_components(H, o):
            if nx.is_isomorphic(c, F): s += 1
    return s
M = sp.Matrix([[deltaF(F, cone_graph(Fp)) for Fp in Fs] for F in Fs])
print("rank of delta_F(cone F') matrix:", M.rank(), "of", len(Fs))
diag_ok = all(M[i, i] == 1 + sum(1 for v in Fs[i].nodes if Fs[i].degree(v) == Fs[i].number_of_nodes() - 1) for i in range(len(Fs)))
print("diagonal = 1+u(F)?", diag_ok)
# triangularity: delta_F(cone F') = 0 when |F| > |F'| or (|F| == |F'| and F != F')
tri = all(M[i, j] == 0 for i in range(len(Fs)) for j in range(len(Fs))
          if Fs[i].number_of_nodes() > Fs[j].number_of_nodes() or (Fs[i].number_of_nodes() == Fs[j].number_of_nodes() and i != j))
print("block triangular as claimed?", tri)

# Paw transport witness for weighting by c3 and exp(s c3)
c3 = {o: link(paw, o).number_of_edges() for o in paw.nodes}
leaf = 3; nb = 0
print("paw c3:", c3, "outgoing weight (leaf)", c3[leaf], "incoming weight (neighbor)", c3[nb])
# P3 witness for c_{K1}
P3 = nx.path_graph(3)
cK1 = {o: sum(1 for c in link_components(P3, o) if c.number_of_nodes() == 1) for o in P3.nodes}
print("P3 c_K1:", cK1)
# cone(F)+leaf witness
for F in Fs[1:6]:
    H = cone_graph(F); H.add_edge("apex", "leaf")
    print("  F", sorted(F.edges), "apex c_F", deltaF_local := sum(1 for c in link_components(H, "apex") if nx.is_isomorphic(c, F)),
          "leaf c_F", sum(1 for c in link_components(H, "leaf") if nx.is_isomorphic(c, F)))
