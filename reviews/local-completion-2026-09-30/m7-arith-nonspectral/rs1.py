import sys, itertools
from fractions import Fraction as Q
import networkx as nx
import sympy as sp
from gtools import *

R, S = rook(), shrikhande()
for name, G in (("rook", R), ("shrikhande", S)):
    A = sp.Matrix(nx.to_numpy_array(G, nodelist=sorted(G.nodes), dtype=int).astype(int).tolist())
    n = A.shape[0]
    deg = set(dict(G.degree).values())
    lam = sp.symbols("lam")
    cpA = sp.factor(A.charpoly(lam).as_expr())
    L = sp.diag(*[6] * n) - A
    cpL = sp.factor(L.charpoly(lam).as_expr())
    A2 = A * A
    srg = all(A2[i, j] == (6 if i == j else 2) for i in range(n) for j in range(n))
    print(name, "n=", n, "edges=", G.number_of_edges(), "degrees", deg, "diam", nx.diameter(G))
    print("  charpoly A:", cpA)
    print("  charpoly L:", cpL)
    print("  A^2 = 4I+2J ?", srg)
    # links
    kinds = set()
    for o in G.nodes:
        comps = link_components(G, o)
        sig = tuple(sorted((c.number_of_nodes(), c.number_of_edges(), nx.is_isomorphic(c, nx.complete_graph(3)), nx.is_isomorphic(c, nx.cycle_graph(6))) for c in comps))
        kinds.add(sig)
    print("  link component signatures over all roots:", kinds)
    # 4-cliques through each root, triangles through each root
    c4 = set(); c3 = set()
    for o in G.nodes:
        L_ = link(G, o)
        c3.add(L_.number_of_edges())
        c4.add(sum(1 for tri in itertools.combinations(L_.nodes, 3) if all(L_.has_edge(u, v) for u, v in itertools.combinations(tri, 2))))
    print("  root triangle counts:", c3, " root K4 counts:", c4)

reg = TypeRegistry()
for r in (1, 2, 3):
    hR = histogram(R, r, reg, Q(1, 16))
    hS = histogram(S, r, reg, Q(1, 16))
    print(f"r={r}: rook hist {hR}  shrikhande hist {hS}")
    X = dict(hR)
    for k, v in hS.items():
        X[k] = X.get(k, 0) - v
    X = {k: v for k, v in X.items() if v}
    print(f"   T_{r}(X) = {X}, l1 = {l1(X)}, sizes = {[reg.reps[k].number_of_nodes() for k in X]}")
    for k in X:
        B = reg.reps[k]
        print("     type", k, "nodes", B.number_of_nodes(), "edges", B.number_of_edges())
# radius-1 ball identification
B_R = reg.reps[list(histogram(R, 1, reg).keys())[0]]
B_S = reg.reps[list(histogram(S, 1, reg).keys())[0]]
def cone(Lk):
    G = nx.Graph(); G.add_node("apex")
    G.add_nodes_from(Lk.nodes)
    G.add_edges_from(Lk.edges)
    for v in Lk.nodes: G.add_edge("apex", v)
    return rooted_from(G, "apex")
print("B_R == cone(2K3)?", rooted_iso(B_R, cone(nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3)))))
print("B_S == cone(C6)?", rooted_iso(B_S, cone(nx.cycle_graph(6))))
