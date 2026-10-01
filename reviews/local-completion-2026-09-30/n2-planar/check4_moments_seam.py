import numpy as np
import networkx as nx
from rb import *

E = ((0, 0), (1, 0))


def rel_traces(Gb, Ga, order=6):
    import scipy.sparse as sps
    nodes = list(Gb.nodes)
    Lb = sps.csr_matrix(nx.laplacian_matrix(Gb, nodelist=nodes), dtype=np.int64)
    La = sps.csr_matrix(nx.laplacian_matrix(Ga, nodelist=nodes), dtype=np.int64)
    out = []
    Pb = sps.identity(len(nodes), dtype=np.int64, format='csr'); Pa = sps.identity(len(nodes), dtype=np.int64, format='csr')
    for j in range(order + 1):
        out.append(int(Pa.diagonal().sum() - Pb.diagonal().sum()))
        Pb = Pb @ Lb; Pa = Pa @ La
    return out

# lattice: box margin 6 around the edge (closed walks of length <= 6 cannot reach the boundary effect)
G = nx.Graph()
for x in range(-6, 8):
    for y in range(-6, 7):
        if x + 1 < 8: G.add_edge((x, y), (x + 1, y))
        if y + 1 < 7: G.add_edge((x, y), (x, y + 1))
Ga = G.copy(); Ga.remove_edge(*E)
print("lattice relative Laplacian traces j=0..6:", rel_traces(G, Ga))
G2 = nx.Graph()
for x in range(-8, 10):
    for y in range(-8, 9):
        if x + 1 < 10: G2.add_edge((x, y), (x + 1, y))
        if y + 1 < 9: G2.add_edge((x, y), (x, y + 1))
G2a = G2.copy(); G2a.remove_edge(*E)
print("lattice (bigger box)          j=0..6:", rel_traces(G2, G2a))

# 4-regular tree: two 3-ary trees of depth 6 joined at roots
T = nx.Graph(); T.add_edge(0, 1); frontier = [0, 1]; n = 2
for _ in range(5):
    nf = []
    for u in frontier:
        for _ in range(3):
            T.add_edge(u, n); nf.append(n); n += 1
    frontier = nf
Ta = T.copy(); Ta.remove_edge(0, 1)
print("4-tree relative Laplacian traces j=0..6:", rel_traces(T, Ta))

# seam element E*L: (P_n x C_m - C_n x C_m)/m, one root per column (all columns equivalent)
for r in (2, 3):
    n, m = 4 * r + 4, 4 * r + 3
    PnCm = nx.cartesian_product(nx.path_graph(n), nx.cycle_graph(m))
    CnCm = nx.cartesian_product(nx.cycle_graph(n), nx.cycle_graph(m))
    cls = Classifier(); h = {}
    for u in range(n):
        add_hist(h, cls.classify(ball_of_graph(PnCm, (u, 0), r)), 1)
        add_hist(h, cls.classify(ball_of_graph(CnCm, (u, 0), r)), -1)
    degs = {}
    for c, v in h.items():
        d = root_degree(cls.reps[c]); degs[d] = degs.get(d, 0) + v
    print(f"E*L at r={r}: variation={norm(h)} (claimed 4r={4*r}); root-degree generating fn coefficients={degs}")
