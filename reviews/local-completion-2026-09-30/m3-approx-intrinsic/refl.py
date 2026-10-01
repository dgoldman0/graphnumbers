"""Independent implementation of the parity filter F, sections R and O."""
import networkx as nx
from fractions import Fraction as Fr
import indep

def parity_filter(G):
    Q = nx.Graph(); Q.add_nodes_from(G.nodes)
    Q.add_edges_from((u, v) for u, v in G.edges if G.degree(u) % 2 == G.degree(v) % 2)
    return Q

def F_graph(G):
    """list of (sign, component graph)"""
    Q = parity_filter(G)
    out = []
    for comp in nx.connected_components(Q):
        v = next(iter(comp))
        out.append(((-1) ** (G.degree(v) % 2), nx.convert_node_labels_to_integers(Q.subgraph(comp).copy())))
    return out

def F(terms):
    return [(c * s, H) for c, G in terms for s, H in F_graph(G)]

def R_graph(G):
    H = nx.convert_node_labels_to_integers(G)
    n = H.number_of_nodes()
    odd = [v for v in H.nodes if H.degree(v) % 2]
    for i, v in enumerate(odd):
        H.add_edge(v, n + i)
    m = H.number_of_nodes()
    H.add_nodes_from(range(m, m + len(odd)))  # isolated vertices
    return H

def O_graph(G):
    H = nx.convert_node_labels_to_integers(G)
    n = H.number_of_nodes()
    even = [v for v in list(H.nodes) if H.degree(v) % 2 == 0]
    for i, v in enumerate(even):
        H.add_edge(v, n + 2 * i); H.add_edge(n + 2 * i, n + 2 * i + 1)
    return H

def components(terms):
    out = []
    for c, G in terms:
        for comp in nx.connected_components(G):
            out.append((c, nx.convert_node_labels_to_integers(G.subgraph(comp).copy())))
    return out

class GraphTypes:
    def __init__(self):
        self.reps = []
    def tid(self, G):
        for i, H in enumerate(self.reps):
            if H.number_of_nodes() == G.number_of_nodes() and H.number_of_edges() == G.number_of_edges() and nx.is_isomorphic(G, H):
                return i
        self.reps.append(G); return len(self.reps) - 1

GT = GraphTypes()

def coeffs(terms):
    d = {}
    for c, G in components(terms):
        t = GT.tid(G)
        d[t] = d.get(t, 0) + Fr(c)
    return {t: v for t, v in d.items() if v}

def cart(G, H):
    return nx.convert_node_labels_to_integers(nx.cartesian_product(G, H))
