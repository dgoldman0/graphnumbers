import networkx as nx
from fractions import Fraction as Q
from graphlocal.graphs import Graph, graph, isomorphic, IsoGraph, BudgetExceeded

def to_nx(g):
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v in g.neighbors(u):
            if u < v:
                G.add_edge(u, v)
    return G

def from_nx(G, root=None):
    nodes = list(G.nodes)
    if root is not None:
        nodes.remove(root)
        nodes = [root] + nodes
    idx = {v: i for i, v in enumerate(nodes)}
    return graph(len(nodes), [(idx[u], idx[v]) for u, v in G.edges])

def nx_iso(a, b, rooted=False):
    A, B = to_nx(a), to_nx(b)
    if rooted:
        nx.set_node_attributes(A, {u: int(u == 0) for u in A}, "r")
        nx.set_node_attributes(B, {u: int(u == 0) for u in B}, "r")
        return nx.is_isomorphic(A, B, node_match=lambda x, y: x["r"] == y["r"])
    return nx.is_isomorphic(A, B)

def nx_ball(G, root, r):
    """Induced r-ball as networkx graph, root kept."""
    d = nx.single_source_shortest_path_length(G, root, cutoff=r)
    return G.subgraph(d.keys()).copy()

def rooted_hist(G, r, coeff=Q(1), roots=None):
    """Oracle histogram: list of (nx rooted ball, coeff) aggregated by nx isomorphism."""
    classes = []  # list of [H, root, coeff]
    for v in (G.nodes if roots is None else roots):
        H = nx_ball(G, v, r)
        nx.set_node_attributes(H, {u: int(u == v) for u in H}, "r")
        for c in classes:
            if c[0].number_of_nodes() == H.number_of_nodes() and c[0].number_of_edges() == H.number_of_edges() and \
               nx.is_isomorphic(c[0], H, node_match=lambda x, y: x["r"] == y["r"]):
                c[2] += coeff
                break
        else:
            classes.append([H, v, coeff])
    return classes

def merge_hists(*hs):
    out = []
    for h in hs:
        for H, v, c in h:
            for o in out:
                if o[0].number_of_nodes() == H.number_of_nodes() and o[0].number_of_edges() == H.number_of_edges() and \
                   nx.is_isomorphic(o[0], H, node_match=lambda x, y: x["r"] == y["r"]):
                    o[2] += c
                    break
            else:
                out.append([H, v, c])
    return [o for o in out if o[2] != 0]

def scale_hist(h, s):
    return [[H, v, c * s] for H, v, c in h]

def hist_to_lib(h, r):
    from graphlocal.local import LocalHistogram
    return LocalHistogram(r, [(from_nx(H, v), c) for H, v, c in h])

def lib_to_classes(hist):
    out = []
    for key, c in hist.values.items():
        H = to_nx(key.graph)
        nx.set_node_attributes(H, {u: int(u == 0) for u in H}, "r")
        out.append([H, 0, c])
    return out

def classes_equal(h1, h2):
    """Compare two oracle-class lists via networkx rooted isomorphism; returns (ok, diff)."""
    diff = merge_hists(h1, scale_hist(h2, -1))
    return (len(diff) == 0, diff)

def lib_vs_oracle(hist, oracle):
    return classes_equal(lib_to_classes(hist), oracle)
