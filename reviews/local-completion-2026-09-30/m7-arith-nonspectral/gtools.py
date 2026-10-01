"""Independent graph tools for refereeing (no repo code used).

Graphs: networkx.Graph with integer or tuple nodes.
Rooted types: networkx.Graph with node attribute 'root' (True for root).
Canonical classification: WL hash (with root label) + explicit isomorphism check.
"""
from fractions import Fraction as Q
from collections import defaultdict
import itertools
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher


def ball(G, o, r):
    dist = nx.single_source_shortest_path_length(G, o, cutoff=r)
    B = G.subgraph(dist.keys()).copy()
    B = nx.convert_node_labels_to_integers(B, label_attribute="orig")
    for v in B.nodes:
        B.nodes[v]["root"] = (B.nodes[v]["orig"] == o)
    return B


def rooted_from(G, o):
    """Whole graph G rooted at o as a rooted type."""
    B = nx.convert_node_labels_to_integers(G, label_attribute="orig")
    for v in B.nodes:
        B.nodes[v]["root"] = (B.nodes[v]["orig"] == o)
    return B


def root_of(B):
    return next(v for v in B.nodes if B.nodes[v]["root"])


def whash(B):
    H = B.copy()
    for v in H.nodes:
        H.nodes[v]["lab"] = "R" if H.nodes[v]["root"] else "x"
    return (B.number_of_nodes(), B.number_of_edges(),
            nx.weisfeiler_lehman_graph_hash(H, node_attr="lab", iterations=4))


def rooted_iso(B, C):
    if B.number_of_nodes() != C.number_of_nodes() or B.number_of_edges() != C.number_of_edges():
        return False
    gm = GraphMatcher(B, C, node_match=lambda a, b: a["root"] == b["root"])
    return gm.is_isomorphic()


class TypeRegistry:
    """Assigns integer ids to rooted isomorphism classes."""
    def __init__(self):
        self.by_hash = defaultdict(list)
        self.reps = []

    def id(self, B):
        h = whash(B)
        for tid in self.by_hash[h]:
            if rooted_iso(self.reps[tid], B):
                return tid
        tid = len(self.reps)
        self.reps.append(B)
        self.by_hash[h].append(tid)
        return tid


def histogram(G, r, reg, scale=Q(1)):
    h = defaultdict(Q)
    for o in G.nodes:
        h[reg.id(ball(G, o, r))] += scale
    return dict(h)


def cart(G, H):
    return nx.cartesian_product(G, H)


def star_r(B, C, r, reg):
    """B *_r C = B_r(B box C, (oB, oC))."""
    P = nx.cartesian_product(B, C)
    oB, oC = root_of(B), root_of(C)
    # strip attributes from product nodes
    P2 = nx.Graph()
    P2.add_nodes_from(P.nodes)
    P2.add_edges_from(P.edges)
    return reg.id(ball(P2, (oB, oC), r))


def convolve(a, b, r, reg, cache):
    out = defaultdict(Q)
    for x, cx in a.items():
        for y, cy in b.items():
            key = (min(x, y), max(x, y))
            if key not in cache:
                cache[key] = star_r(reg.reps[x], reg.reps[y], r, reg)
            out[cache[key]] += cx * cy
    return {k: v for k, v in out.items() if v != 0}


def l1(a):
    return sum(abs(v) for v in a.values())


def link(G, o):
    return G.subgraph(list(G.neighbors(o))).copy()


def link_components(G, o):
    L = link(G, o)
    return [L.subgraph(c).copy() for c in nx.connected_components(L)]


def rook():
    return nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4))


def shrikhande():
    G = nx.Graph()
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    for x in range(4):
        for y in range(4):
            G.add_node((x, y))
            for dx, dy in steps:
                G.add_edge((x, y), ((x + dx) % 4, (y + dy) % 4))
    return G
