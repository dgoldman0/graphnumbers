"""Independent graph utilities for refereeing (no repo code used)."""
import itertools, random
from collections import defaultdict, deque
import networkx as nx

# ---------- exact isomorphism registry (general graphs, optionally rooted) ----------
class Registry:
    def __init__(self):
        self.buckets = {}
        self.reps = []
    def _inv(self, G, root):
        # cheap invariant: sorted degree seq + WL hash with root label
        H = nx.Graph(G)
        lab = {v: ('R' if v == root else 'n') for v in H}
        nx.set_node_attributes(H, lab, 'lab')
        h = nx.weisfeiler_lehman_graph_hash(H, node_attr='lab', iterations=3)
        return H, (h, H.number_of_nodes(), H.number_of_edges(), root is not None)
    def key(self, G, root=None):
        H, inv = self._inv(G, root)
        lst = self.buckets.setdefault(inv, [])
        nm = lambda a, b: a['lab'] == b['lab']
        for (K, i) in lst:
            if nx.is_isomorphic(H, K, node_match=nm):
                return i
        i = len(self.reps)
        self.reps.append((H, root))
        lst.append((H, i))
        return i

REG = Registry()

def components(G):
    return [G.subgraph(c).copy() for c in nx.connected_components(G)]

def component_vector(G, reg=REG):
    """Graph-algebra element of G: dict class -> multiplicity."""
    out = defaultdict(int)
    for C in components(G):
        out[reg.key(C)] += 1
    return out

def rooted_ball(G, o, r):
    dist = nx.single_source_shortest_path_length(G, o, cutoff=r)
    return G.subgraph(dist.keys()).copy()

def histogram(G, r, reg=REG):
    out = defaultdict(int)
    for o in G.nodes():
        B = rooted_ball(G, o, r)
        out[reg.key(B, o)] += 1
    return out

def add_into(acc, vec, c):
    for k, v in vec.items():
        acc[k] += c * v

def clean(d):
    return {k: v for k, v in d.items() if v != 0}

def interaction(G, F, observable):
    """sum_{S subset F} (-1)^{k-|S|} observable(G - S)."""
    k = len(F)
    acc = defaultdict(int)
    for m in range(k + 1):
        for S in itertools.combinations(F, m):
            H = G.copy()
            H.remove_edges_from(S)
            add_into(acc, observable(H), (-1) ** (k - m))
    return clean(acc)

def quotient_leaf_edges(G, F):
    """Return (is_tree, leaf_edges) for the selected-cut quotient (multigraph)."""
    H = G.copy(); H.remove_edges_from(F)
    lab = {}
    for i, c in enumerate(nx.connected_components(H)):
        for v in c: lab[v] = i
    nb = max(lab.values()) + 1
    deg = [0] * nb
    loops = False
    for u, v in F:
        if lab[u] == lab[v]: loops = True
        deg[lab[u]] += 1; deg[lab[v]] += 1
    is_tree = nx.is_connected(G) and nb == len(F) + 1 and not loops
    leaves = [e for e in F if deg[lab[e[0]]] == 1 or deg[lab[e[1]]] == 1]
    return is_tree, leaves, lab
