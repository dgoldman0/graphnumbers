"""Independent rooted-ball machinery (networkx based), written from scratch for refereeing."""
import networkx as nx
from collections import Counter, defaultdict
from fractions import Fraction as Fr
from networkx.algorithms.isomorphism import categorical_node_match

NM = categorical_node_match('root', False)

class Registry:
    def __init__(self):
        self.reps = []
        self.buckets = defaultdict(list)
    def key(self, B):
        # cheap invariant: sizes + WL hash with root label
        return (B.number_of_nodes(), B.number_of_edges(),
                nx.weisfeiler_lehman_graph_hash(B, node_attr='rootlbl', iterations=4))
    def register(self, B):
        k = self.key(B)
        for i in self.buckets[k]:
            if nx.is_isomorphic(B, self.reps[i], node_match=NM):
                return i
        i = len(self.reps)
        self.reps.append(B)
        self.buckets[k].append(i)
        return i

REG = Registry()

def rooted(G, v):
    H = nx.Graph(G)
    for u in H.nodes:
        H.nodes[u]['root'] = (u == v)
        H.nodes[u]['rootlbl'] = 'R' if u == v else 'x'
    return H

def ball(G, v, r):
    d = nx.single_source_shortest_path_length(G, v, cutoff=r)
    B = G.subgraph(d.keys()).copy()
    return rooted(B, v)

def hist(G, r, reg=REG):
    return Counter(reg.register(ball(G, v, r)) for v in G.nodes)

def lin_hist(terms, r, reg=REG):
    h = defaultdict(Fr)
    for c, G in terms:
        for t, m in hist(G, r, reg).items():
            h[t] += Fr(c) * m
    return {t: v for t, v in h.items() if v}

def wnorm(h, k, reg=REG):
    return sum((Fr(reg.reps[t].number_of_nodes()) ** k * abs(v) for t, v in h.items()), Fr(0))

def connected_graphs(maxn, maxdeg=None):
    out = []
    for G in nx.graph_atlas_g():
        n = G.number_of_nodes()
        if n == 0 or n > maxn:
            continue
        if not nx.is_connected(G):
            continue
        if maxdeg is not None and max(dict(G.degree).values()) > maxdeg:
            continue
        out.append(nx.convert_node_labels_to_integers(G))
    return out

def rank_mod_p(rows, p=2**61-1):
    """rank over GF(p) of an integer matrix given as list of rows; <= rank over Q"""
    M = [[x % p for x in row] for row in rows]
    rank = 0
    ncols = len(M[0]) if M else 0
    for c in range(ncols):
        piv = None
        for i in range(rank, len(M)):
            if M[i][c]:
                piv = i; break
        if piv is None: continue
        M[rank], M[piv] = M[piv], M[rank]
        inv = pow(M[rank][c], p-2, p)
        M[rank] = [(x*inv) % p for x in M[rank]]
        for i in range(len(M)):
            if i != rank and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f*b) % p for a, b in zip(M[i], M[rank])]
        rank += 1
    return rank
