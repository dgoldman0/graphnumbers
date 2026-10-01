"""Independent from-scratch tools for refereeing MIXED_MEDIUM_ARITHMETIC.md.

Nothing here imports the repo package. Rooted balls are networkx graphs whose
nodes carry 'dist' = distance from the root (root is the unique dist-0 node).
Rooted isomorphism == isomorphism preserving 'dist' (a rooted iso preserves
distances; conversely a dist-preserving iso fixes the unique dist-0 vertex).
"""
from collections import defaultdict, Counter
from fractions import Fraction as Q
import itertools
import networkx as nx


# ---------------------------------------------------------------- balls
def ball(G, root, r):
    dist = nx.single_source_shortest_path_length(G, root, cutoff=r)
    B = nx.Graph()
    for v, d in dist.items():
        B.add_node(v, dist=d)
    for v in dist:
        for w in G[v]:
            if w in dist:
                B.add_edge(v, w)
    return B


def product_ball(G, H, root, r):
    """r-ball of the Cartesian product G□H at root=(x,y), by BFS in the product."""
    dist = {root: 0}
    frontier = [root]
    for k in range(1, r + 1):
        new = []
        for (x, y) in frontier:
            for x2 in G[x]:
                p = (x2, y)
                if p not in dist:
                    dist[p] = k
                    new.append(p)
            for y2 in H[y]:
                p = (x, y2)
                if p not in dist:
                    dist[p] = k
                    new.append(p)
        frontier = new
    B = nx.Graph()
    for p, d in dist.items():
        B.add_node(p, dist=d)
    for (x, y) in dist:
        for x2 in G[x]:
            if (x2, y) in dist:
                B.add_edge((x, y), (x2, y))
        for y2 in H[y]:
            if (x, y2) in dist:
                B.add_edge((x, y), (x, y2))
    return B


def root_of(B):
    for v, d in B.nodes(data="dist"):
        if d == 0:
            return v
    raise ValueError("no root")


def star_product(B, C, r):
    """B ⋆_r C for rooted balls given as graphs with 'dist'."""
    return product_ball(B, C, (root_of(B), root_of(C)), r)


def labeled_signature(B):
    return (frozenset(B.nodes), frozenset(frozenset(e) for e in B.edges))


# ---------------------------------------------------------------- canonical types
class Registry:
    """Exact rooted-isomorphism classes: bucket by invariants, then vf2pp."""

    def __init__(self, rooted=True):
        self.rooted = rooted
        self.buckets = defaultdict(list)
        self.reps = []
        self.iso_calls = 0

    def key(self, B):
        """Isomorphism-invariant key; also stores WL-refined colours as node attr 'wl'.

        The colours are computed by a deterministic WL refinement started from
        (dist, degree) (rooted) or degree (unrooted), so any (rooted) isomorphism
        preserves them; using them as vf2pp node labels keeps the test exact while
        pruning the search on highly symmetric tree products."""
        if self.rooted:
            lab = {v: f"{B.nodes[v]['dist']}:{B.degree(v)}" for v in B}
        else:
            lab = {v: f"{B.degree(v)}" for v in B}
        G = nx.Graph()
        G.add_nodes_from((v, {"lab": l}) for v, l in lab.items())
        G.add_edges_from(B.edges)
        sub = nx.weisfeiler_lehman_subgraph_hashes(G, node_attr="lab", iterations=6)
        wl = {v: lab[v] + "|" + hs[-1] if hs else lab[v] for v, hs in sub.items()}
        nx.set_node_attributes(B, wl, "wl")
        h = hash(tuple(sorted(Counter(wl.values()).items())))
        return (B.number_of_nodes(), B.number_of_edges(), h)

    def id(self, B):
        k = self.key(B)
        for tid in self.buckets[k]:
            self.iso_calls += 1
            R = self.reps[tid]
            ok = nx.vf2pp_is_isomorphic(B, R, node_label="wl")
            if ok:
                return tid
        tid = len(self.reps)
        # store a relabeled copy (integers) to keep reps light
        mapping = {v: i for i, v in enumerate(B.nodes)}
        R = nx.relabel_nodes(B, mapping, copy=True)
        self.reps.append(R)
        self.buckets[k].append(tid)
        return tid


REG = Registry(rooted=True)
UREG = Registry(rooted=False)


def hist_add(h, tid, c):
    h[tid] = h.get(tid, 0) + c
    if h[tid] == 0:
        del h[tid]


def l1(h):
    return sum(abs(c) for c in h.values())


def weighted(h, k):
    return sum(abs(c) * REG.reps[t].number_of_nodes() ** k for t, c in h.items())


# ---------------------------------------------------------------- media with a deleted edge
def tree_with_edge(d, L):
    """Finite piece of the d-regular tree: central edge (('a',),('b',)) and depth L each side."""
    G = nx.Graph()
    a, b = ("a",), ("b",)
    G.add_edge(a, b)
    for side in (a, b):
        frontier = [side]
        for depth in range(L):
            new = []
            for p in frontier:
                for i in range(d - 1):
                    c = p + (i,)
                    G.add_edge(p, c)
                    new.append(c)
            frontier = new
    return G, (a, b)


def grid_with_edge(M):
    """Box [-M, M+1] x [-M, M] of Z^2 and the edge (0,0)-(1,0)."""
    G = nx.grid_2d_graph(range(-M, M + 2), range(-M, M + 1))
    return G, ((0, 0), (1, 0))


class Cut:
    """x = (G minus e) - G, as a pair of explicit finite graphs on one vertex set."""

    def __init__(self, name, G, e):
        self.name = name
        self.before = G
        self.after = G.copy()
        self.after.remove_edge(*e)
        self.e = e

    def affected(self, r):
        out = []
        for v in self.before:
            if labeled_signature(ball(self.before, v, r)) != labeled_signature(ball(self.after, v, r)):
                out.append(v)
        return out

    def near(self, R):
        s = set()
        for u in self.e:
            s |= set(nx.single_source_shortest_path_length(self.before, u, cutoff=R))
        return s


def T(cut, r, roots=None):
    """Signed histogram over all roots (or given roots) of after - before."""
    h = {}
    roots = cut.before.nodes if roots is None else roots
    for v in roots:
        Ba, Bb = ball(cut.after, v, r), ball(cut.before, v, r)
        if labeled_signature(Ba) == labeled_signature(Bb):
            continue
        hist_add(h, REG.id(Ba), 1)
        hist_add(h, REG.id(Bb), -1)
    return h


def T_product_direct(c1, c2, r, roots1=None, roots2=None):
    """T_r((G1a-G1b)(G2a-G2b)) computed from balls of the four product graphs."""
    roots1 = list(c1.before.nodes) if roots1 is None else list(roots1)
    roots2 = list(c2.before.nodes) if roots2 is None else list(roots2)
    h = {}
    nonzero_roots = 0
    for u in roots1:
        for v in roots2:
            terms = []
            for s1, G in ((1, c1.after), (-1, c1.before)):
                for s2, H in ((1, c2.after), (-1, c2.before)):
                    B = product_ball(G, H, (u, v), r)
                    terms.append((s1 * s2, B))
            # cancel identical labeled balls first
            sig = defaultdict(int)
            rep = {}
            for s, B in terms:
                k = labeled_signature(B)
                sig[k] += s
                rep[k] = B
            live = [(c, rep[k]) for k, c in sig.items() if c]
            if live:
                nonzero_roots += 1
            for c, B in live:
                hist_add(h, REG.id(B), c)
    return h, nonzero_roots


def convolve(h1, h2, r):
    out = {}
    for t1, a in h1.items():
        for t2, b in h2.items():
            P = star_product(REG.reps[t1], REG.reps[t2], r)
            hist_add(out, REG.id(P), a * b)
    return out


def scale(h, c):
    return {t: c * v for t, v in h.items() if c * v}


def add(h1, h2):
    out = dict(h1)
    for t, v in h2.items():
        hist_add(out, t, v)
    return out


def unit_hist():
    B = nx.Graph()
    B.add_node(0, dist=0)
    return {REG.id(B): 1}


# ---------------------------------------------------------------- the note's invariants
def interior_regular(B, r):
    o = root_of(B)
    d0 = B.degree(o)
    return all(B.degree(v) == d0 for v, dv in B.nodes(data="dist") if dv < r)


def Gamma(B):
    """Root neighbours; join u,v iff they have no common neighbour other than the root."""
    o = root_of(B)
    N = list(B[o])
    Gm = nx.Graph()
    Gm.add_nodes_from(range(len(N)))
    for i, j in itertools.combinations(range(len(N)), 2):
        common = (set(B[N[i]]) & set(B[N[j]])) - {o}
        if not common:
            Gm.add_edge(i, j)
    return Gm


def Gamma_share4(B):
    """Hypothetical complement version: join iff they DO share a 4-cycle."""
    return nx.complement(Gamma(B))


def component_multiset(Gm):
    c = Counter()
    for comp in nx.connected_components(Gm):
        S = Gm.subgraph(comp).copy()
        S = nx.relabel_nodes(S, {v: i for i, v in enumerate(S.nodes)})
        c[UREG.id(S)] += 1
    return c


def complete_id(n):
    return UREG.id(nx.complete_graph(n))


def nu(B):
    return component_multiset(Gamma(B))


def face_projection(h, r):
    return {t: c for t, c in h.items() if interior_regular(REG.reps[t], r)}
