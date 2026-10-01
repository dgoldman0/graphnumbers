"""Independent rooted-ball toolkit (written from scratch for refereeing).

Graphs are neighbor functions nb(v) -> iterable of vertices, vertices hashable.
Rooted r-ball = induced subgraph on vertices at graph distance <= r.
Exact rooted isomorphism classes: cheap invariants + WL hash, then VF2 with
node_match on distance-from-root (equivalent to fixing the root).
"""
from collections import deque, defaultdict
from fractions import Fraction as Fr
import itertools
import networkx as nx
from networkx.algorithms import isomorphism as iso


def grid_adj(xmin, xmax, ymin, ymax, deleted=()):
    adj = {}
    dele = {frozenset(e) for e in deleted}
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            nbrs = set()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                u = (x + dx, y + dy)
                if xmin <= u[0] <= xmax and ymin <= u[1] <= ymax:
                    if frozenset(((x, y), u)) not in dele:
                        nbrs.add(u)
            adj[(x, y)] = nbrs
    return adj


def adj_nb(adj):
    return lambda v: adj[v]


def product_nb(nb1, nb2):
    def nb(v):
        a, b = v
        for a2 in nb1(a):
            yield (a2, b)
        for b2 in nb2(b):
            yield (a, b2)
    return nb


def ball(nb, root, r):
    """Return networkx Graph of induced r-ball with node attr 'd'."""
    dist = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        if dist[u] == r:
            continue
        for w in nb(u):
            if w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    G = nx.Graph()
    for v, dv in dist.items():
        G.add_node(v, d=dv)
    for v in dist:
        for w in nb(v):
            if w in dist:
                G.add_edge(v, w)
    G.graph['root'] = root
    G.graph['r'] = r
    return G


def ball_of_graph(G, root, r):
    """r-ball of a networkx graph G (rooted at root)."""
    return ball(lambda v: G.neighbors(v), root, r)


def invariants(G):
    """Rooted-isomorphism invariants (used only to group before an exact test)."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    layers = defaultdict(list)
    for v, dv in G.nodes(data='d'):
        layers[dv].append(G.degree(v))
    lay = tuple((k, tuple(sorted(layers[k]))) for k in sorted(layers))
    for v in G.nodes:
        dv = G.nodes[v]['d']
        up = sum(1 for w in G.neighbors(v) if G.nodes[w]['d'] > dv)
        dn = sum(1 for w in G.neighbors(v) if G.nodes[w]['d'] < dv)
        nb = set(G.neighbors(v))
        sq = sum(len((set(G.neighbors(a)) & set(G.neighbors(b))) - {v}) for a, b in itertools.combinations(nb, 2))
        G.nodes[v]['lab'] = f"{dv},{G.degree(v)},{up},{dn},{sq}"
    h = nx.weisfeiler_lehman_graph_hash(G, node_attr='lab', iterations=5)
    return (n, m, lay, h)


_nm = lambda a, b: a['d'] == b['d']


def rooted_iso(G, H):
    if G.number_of_nodes() != H.number_of_nodes() or G.number_of_edges() != H.number_of_edges():
        return False
    return nx.vf2pp_is_isomorphic(G, H, node_label='d')


class Classifier:
    def __init__(self):
        self.groups = defaultdict(list)   # key -> list of (rep, cid)
        self.reps = []

    def classify(self, G):
        key = invariants(G)
        for rep, cid in self.groups[key]:
            if rooted_iso(G, rep):
                return cid
        cid = len(self.reps)
        self.reps.append(G)
        self.groups[key].append((G, cid))
        return cid


def add_hist(h, cid, c):
    h[cid] = h.get(cid, 0) + c
    if h[cid] == 0:
        del h[cid]


def histogram(nb, roots, r, cls, coeff=1, h=None):
    h = {} if h is None else h
    for v in roots:
        add_hist(h, cls.classify(ball(nb, v, r)), coeff)
    return h


def norm(h, cls=None, k=0):
    if k == 0:
        return sum(abs(c) for c in h.values())
    return sum(abs(c) * cls.reps[cid].number_of_nodes() ** k for cid, c in h.items())


def root_degree(G):
    return G.degree(G.graph['root'])


def is_regular_face(G):
    r = G.graph['r']
    d0 = root_degree(G)
    return all(G.degree(v) == d0 for v, dv in G.nodes(data='d') if dv < r)


def square_count(G):
    """Simple 4-cycles through the root (chords allowed)."""
    o = G.graph['root']
    nbrs = list(G.neighbors(o))
    tot = 0
    for a, b in itertools.combinations(nbrs, 2):
        common = (set(G.neighbors(a)) & set(G.neighbors(b))) - {o}
        tot += len(common)
    return tot


def sphere_vector(G):
    r = G.graph['r']
    s = [0] * (r + 1)
    for v, dv in G.nodes(data='d'):
        s[dv] += 1
    return tuple(s)


def J_inv(G):
    o = G.graph['root']
    d = G.degree(o)
    return sum((G.degree(v) - d) ** 2 for v in G.neighbors(o))


def star_product_ball(B, D, r):
    """B star_r D = r-ball of B x D at (root,root), computed from explicit product."""
    nb = product_nb(lambda v: B.neighbors(v), lambda v: D.neighbors(v))
    return ball(nb, (B.graph['root'], D.graph['root']), r)


def convolve(h1, h2, cls, r):
    out = {}
    cache = {}
    for a, ca in h1.items():
        for b, cb in h2.items():
            key = (min(a, b), max(a, b))
            if key not in cache:
                cache[key] = cls.classify(star_product_ball(cls.reps[a], cls.reps[b], r))
            add_hist(out, cache[key], ca * cb)
    return out


# ---- formal power series (exact rationals) ----
def ps_mul(a, b, N):
    return [sum(a[i] * b[n - i] for i in range(n + 1) if i < len(a) and n - i < len(b)) for n in range(N + 1)]


def ps_log(a, N):
    """log of series with a[0]=1, truncated at degree N."""
    assert a[0] == 1
    a = [Fr(x) for x in a] + [Fr(0)] * (N + 1 - len(a))
    x = [Fr(0)] + a[1:N + 1]
    res = [Fr(0)] * (N + 1)
    p = [Fr(1)] + [Fr(0)] * N
    for k in range(1, N + 1):
        p = ps_mul(p, x, N)
        for j in range(N + 1):
            res[j] += Fr((-1) ** (k + 1), k) * p[j]
    return res


def ps_inv(a, N):
    a = [Fr(x) for x in a] + [Fr(0)] * (N + 1 - len(a))
    b = [Fr(0)] * (N + 1)
    b[0] = 1 / a[0]
    for n in range(1, N + 1):
        b[n] = -sum(a[i] * b[n - i] for i in range(1, n + 1)) / a[0]
    return b


def cutline_coords(G):
    """Cut-line additive coordinates (c_0..c_{r-1}, c_L) per UNBOUNDED_VARIATION_INVERSION.md (2)-(8),
    implemented independently from the formulas there."""
    r = G.graph['r']
    d = root_degree(G)
    Q = square_count(G)
    N = Fr(3 * d - d * d + 2 * Q, 2)
    S = list(sphere_vector(G))
    logS = ps_log(S, r)
    # F = (1+z)/(1-z)
    F = ps_mul([Fr(1), Fr(1)], ps_inv([Fr(1), Fr(-1)], r), r)
    logF = ps_log(F, r)
    U = [a - N * b for a, b in zip(logS, logF)]
    inv1pz = ps_inv([Fr(1), Fr(1)], r)
    coords = []
    for j in range(r):
        # ell_j = log(1 - z^{j+1}/(1+z))
        zj = [Fr(0)] * (r + 1)
        if j + 1 <= r:
            zj[j + 1] = Fr(1)
        t = ps_mul(zj, inv1pz, r)
        base = [Fr(1) - t[0]] + [-x for x in t[1:]]
        ell = ps_log(base, r)
        c = -U[j + 1] / 1  # leading coefficient of ell_j is -1
        assert ell[j + 1] == -1 and all(ell[i] == 0 for i in range(j + 1))
        coords.append(c)
        U = [a - c * b for a, b in zip(U, ell)]
    assert all(x == 0 for x in U), U
    return tuple(coords) + (N - sum(coords),)
