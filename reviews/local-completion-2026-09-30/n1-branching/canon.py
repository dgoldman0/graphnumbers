"""Independent rooted-ball machinery (written from scratch for refereeing).

Graphs are adjacency dicts {v: set(neighbors)}.  Rooted iso classes are
registered by (WL hash with BFS-distance labels, |V|, |E|) buckets, then
confirmed exactly with VF2 (networkx) using distance labels as node match.
"""
from collections import defaultdict, deque, Counter
from fractions import Fraction
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher


def bfs_dist(adj, root, limit=None):
    dist = {root: 0}
    dq = deque([root])
    while dq:
        u = dq.popleft()
        if limit is not None and dist[u] >= limit:
            continue
        for w in adj[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                dq.append(w)
    return dist


def ball(adj, root, r):
    """Induced r-ball as (adjacency dict on 0..n-1 with root 0, dist list)."""
    dist = bfs_dist(adj, root, r)
    order = sorted(dist, key=lambda v: (dist[v], repr(v)))
    idx = {v: i for i, v in enumerate(order)}
    nadj = {i: set() for i in range(len(order))}
    for v in order:
        for w in adj[v]:
            if w in idx:
                nadj[idx[v]].add(idx[w])
    return nadj, [dist[v] for v in order]


def ahu(adj, root):
    """Exact canonical string of a rooted tree (iterative AHU)."""
    parent = {root: None}
    order = [root]
    for u in order:
        for w in adj[u]:
            if w not in parent:
                parent[w] = u
                order.append(w)
    code = {}
    for u in reversed(order):
        kids = sorted(code[w] for w in adj[u] if parent.get(w) == u and w != parent[u])
        code[u] = "(" + "".join(kids) + ")"
    return code[root]


class Registry:
    def __init__(self):
        self.buckets = defaultdict(list)
        self.reps = []          # cid -> (adj, dist)
        self.nx = []            # cid -> networkx graph with 'd' labels
        self.cache = {}
        self.tree_codes = {}

    def _nxg(self, adj, dist):
        G = nx.Graph()
        for v in adj:
            G.add_node(v, d=dist[v], deg=len(adj[v]))
        for v in adj:
            for w in adj[v]:
                if v < w:
                    G.add_edge(v, w)
        return G

    def cid(self, adj, dist):
        G = self._nxg(adj, dist)
        nE = G.number_of_edges()
        for v in G:
            G.nodes[v]['l0'] = f"{G.nodes[v]['d']}:{G.nodes[v]['deg']}"
        sub = nx.weisfeiler_lehman_subgraph_hashes(G, node_attr='l0', iterations=6, digest_size=16)
        for v in G:
            G.nodes[v]['c'] = sub[v][-1]
        h = tuple(sorted(Counter(G.nodes[v]['c'] for v in G).items()))
        key = (h, G.number_of_nodes(), nE)
        for c in self.buckets[key]:
            H = self.nx[c]
            gm = GraphMatcher(G, H, node_match=lambda a, b: a['c'] == b['c'])
            if gm.is_isomorphic():
                return c
        c = len(self.reps)
        self.reps.append((adj, dist))
        self.nx.append(G)
        self.buckets[key].append(c)
        return c

    def ball_cid(self, adj, root, r):
        b, d = ball(adj, root, r)
        nE = sum(len(s) for s in b.values()) // 2
        if nE == len(b) - 1:
            # rooted tree: exact AHU canonical string (root is vertex 0)
            code = ahu(b, 0)
            t = self.tree_codes.get(code)
            if t is None:
                t = self.cid(b, d)
                self.tree_codes[code] = t
            return t
        return self.cid(b, d)

    # --- local Cartesian product of two registered rooted r-balls ---
    def star(self, c1, c2, r):
        k = (min(c1, c2), max(c1, c2), r)
        if k in self.cache:
            return self.cache[k]
        (a1, d1), (a2, d2) = self.reps[c1], self.reps[c2]
        verts = [(x, y) for x in a1 for y in a2 if d1[x] + d2[y] <= r]
        vs = set(verts)
        adj = {v: set() for v in verts}
        for (x, y) in verts:
            for x2 in a1[x]:
                if (x2, y) in vs:
                    adj[(x, y)].add((x2, y))
            for y2 in a2[y]:
                if (x, y2) in vs:
                    adj[(x, y)].add((x, y2))
        res = self.ball_cid(adj, (0, 0), r)
        self.cache[k] = res
        return res

    def size(self, c):
        return len(self.reps[c][0])

    def root_degree(self, c):
        return len(self.reps[c][0][0])


def histogram(reg, adj, r, coeff=1, roots=None, out=None):
    out = defaultdict(Fraction) if out is None else out
    for v in (adj if roots is None else roots):
        out[reg.ball_cid(adj, v, r)] += Fraction(coeff)
    return out


def clean(h):
    return {k: v for k, v in h.items() if v != 0}


def add(h1, h2, s=1):
    out = defaultdict(Fraction)
    for k, v in h1.items():
        out[k] += v
    for k, v in h2.items():
        out[k] += s * v
    return clean(out)


def convolve(reg, h1, h2, r):
    out = defaultdict(Fraction)
    for k1, v1 in h1.items():
        for k2, v2 in h2.items():
            out[reg.star(k1, k2, r)] += v1 * v2
    return clean(out)


def norm(reg, h, k=0):
    return sum(abs(v) * reg.size(c) ** k for c, v in h.items())


# ---------------- graph constructors ----------------

def edge_centered_tree(d, L):
    """Two (d-1)-ary trees of depth L joined by the central edge (0,1)."""
    adj = {0: {1}, 1: {0}}
    frontier, n = [0, 1], 2
    for _ in range(L):
        nf = []
        for u in frontier:
            for _ in range(d - 1):
                adj[n] = {u}
                adj[u].add(n)
                nf.append(n)
                n += 1
        frontier = nf
    return adj


def delete_edge(adj, u, v):
    new = {x: set(s) for x, s in adj.items()}
    new[u].discard(v)
    new[v].discard(u)
    return new


def path_graph(n):
    return {i: {j for j in (i - 1, i + 1) if 0 <= j < n} for i in range(n)}


def cycle_graph(n):
    return {i: {(i - 1) % n, (i + 1) % n} for i in range(n)}


def grid_box(n):
    """[-n,n]^2 induced square grid."""
    adj = {}
    for x in range(-n, n + 1):
        for y in range(-n, n + 1):
            s = set()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if -n <= x + dx <= n and -n <= y + dy <= n:
                    s.add((x + dx, y + dy))
            adj[(x, y)] = s
    return adj


def cartesian(a1, a2):
    adj = {}
    for x in a1:
        for y in a2:
            s = {(x2, y) for x2 in a1[x]} | {(x, y2) for y2 in a2[y]}
            adj[(x, y)] = s
    return adj
