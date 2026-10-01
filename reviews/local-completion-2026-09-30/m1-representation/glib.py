"""Independent small-graph toolkit for refereeing (does not import repo code).

Graphs: dict[int, frozenset[int]] adjacency (simple, undirected).
Rooted types: canonical certificate computed by exhaustive search within
refined color cells (exact for the small sizes used here).
"""
from __future__ import annotations
from collections import deque, defaultdict
from itertools import permutations, product
from fractions import Fraction as Fr
import random


def mk(n, edges):
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        if u == v:
            raise ValueError
        adj[u].add(v); adj[v].add(u)
    return {k: frozenset(s) for k, s in adj.items()}


def cycle(n):
    return mk(n, [(i, (i + 1) % n) for i in range(n)])


def path(n):
    return mk(n, [(i, i + 1) for i in range(n - 1)])


def complete(n):
    return mk(n, [(i, j) for i in range(n) for j in range(i + 1, n)])


def star(k):
    return mk(k + 1, [(0, i) for i in range(1, k + 1)])


def edges_of(g):
    return [(u, v) for u in g for v in g[u] if u < v]


def dists(g, o):
    d = {o: 0}
    q = deque([o])
    while q:
        u = q.popleft()
        for w in g[u]:
            if w not in d:
                d[w] = d[u] + 1
                q.append(w)
    return d


def ball(g, o, r):
    """Induced rooted r-ball, relabelled so the root is 0. Returns adjacency."""
    d = dists(g, o)
    verts = [o] + sorted(v for v in d if v != o and d[v] <= r)
    idx = {v: i for i, v in enumerate(verts)}
    return {idx[v]: frozenset(idx[w] for w in g[v] if w in idx) for v in verts}


def components(g):
    seen = set()
    out = []
    for v in g:
        if v not in seen:
            d = dists(g, v)
            seen |= set(d)
            out.append(sorted(d))
    return out


def induced(g, verts):
    idx = {v: i for i, v in enumerate(verts)}
    return {idx[v]: frozenset(idx[w] for w in g[v] if w in idx) for v in verts}


def _refine(g, colors):
    while True:
        sig = {v: (colors[v], tuple(sorted(colors[w] for w in g[v]))) for v in g}
        keys = sorted(set(sig.values()))
        newc = {v: keys.index(sig[v]) for v in g}
        if len(set(newc.values())) == len(set(colors.values())):
            return newc
        colors = newc


def canon(g, root=None):
    """Exact canonical certificate of (g, root)."""
    n = len(g)
    if n == 0:
        return (0,)
    d = dists(g, root) if root is not None else None
    base = {v: ((0 if root is None else (d.get(v, 10**6))), len(g[v])) for v in g}
    keys = sorted(set(base.values()))
    colors = _refine(g, {v: keys.index(base[v]) for v in g})
    # individualize-refine search for the lexicographically smallest adjacency string
    best = None

    def rec(colors):
        nonlocal best
        cells = defaultdict(list)
        for v, c in colors.items():
            cells[c].append(v)
        if all(len(c) == 1 for c in cells.values()):
            order = sorted(g, key=lambda v: colors[v])
            pos = {v: i for i, v in enumerate(order)}
            cert = tuple(sorted((min(pos[u], pos[w]), max(pos[u], pos[w]))
                                for u in g for w in g[u] if u < w))
            cert = (n, tuple(colors_key_tuple(order)), cert)
            if best is None or cert < best:
                best = cert
            return
        # pick first smallest nontrivial cell
        c = min((c for c in cells if len(cells[c]) > 1), key=lambda c: (len(cells[c]), c))
        for v in cells[c]:
            nc = dict(colors)
            # individualize v: give it a new color just below its cell
            nc = {u: (2 * col + (0 if u == v else 1)) if col == c else 2 * col for u, col in colors.items()}
            keys2 = sorted(set(nc.values()))
            nc = {u: keys2.index(nc[u]) for u in nc}
            rec(_refine(g, nc))

    def colors_key_tuple(order):
        return [base[v] for v in order]

    rec(colors)
    return best


class Registry:
    def __init__(self):
        self.ids = {}
        self.reps = []

    def rid(self, g, root=0):
        c = canon(g, root)
        if c not in self.ids:
            self.ids[c] = len(self.reps)
            self.reps.append(g)
        return self.ids[c]


def inj_rooted(F, u, G, o):
    """Number of injective edge-preserving maps F->G with u->o (backtracking)."""
    if len(F[u]) > len(G[o]):
        return 0
    # BFS order of F from u (F connected assumed)
    order = [u]
    seen = {u}
    parent = {}
    q = deque([u])
    while q:
        x = q.popleft()
        for y in sorted(F[x]):
            if y not in seen:
                seen.add(y); parent[y] = x; order.append(y); q.append(y)
    assert len(order) == len(F), "pattern must be connected"
    m = {u: o}
    used = {o}

    def rec(i):
        if i == len(order):
            return 1
        x = order[i]
        tot = 0
        for y in G[m[parent[x]]]:
            if y in used or len(G[y]) < len(F[x]):
                continue
            ok = True
            for z in F[x]:
                if z in m and m[z] not in G[y]:
                    ok = False; break
            if not ok:
                continue
            m[x] = y; used.add(y)
            tot += rec(i + 1)
            del m[x]; used.discard(y)
        return tot

    return rec(1)


def inj_total(F, G):
    return sum(inj_rooted(F, 0, G, o) for o in G)


def maxdeg(g):
    return max((len(s) for s in g.values()), default=0)


def is_connected(g):
    return len(g) > 0 and len(dists(g, next(iter(g)))) == len(g)


def hist(g, r, reg):
    h = defaultdict(int)
    for o in g:
        h[reg.rid(ball(g, o, r))] += 1
    return h


def lin_hist(terms, r, reg):
    h = defaultdict(Fr)
    for c, g in terms:
        for k, v in hist(g, r, reg).items():
            h[k] += c * v
    return {k: v for k, v in h.items() if v != 0}


def wnorm(h, reg, k):
    return sum(abs(v) * len(reg.reps[t]) ** k for t, v in h.items())


def cart(a, b):
    na, nb = len(a), len(b)
    adj = {}
    for x in range(na):
        for y in range(nb):
            s = set()
            for x2 in a[x]:
                s.add(x2 * nb + y)
            for y2 in b[y]:
                s.add(x * nb + y2)
            adj[x * nb + y] = frozenset(s)
    return adj


def random_maxdeg_graph(n, D, p, rng):
    edges = []
    deg = [0] * n
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pairs)
    for i, j in pairs:
        if deg[i] < D and deg[j] < D and rng.random() < p:
            edges.append((i, j)); deg[i] += 1; deg[j] += 1
    return mk(n, edges)
