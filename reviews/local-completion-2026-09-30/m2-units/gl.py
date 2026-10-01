"""Independent rooted-graph toolkit for refereeing (does not import repo code).

Rooted graphs: root is vertex 0.  Simple graphs as tuple of frozensets.
Multigraph patterns: (n, tuple of edges (u,v) with u<v, repeats allowed), root 0.
"""
from __future__ import annotations
import itertools, math
from collections import defaultdict, Counter, deque
from fractions import Fraction as Q
from functools import lru_cache
import networkx as nx


class G:
    __slots__ = ("n", "adj", "_h")

    def __init__(self, n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            assert u != v
            adj[u].add(v); adj[v].add(u)
        self.n = n
        self.adj = tuple(frozenset(s) for s in adj)
        self._h = None

    def edges(self):
        return [(u, v) for u in range(self.n) for v in self.adj[u] if u < v]

    def deg(self, u=0):
        return len(self.adj[u])


def dist_from(g, o):
    d = {o: 0}; dq = deque([o])
    while dq:
        u = dq.popleft()
        for v in g.adj[u]:
            if v not in d:
                d[v] = d[u] + 1; dq.append(v)
    return d


def ball(g, o, r):
    d = dist_from(g, o)
    verts = [o] + sorted(v for v, x in d.items() if x <= r and v != o)
    idx = {v: i for i, v in enumerate(verts)}
    es = [(idx[u], idx[v]) for u in verts for v in g.adj[u] if v in idx and idx[u] < idx[v]]
    return G(len(verts), es)


def cart(a, b):
    nb = b.n
    es = []
    for u in range(a.n):
        for v in range(nb):
            x = u * nb + v
            for w in a.adj[u]:
                if w > u: es.append((x, w * nb + v))
            for w in b.adj[v]:
                if w > v: es.append((x, u * nb + w))
    return G(a.n * nb, es)


def ecc(g):
    return max(dist_from(g, 0).values())


def cycle(n):
    return G(n, [(i, (i + 1) % n) for i in range(n)])


def path(n, root=0):
    g = G(n, [(i, i + 1) for i in range(n - 1)])
    return ball(g, root, n)


def complete(n):
    return G(n, list(itertools.combinations(range(n), 2)))


def to_nx(g):
    h = nx.Graph()
    h.add_nodes_from((i, {"root": int(i == 0)}) for i in range(g.n))
    h.add_edges_from(g.edges())
    return h


class Registry:
    """Rooted isomorphism classes -> integer ids."""

    def __init__(self):
        self.buckets = defaultdict(list)
        self.reps = []
        self.nxreps = []

    def key(self, g):
        h = to_nx(g)
        degs = tuple(sorted(len(a) for a in g.adj))
        dist = Counter(dist_from(g, 0).values())
        wl = nx.weisfeiler_lehman_graph_hash(h, node_attr="root", iterations=3)
        return (g.n, len(g.edges()), degs, tuple(sorted(dist.items())), g.deg(0), wl), h

    def id(self, g):
        k, h = self.key(g)
        nm = lambda a, b: a["root"] == b["root"]
        for tid in self.buckets[k]:
            if nx.is_isomorphic(self.nxreps[tid], h, node_match=nm):
                return tid
        tid = len(self.reps)
        self.reps.append(g); self.nxreps.append(h); self.buckets[k].append(tid)
        return tid


REG = Registry()


@lru_cache(None)
def star(a, b, r):
    """type id of B_r(A box B) for type ids a,b."""
    return REG.id(ball(cart(REG.reps[a], REG.reps[b]), 0, r))


@lru_cache(None)
def trunc(a, r):
    return REG.id(ball(REG.reps[a], 0, r))


def hist(g, r):
    c = Counter()
    for o in range(g.n):
        c[REG.id(ball(g, o, r))] += 1
    return c


def lin_hist(terms, r):
    out = defaultdict(Q)
    for coef, g in terms:
        for t, m in hist(g, r).items():
            out[t] += Q(coef) * m
    return {t: v for t, v in out.items() if v != 0}


def conv(a, b, r):
    out = defaultdict(Q)
    for x, ax in a.items():
        for y, by in b.items():
            out[star(x, y, r)] += ax * by
    return {t: v for t, v in out.items() if v != 0}


# ---------------- walk counts / cumulants -----------------
def walks(g, L, closed=False):
    cnt = [0] * g.n; cnt[0] = 1
    out = [1]
    for _ in range(L):
        new = [0] * g.n
        for u in range(g.n):
            if cnt[u]:
                for v in g.adj[u]:
                    new[v] += cnt[u]
        cnt = new
        out.append(cnt[0] if closed else sum(cnt))
    return out


def cumulants_from_moments(m):
    """Standard moment->cumulant via EGF log, computed by sympy-free recursion
    k_n = m_n - sum_{j=1}^{n-1} C(n-1,j-1) k_j m_{n-j}."""
    k = [0]
    for n in range(1, len(m)):
        k.append(m[n] - sum(math.comb(n - 1, j - 1) * k[j] * m[n - j] for j in range(1, n)))
    return k


# ---------------- multigraph patterns, homs, coproduct -----------------
def canon_multi(n, edges):
    """Rooted canonical form of a loopless multigraph by brute force (small n)."""
    best = None
    for perm in itertools.permutations(range(1, n)):
        mp = (0,) + perm
        es = tuple(sorted(tuple(sorted((mp[u], mp[v]))) for u, v in edges))
        if best is None or es < best:
            best = es
    return (n, best)


def hom_count(pattern, g):
    """rooted homomorphisms of multigraph pattern (n,edges) into simple rooted graph g."""
    n, edges = pattern
    nbrs = [set() for _ in range(n)]
    for u, v in edges:
        nbrs[u].add(v); nbrs[v].add(u)
    # BFS order from root
    order = [0]; seen = {0}
    for u in order:
        for v in sorted(nbrs[u]):
            if v not in seen:
                seen.add(v); order.append(v)
    assert len(order) == n
    pos = {v: i for i, v in enumerate(order)}
    back = [[w for w in nbrs[v] if pos[w] < pos[v]] for v in order]
    img = [None] * n

    def rec(i):
        if i == n:
            return 1
        v = order[i]
        bs = back[i]
        cand = g.adj[img[bs[0]]]
        tot = 0
        for c in cand:
            if all(c in g.adj[img[w]] for w in bs[1:]):
                img[v] = c
                tot += rec(i + 1)
        img[v] = None
        return tot

    img[0] = 0
    return rec(1)


def quotient(pattern, coloring, color):
    n, edges = pattern
    par = list(range(n))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for (u, v), c in zip(edges, coloring):
        if c != color:
            par[f(u)] = f(v)
    cls = [f(u) for u in range(n)]
    root = cls[0]
    others = sorted(set(cls) - {root})
    mp = {root: 0}
    for i, x in enumerate(others, 1):
        mp[x] = i
    kept = []
    for (u, v), c in zip(edges, coloring):
        if c == color:
            a, b = mp[cls[u]], mp[cls[v]]
            if a == b:
                return None
            kept.append((min(a, b), max(a, b)))
    return canon_multi(len(mp), kept)


@lru_cache(None)
def coproduct(pattern, arity, onto):
    n, edges = pattern
    res = Counter()
    for col in itertools.product(range(arity), repeat=len(edges)):
        if onto and len(set(col)) != arity:
            continue
        fs = []
        ok = True
        for i in range(arity):
            qf = quotient(pattern, col, i)
            if qf is None:
                ok = False; break
            fs.append(qf)
        if ok:
            res[tuple(fs)] += 1
    return res


_homcache = {}


def h(pattern, tid):
    key = (pattern, tid)
    if key not in _homcache:
        _homcache[key] = hom_count(pattern, REG.reps[tid])
    return _homcache[key]


def K(pattern, tid):
    m = len(pattern[1])
    if m == 0:
        return Q(0)
    tot = Q(0)
    for nn in range(1, m + 1):
        s = 0
        for fs, mult in coproduct(pattern, nn, True).items():
            s += mult * math.prod(h(f, tid) for f in fs)
        tot += Q((-1) ** (nn - 1), nn) * s
    return tot


def pattern_ecc(pattern):
    n, edges = pattern
    g = G(n, set(edges))
    return ecc(g)


def all_patterns(max_edges, max_vertices=None):
    """All connected rooted loopless multigraph patterns with 1..max_edges edges."""
    out = set()
    for m in range(1, max_edges + 1):
        nv_max = m + 1 if max_vertices is None else min(m + 1, max_vertices)
        for nv in range(2, nv_max + 1):
            pairs = list(itertools.combinations(range(nv), 2))
            for ms in itertools.combinations_with_replacement(pairs, m):
                g = G(nv, set(ms))
                if len(dist_from(g, 0)) != nv:
                    continue
                out.add(canon_multi(nv, ms))
    return sorted(out, key=lambda p: (len(p[1]), p[0], p[1]))


def all_rooted(maxn, max_ecc=None):
    ids = set()
    for n in range(1, maxn + 1):
        pairs = list(itertools.combinations(range(n), 2))
        for mask in range(1 << len(pairs)):
            es = [pairs[i] for i in range(len(pairs)) if mask >> i & 1]
            g = G(n, es)
            d = dist_from(g, 0)
            if len(d) != n:
                continue
            if max_ecc is not None and max(d.values()) > max_ecc:
                continue
            ids.add(REG.id(g))
    return sorted(ids)
