"""Independent referee code: graphs, Cartesian products, induced rooted balls,
exact rooted canonical forms, and local histograms.  Nothing from the repo."""
import itertools
from collections import Counter, defaultdict
from fractions import Fraction as Fr
from math import comb, factorial


# ---------------------------------------------------------------- graphs
def path(m):
    adj = [set() for _ in range(m)]
    for i in range(m - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    return adj


def cycle(m):
    adj = [set() for _ in range(m)]
    for i in range(m):
        adj[i].add((i + 1) % m)
        adj[(i + 1) % m].add(i)
    return adj


def cartesian(graphs):
    sizes = [len(g) for g in graphs]
    verts = list(itertools.product(*[range(s) for s in sizes]))
    index = {v: i for i, v in enumerate(verts)}
    adj = [set() for _ in verts]
    for v in verts:
        i = index[v]
        for c, g in enumerate(graphs):
            for w in g[v[c]]:
                u = v[:c] + (w,) + v[c + 1:]
                adj[i].add(index[u])
    return adj, verts


def ball(adj, root, r):
    """Induced r-ball; returns (adjacency list relabelled with root 0, dist list, original ids)."""
    dist = {root: 0}
    frontier = [root]
    order = [root]
    for d in range(1, r + 1):
        nxt = []
        for u in frontier:
            for w in adj[u]:
                if w not in dist:
                    dist[w] = d
                    nxt.append(w)
                    order.append(w)
        frontier = nxt
    pos = {v: i for i, v in enumerate(order)}
    badj = [set() for _ in order]
    for v in order:
        for w in adj[v]:
            if w in pos:
                badj[pos[v]].add(pos[w])
    return badj, [dist[v] for v in order], order


# ------------------------------------------------- canonical rooted form
def refine(adj, colors):
    """Colour refinement (1-WL) with canonical relabelling; isomorphism-invariant."""
    n = len(adj)
    while True:
        sigs = [(colors[v], tuple(sorted(colors[w] for w in adj[v]))) for v in range(n)]
        ranking = {s: i for i, s in enumerate(sorted(set(sigs)))}
        new = [ranking[s] for s in sigs]
        if len(set(new)) == len(set(colors)):
            return new
        colors = new


def canonical_form(adj, root=0):
    """Exact canonical form of a rooted graph by individualisation-refinement
    (full search tree, minimum leaf certificate).  Root individualised first."""
    n = len(adj)
    init = [1] * n
    init[root] = 0
    best = [None]

    def leaf_cert(colors):
        order = sorted(range(n), key=lambda v: colors[v])
        pos = {v: i for i, v in enumerate(order)}
        edges = tuple(sorted((min(pos[u], pos[w]), max(pos[u], pos[w]))
                             for u in range(n) for w in adj[u] if u < w))
        return (n, edges)

    def search(colors):
        colors = refine(adj, colors)
        counts = Counter(colors)
        if len(counts) == n:
            cert = leaf_cert(colors)
            if best[0] is None or cert < best[0]:
                best[0] = cert
            return
        # target cell: smallest colour with size > 1 (invariant choice)
        target = min(c for c, k in counts.items() if k > 1)
        for v in [u for u in range(n) if colors[u] == target]:
            new = [2 * c for c in colors]
            new[v] = 2 * target + 1  # individualised vertex, label-independent rule
            search(new)

    search(init)
    return best[0]


# A faster exact path for large balls: WL bucket + VF2 (networkx), used only
# as a cross-check / for bigger cases.
def nx_rooted(adj, root=0):
    import networkx as nx
    G = nx.Graph()
    for v in range(len(adj)):
        G.add_node(v, tag='root' if v == root else 'x')
    for v in range(len(adj)):
        for w in adj[v]:
            if v < w:
                G.add_edge(v, w)
    return G


class VF2Classifier:
    def __init__(self):
        self.buckets = defaultdict(list)  # hash -> list of (id, G)
        self.count = 0

    def classify(self, adj, root=0):
        import networkx as nx
        from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match
        G = nx_rooted(adj, root)
        h = nx.weisfeiler_lehman_graph_hash(G, node_attr='tag', iterations=5)
        h = (len(adj), sum(len(a) for a in adj), h)
        for cid, H in self.buckets[h]:
            if GraphMatcher(G, H, node_match=categorical_node_match('tag', None)).is_isomorphic():
                return cid
        cid = self.count
        self.count += 1
        self.buckets[h].append((cid, G))
        return cid


# ------------------------------------------------------- invariants
def square_count(adj, root=0):
    nb = sorted(adj[root])
    total = 0
    for a, b in itertools.combinations(nb, 2):
        total += len((adj[a] & adj[b]) - {root})
    return total


def sphere_counts(adj, r, root=0):
    dist = {root: 0}
    frontier = [root]
    out = [1]
    for d in range(1, r + 1):
        nxt = []
        for u in frontier:
            for w in adj[u]:
                if w not in dist:
                    dist[w] = d
                    nxt.append(w)
        out.append(len(nxt))
        frontier = nxt
    return out


def ps_mul(a, b, order):
    return [sum(a[i] * b[k - i] for i in range(k + 1) if i < len(a) and k - i < len(b))
            for k in range(order + 1)]


def ps_log(s, order):
    """log of a power series with constant term 1, truncated at z^order (exact)."""
    s = [Fr(x) for x in s] + [Fr(0)] * (order + 1 - len(s))
    s = s[:order + 1]
    assert s[0] == 1
    # log' = s'/s ;  use recurrence: L = log s,  s' = s L'
    L = [Fr(0)] * (order + 1)
    # n s_n = sum_{k=1}^n k L_k s_{n-k}
    for n in range(1, order + 1):
        acc = n * s[n] - sum(k * L[k] * s[n - k] for k in range(1, n))
        L[n] = acc / n  # since s_0 = 1
    return L


def coordinates(adj, r, root=0):
    """(c_0..c_{r-1}, c_L) of UNBOUNDED_VARIATION_INVERSION (3)-(8), my own implementation."""
    assert r >= 2
    d = len(adj[root])
    Q = square_count(adj, root)
    N = Fr(3 * d - d * d + 2 * Q, 2)
    S = sphere_counts(adj, r, root)
    F = [Fr(1)] + [Fr(2)] * r  # (1+z)/(1-z) mod z^{r+1}
    U = [a - N * b for a, b in zip(ps_log(S, r), ps_log(F, r))]
    ells = []
    for j in range(r):
        # 1 - z^{j+1}/(1+z) = 1 - sum_{i>=0} (-1)^i z^{j+1+i}
        base = [Fr(1)] + [Fr(0)] * r
        for i in range(0, r + 1):
            if j + 1 + i <= r:
                base[j + 1 + i] -= (-1) ** i
        ells.append(ps_log(base, r))
    c = []
    resid = U[:]
    for j in range(r):
        cj = -resid[j + 1]
        c.append(cj)
        resid = [a - cj * b for a, b in zip(resid, ells[j])]
    assert all(x == 0 for x in resid), resid
    return tuple(c) + (N - sum(c),), N


# ------------------------------------------------------- histograms
def histogram(adj, r, classify):
    """T_r of a finite graph as Counter over canonical keys."""
    H = Counter()
    reps = {}
    for v in range(len(adj)):
        b, _, _ = ball(adj, v, r)
        key = classify(b)
        H[key] += 1
        reps.setdefault(key, b)
    return H, reps
