"""Finite simple graphs and exact isomorphism, with explicit work limits.

The refinement/backtracking algorithm is adapted from the project's finite
research verifier. Labels are storage coordinates; IsoGraph supplies equality
up to (optionally rooted) isomorphism. No probabilistic hash decides equality.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations


class BudgetExceeded(RuntimeError):
    """A work limit was reached; this says nothing about the mathematical result."""


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def rational(value):
    """Accept explicit exact numbers; use a decimal string instead of a float."""
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise TypeError("Use int, Fraction, or an exact rational/decimal string")
    return Fraction(value)


@dataclass(frozen=True)
class Graph:
    rows: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "rows", tuple(self.rows))
        n = self.n
        for u, row in enumerate(self.rows):
            integer(row, "adjacency row")
            if row >= 1 << n or row & (1 << u):
                raise ValueError("Graph must have in-range vertices and no loops")
            for v in self.neighbors(u):
                if not self.rows[v] & (1 << u):
                    raise ValueError("Adjacency must be symmetric")

    @property
    def n(self):
        return len(self.rows)

    @property
    def edges(self):
        return sum(row.bit_count() for row in self.rows) // 2

    @property
    def max_degree(self):
        return max((row.bit_count() for row in self.rows), default=0)

    def neighbors(self, u):
        row = self.rows[u]
        while row:
            bit = row & -row
            yield bit.bit_length() - 1
            row -= bit


def graph(n, edges=()):
    """Build a simple graph, requiring each undirected edge exactly once."""
    integer(n, "vertex count")
    rows = [0] * n
    for u, v in edges:
        integer(u, "vertex")
        integer(v, "vertex")
        if u >= n or v >= n or u == v:
            raise ValueError("Edges require distinct vertices in range")
        if rows[u] & (1 << v):
            raise ValueError("Repeated undirected edges are not allowed")
        rows[u] |= 1 << v
        rows[v] |= 1 << u
    return Graph(tuple(rows))


def path(n):
    integer(n, "vertex count")
    return graph(n, ((i, i + 1) for i in range(n - 1)))


def cycle(n):
    integer(n, "cycle size", 3)
    return graph(n, ((i, (i + 1) % n) for i in range(n)))


def complete(n):
    integer(n, "vertex count")
    return graph(n, combinations(range(n), 2))


def star(leaves):
    integer(leaves, "leaf count")
    return graph(leaves + 1, ((0, i) for i in range(1, leaves + 1)))


def disjoint_union(*graphs):
    rows = []
    for g in graphs:
        offset = len(rows)
        rows.extend(row << offset for row in g.rows)
    return Graph(tuple(rows))


def induced(g, order):
    order = tuple(order)
    index = {u: i for i, u in enumerate(order)}
    if len(index) != len(order) or any(u < 0 or u >= g.n for u in order):
        raise ValueError("Induced vertex list must be distinct and in range")
    return Graph(tuple(sum(1 << index[v] for v in g.neighbors(u) if v in index)
                       for u in order))


def distances(g, root, radius=None):
    integer(root, "root")
    if root >= g.n:
        raise ValueError("Root is outside graph")
    if radius is not None:
        integer(radius, "radius")
    found, queue = {root: 0}, deque([root])
    while queue:
        u = queue.popleft()
        if radius is not None and found[u] >= radius:
            continue
        for v in g.neighbors(u):
            if v not in found:
                found[v] = found[u] + 1
                queue.append(v)
    return found


def components(g):
    unseen = set(range(g.n))
    while unseen:
        order = tuple(distances(g, min(unseen)))
        unseen.difference_update(order)
        yield induced(g, order)


@lru_cache(maxsize=8192)
def ball(g, root, radius):
    return induced(g, distances(g, root, radius))


def cartesian(g, h, max_vertices=10000):
    integer(max_vertices, "max_vertices", 1)
    if g.n * h.n > max_vertices:
        raise BudgetExceeded("Cartesian product exceeds max_vertices")
    rows = []
    for u in range(g.n):
        for v in range(h.n):
            rows.append(sum(1 << (x * h.n + v) for x in g.neighbors(u))
                        | sum(1 << (u * h.n + y) for y in h.neighbors(v)))
    return Graph(tuple(rows))


def local_product(g, h, radius, max_vertices=10000):
    """Construct only pairs at distance <= radius from the root pair (0,0)."""
    integer(radius, "radius")
    integer(max_vertices, "max_vertices", 1)
    dg, dh = distances(g, 0, radius), distances(h, 0, radius)
    order = []
    for u, d in dg.items():
        for v, e in dh.items():
            if d + e <= radius:
                order.append((u, v))
                if len(order) > max_vertices:
                    raise BudgetExceeded("Rooted Cartesian ball exceeds max_vertices")
    index = {uv: i for i, uv in enumerate(order)}
    rows = []
    for u, v in order:
        neighbors = [(x, v) for x in g.neighbors(u)]
        neighbors.extend((u, y) for y in h.neighbors(v))
        rows.append(sum(1 << index[w] for w in neighbors if w in index))
    return Graph(tuple(rows))


@lru_cache(maxsize=16384)
def isomorphic(a, b, rooted=False, search_budget=100000):
    integer(search_budget, "search_budget", 1)
    if a == b:
        return True
    if a.n != b.n or a.edges != b.edges:
        return False
    ca = [(int(rooted and u == 0), row.bit_count()) for u, row in enumerate(a.rows)]
    cb = [(int(rooted and u == 0), row.bit_count()) for u, row in enumerate(b.rows)]
    palette = {key: i for i, key in enumerate(sorted(set(ca + cb)))}
    ca, cb = [palette[k] for k in ca], [palette[k] for k in cb]
    while True:
        ka = [(ca[u], tuple(sorted(ca[v] for v in a.neighbors(u)))) for u in range(a.n)]
        kb = [(cb[u], tuple(sorted(cb[v] for v in b.neighbors(u)))) for u in range(b.n)]
        palette = {key: i for i, key in enumerate(sorted(set(ka + kb)))}
        na, nb = [palette[k] for k in ka], [palette[k] for k in kb]
        if Counter(na) != Counter(nb):
            return False
        stable = len(set(na + nb)) == len(set(ca + cb))
        ca, cb = na, nb
        if stable:
            break
    mapping = {0: 0} if rooted else {}
    used, work = set(mapping.values()), 0

    def visit():
        nonlocal work
        work += 1
        if work > search_budget:
            raise BudgetExceeded("Exact isomorphism exceeds search_budget")
        if len(mapping) == a.n:
            return True
        choices = []
        for u in range(a.n):
            if u in mapping:
                continue
            opts = [v for v in range(b.n) if v not in used and ca[u] == cb[v]
                    and all(bool(a.rows[u] & (1 << x)) == bool(b.rows[v] & (1 << y))
                            for x, y in mapping.items())]
            choices.append((opts, u))
        opts, u = min(choices, key=lambda p: len(p[0]))
        for v in opts:
            mapping[u] = v
            used.add(v)
            if visit():
                return True
            del mapping[u]
            used.remove(v)
        return False

    return visit()


@dataclass(frozen=True, eq=False)
class IsoGraph:
    graph: Graph
    rooted: bool = False

    def __post_init__(self):
        if self.rooted and not self.graph.n:
            raise ValueError("A rooted graph must be nonempty")

    def __hash__(self):
        g = self.graph
        return hash((self.rooted, g.n, g.edges,
                     tuple(sorted(row.bit_count() for row in g.rows)),
                     g.rows[0].bit_count() if self.rooted else None))

    def __eq__(self, other):
        if not isinstance(other, IsoGraph):
            return NotImplemented
        return self.rooted == other.rooted and isomorphic(self.graph, other.graph, self.rooted)


def from_networkx(value):
    """Duck-typed adapter; labels/attributes are discarded after validation."""
    if value.is_directed() or value.is_multigraph():
        raise ValueError("Only simple undirected graphs are supported")
    labels = {v: i for i, v in enumerate(value.nodes)}
    return graph(len(labels), ((labels[u], labels[v]) for u, v in value.edges))
