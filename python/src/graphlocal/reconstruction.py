"""Exact finite-catalog reconstruction with primal/dual certificates.

The basis-enumeration method is adapted from research/local-completion/
reconstruct_local.py. The catalog is explicit: infeasibility is never a claim
about all finite graphs, and budget exhaustion has no feasibility implication.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import chain, combinations, product

from .elements import Finite
from .graphs import BudgetExceeded, Graph, distances, graph, integer, isomorphic
from .local import LocalHistogram


class OutOfSpan(ValueError):
    def __init__(self, witness, value):
        super().__init__("Target is outside this catalog's histogram span")
        self.witness, self.value = witness, value


def catalog(max_vertices, max_degree=None, label_budget=200000):
    integer(max_vertices, "max_vertices", 1)
    integer(label_budget, "label_budget", 1)
    if max_degree is not None:
        integer(max_degree, "max_degree")
    total = 0
    for n in range(1, max_vertices + 1):
        total += 1 << (n * (n - 1) // 2)
        if total > label_budget:
            raise BudgetExceeded("Complete labeled enumeration exceeds label_budget")
    buckets, result = defaultdict(list), []
    for n in range(1, max_vertices + 1):
        possible = list(combinations(range(n), 2))
        for mask in range(1 << len(possible)):
            g = graph(n, (e for j, e in enumerate(possible) if mask & (1 << j)))
            degrees = tuple(sorted(row.bit_count() for row in g.rows))
            if max_degree is not None and degrees[-1] > max_degree:
                continue
            if len(distances(g, 0)) != n:
                continue
            key = (n, g.edges, degrees)
            if not any(isomorphic(g, h) for h in buckets[key]):
                buckets[key].append(g)
                result.append(g)
    return tuple(sorted(result, key=lambda g: (g.n, g.edges, g.rows)))


def _pivots(matrix):
    a = [[Q(x) for x in row] for row in matrix]
    if not a:
        return []
    rank, pivots = 0, []
    for col in range(len(a[0])):
        pivot = next((i for i in range(rank, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][col]
        a[rank] = [x / scale for x in a[rank]]
        for i in range(rank + 1, len(a)):
            scale = a[i][col]
            a[i] = [x - scale * y for x, y in zip(a[i], a[rank])]
        pivots.append(col)
        rank += 1
        if rank == len(a):
            break
    return pivots


def _inverse(matrix):
    n = len(matrix)
    a = [[Q(x) for x in row] + [Q(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((i for i in range(col, n) if a[i][col]), None)
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [x / scale for x in a[col]]
        for i in range(n):
            if i != col and a[i][col]:
                scale = a[i][col]
                a[i] = [x - scale * y for x, y in zip(a[i], a[col])]
    return [row[n:] for row in a]


def _matvec(matrix, vector):
    return [sum((x * y for x, y in zip(row, vector)), Q(0)) for row in matrix]


@dataclass(frozen=True)
class Reconstruction:
    element: Finite
    coefficients: tuple[Q, ...]
    dual: LocalHistogram
    cost: Q
    negative_mass: Q
    search_work: int


def reconstruct(target, graphs, search_budget=100000):
    integer(search_budget, "search_budget", 1)
    graphs = tuple(graphs)
    if not graphs or any(not isinstance(g, Graph) or not g.n for g in graphs):
        raise ValueError("Supply a nonempty catalog of nonempty graphs")
    if any(len(distances(g, 0)) != g.n for g in graphs):
        raise ValueError("Reconstruction catalog graphs must be connected")
    radius = target.radius
    columns = [LocalHistogram.from_graph(g, radius).values for g in graphs]
    rows = sorted(set(target.values).union(*(set(c) for c in columns)), key=lambda key: key.graph.rows)
    h = [[col.get(t, Q(0)) for col in columns] for t in rows]
    b = [target.values.get(t, Q(0)) for t in rows]
    row_basis = _pivots(list(zip(*h)))
    a, reduced_b = [h[i] for i in row_basis], [b[i] for i in row_basis]
    rank = len(row_basis)
    first_basis = _pivots(a)
    inv = _inverse([[row[j] for j in first_basis] for row in a])
    if inv is None:
        raise ArithmeticError("Internal rank inconsistency")
    prediction = _matvec([[row[j] for j in first_basis] for row in h], _matvec(inv, reduced_b))
    mismatch = next((i for i in range(len(rows)) if prediction[i] != b[i]), None)
    if mismatch is not None:
        f = [Q(0) for _ in rows]
        f[mismatch] = Q(1)
        for k, row_id in enumerate(row_basis):
            f[row_id] -= sum(h[mismatch][j] * inv[l][k] for l, j in enumerate(first_basis))
        raise OutOfSpan(LocalHistogram(radius, zip(rows, f)), sum(x * y for x, y in zip(f, b)))
    sizes, work = [g.n for g in graphs], 0
    for basis in combinations(range(len(graphs)), rank):
        work += 1
        if work > search_budget:
            raise BudgetExceeded("Independent-basis search exceeds search_budget")
        inv = _inverse([[row[j] for j in basis] for row in a])
        if inv is None:
            continue
        coefficients = _matvec(inv, reduced_b)
        zero_positions = [i for i, c in enumerate(coefficients) if not c]
        signs = [Q(1 if c > 0 else -1 if c < 0 else 0) for c in coefficients]
        for inactive in chain([(Q(0),) * len(zero_positions)],
                              product((Q(-1), Q(1)), repeat=len(zero_positions))):
            work += 1
            if work > search_budget:
                raise BudgetExceeded("Dual-certificate search exceeds search_budget")
            trial = signs.copy()
            for i, sign in zip(zero_positions, inactive):
                trial[i] = sign
            rhs = [sizes[j] * trial[i] for i, j in enumerate(basis)]
            dual_reduced = _matvec(list(zip(*inv)), rhs)
            if any(abs(sum(dual_reduced[i] * a[i][j] for i in range(rank))) > sizes[j]
                   for j in range(len(graphs))):
                continue
            primal, dual = [Q(0) for _ in graphs], [Q(0) for _ in rows]
            for j, c in zip(basis, coefficients):
                primal[j] = c
            for i, v in zip(row_basis, dual_reduced):
                dual[i] = v
            cost = sum(n * abs(c) for n, c in zip(sizes, primal))
            if _matvec(h, primal) != b or cost != sum(x * y for x, y in zip(dual, b)):
                raise ArithmeticError("Invalid primal/dual certificate")
            return Reconstruction(Finite(zip(primal, graphs)), tuple(primal),
                                  LocalHistogram(radius, zip(rows, dual)), cost,
                                  (cost - sum(b)) / 2, work)
    raise ArithmeticError("Feasible finite LP had no primal/dual certificate")
