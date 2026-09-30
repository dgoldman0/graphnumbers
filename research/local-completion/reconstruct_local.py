#!/usr/bin/env python3
"""Exact rational reconstruction on a finite graph catalog (Python 3.10+).

Minimizes sum |c_G| |G| subject to a prescribed rooted-ball histogram.
Returns independently checkable primal/dual certificates. Enumeration budgets
are explicit; exhaustion is never reported as infeasibility.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from fractions import Fraction as Q
from itertools import chain, combinations, product
from pathlib import Path

from verify_local_algebra import (
    Graph, TYPES, clean, distances, graph, hist, isomorphic,
)


class BudgetExceeded(RuntimeError):
    pass


class OutOfSpan(ValueError):
    def __init__(self, witness, value):
        super().__init__("Target is outside this catalog's histogram span.")
        self.witness = witness
        self.value = value


def catalog(max_vertices, max_degree=None, label_budget=200000):
    """Complete unrooted connected catalog within the requested bounds."""
    if max_vertices < 1 or (max_degree is not None and max_degree < 0):
        raise ValueError("Invalid graph bounds")
    total = 0
    for n in range(1, max_vertices + 1):
        total += 1 << (n * (n - 1) // 2)
        if total > label_budget:
            raise BudgetExceeded("Complete labeled enumeration exceeds label_budget")
    buckets = defaultdict(list)
    result = []
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
            if any(isomorphic(g, h) for h in buckets[key]):
                continue
            buckets[key].append(g)
            result.append(g)
    return sorted(result, key=lambda g: (g.n, g.edges, g.rows))


def pivot_columns(matrix):
    a = [[Q(x) for x in row] for row in matrix]
    if not a:
        return []
    rank = 0
    pivots = []
    for col in range(len(a[0])):
        pivot = next((i for i in range(rank, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][col]
        a[rank] = [x / scale for x in a[rank]]
        for i in range(rank + 1, len(a)):
            scale = a[i][col]
            if scale:
                a[i] = [x - scale * y for x, y in zip(a[i], a[rank])]
        pivots.append(col)
        rank += 1
        if rank == len(a):
            break
    return pivots


def inverse(matrix):
    n = len(matrix)
    a = [[Q(x) for x in row] + [Q(i == j) for j in range(n)]
         for i, row in enumerate(matrix)]
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


def matvec(matrix, vector):
    return [sum((x * y for x, y in zip(row, vector)), Q(0)) for row in matrix]


def reconstruct(target, graphs, radius, search_budget=100000):
    """Solve and certify the minimum coefficient mass on exactly graphs.

    target maps in-memory rooted type IDs to rational coefficients. A rational
    dual separator accompanies OutOfSpan. BudgetExceeded has no feasibility
    implication. The result includes a full-catalog optimality certificate.
    """
    if radius < 0 or not graphs:
        raise ValueError("A nonnegative radius and nonempty catalog are required")
    target = clean({t: Q(c) for t, c in target.items()})
    for t in target:
        g = TYPES.reps[t]
        ds = distances(g, 0)
        if len(ds) != g.n or max(ds.values()) > radius:
            raise ValueError("A target type is not a rooted ball at this radius")
    columns = [hist(g, radius) for g in graphs]
    rows = sorted(set(target).union(*(set(c) for c in columns)),
                  key=lambda t: (TYPES.reps[t].n, TYPES.reps[t].edges,
                                 TYPES.reps[t].rows))
    h = [[Q(col.get(t, 0)) for col in columns] for t in rows]
    b = [target.get(t, Q(0)) for t in rows]
    row_basis = pivot_columns(list(zip(*h)))
    a = [h[i] for i in row_basis]
    reduced_b = [b[i] for i in row_basis]
    rank = len(row_basis)
    first_basis = pivot_columns(a)
    inv = inverse([[row[j] for j in first_basis] for row in a])
    if inv is None:
        raise ArithmeticError("Internal rank inconsistency")
    first_c = matvec(inv, reduced_b)
    prediction = matvec([[row[j] for j in first_basis] for row in h], first_c)
    mismatch = next((i for i in range(len(rows)) if prediction[i] != b[i]), None)
    if mismatch is not None:
        # f = e_i - H[i,J] A^{-1} on the selected rows, so fH=0.
        f = [Q(0) for _ in rows]
        f[mismatch] = Q(1)
        for k, row_id in enumerate(row_basis):
            f[row_id] -= sum(h[mismatch][j] * inv[l][k]
                             for l, j in enumerate(first_basis))
        raise OutOfSpan(clean(dict(zip(rows, f))), sum(x*y for x, y in zip(f, b)))
    sizes = [g.n for g in graphs]
    work = 0
    for basis in combinations(range(len(graphs)), rank):
        work += 1
        if work > search_budget:
            raise BudgetExceeded("Independent-basis search exceeded search_budget")
        inv = inverse([[row[j] for j in basis] for row in a])
        if inv is None:
            continue
        coefficients = matvec(inv, reduced_b)
        zero_positions = [i for i, c in enumerate(coefficients) if not c]
        signs = [Q(1 if c > 0 else -1 if c < 0 else 0) for c in coefficients]
        # Try the interior choice for inactive coordinates first, then the
        # active choices sufficient to include an optimal dual vertex.
        choices = [(Q(0),) * len(zero_positions)]
        active_choices = product((Q(-1), Q(1)), repeat=len(zero_positions))
        for inactive in chain(choices, active_choices):
            work += 1
            if work > search_budget:
                raise BudgetExceeded("Dual-certificate search exceeded search_budget")
            trial = signs.copy()
            for i, sign in zip(zero_positions, inactive):
                trial[i] = sign
            # A_basis^T f = size * sign(c); use the transposed inverse.
            rhs = [sizes[j] * trial[i] for i, j in enumerate(basis)]
            dual_reduced = matvec(list(zip(*inv)), rhs)
            if any(abs(sum(dual_reduced[i] * a[i][j] for i in range(rank)))
                   > sizes[j] for j in range(len(graphs))):
                continue
            primal = [Q(0) for _ in graphs]
            for j, c in zip(basis, coefficients):
                primal[j] = c
            dual = [Q(0) for _ in rows]
            for i, value in zip(row_basis, dual_reduced):
                dual[i] = value
            cost = sum(n * abs(c) for n, c in zip(sizes, primal))
            dual_value = sum(x*y for x, y in zip(dual, b))
            if matvec(h, primal) != b or cost != dual_value:
                raise ArithmeticError("Invalid reconstruction certificate")
            return {
                "coefficients": primal, "dual": clean(dict(zip(rows, dual))),
                "cost": cost, "negative_mass": (cost - sum(b)) / 2,
                "rank": rank, "catalog_size": len(graphs), "search_work": work,
            }
    raise ArithmeticError("Feasible finite LP had no primal-dual basis certificate")


def encode_hist(values):
    return [{"rows": list(TYPES.reps[t].rows), "value": str(value)}
            for t, value in sorted(values.items(), key=lambda p: TYPES.reps[p[0]].rows)
            if value]


def encode_result(result, graphs, target, radius):
    return {
        "status": "optimal_on_catalog", "radius": radius,
        "target": encode_hist(target),
        "catalog": [{"rows": list(g.rows), "coefficient": str(c),
                     "normalized_coefficient": str(g.n*c)}
                    for g, c in zip(graphs, result["coefficients"])],
        "dual": encode_hist(result["dual"]),
        **{key: str(result[key]) for key in ("cost", "negative_mass")},
        **{key: result[key] for key in ("rank", "catalog_size", "search_work")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    target = defaultdict(Q)
    for item in data["target"]:
        if not isinstance(item["value"], (int, str)):
            raise ValueError("Target values must be integers or rational strings")
        target[TYPES.register(Graph(tuple(item["rows"])))] += Q(item["value"])
    target = clean(target)
    try:
        graphs = catalog(data["max_vertices"], data.get("max_degree"),
                         data.get("label_budget", 200000))
        result = reconstruct(target, graphs, data["radius"],
                             data.get("search_budget", 100000))
        output = encode_result(result, graphs, target, data["radius"])
    except BudgetExceeded as error:
        output = {"status": "budget_exhausted", "message": str(error)}
    except OutOfSpan as error:
        output = {"status": "outside_catalog_span", "message": str(error),
                  "separator": encode_hist(error.witness),
                  "separator_value": str(error.value)}
    text = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
