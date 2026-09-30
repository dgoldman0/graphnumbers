"""Exact fixtures for higher defect interactions and geometric cancellation.

Dense integer subset traces provide an independent oracle for the incidence
runtime. Direct cyclic-word enumeration separates contributions by the number
of defect occurrences, including repetitions. Only the finite fixtures below
are enumerated; no claim of an exhaustive graph classification is made.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction as Q
from itertools import product
from math import factorial
from pathlib import Path

from graphlocal import EdgeInteraction, controlled_heat, graph
from graphlocal.interaction_moments import interaction_moments, tree_interaction_leading


FIXTURES = (
    ("repeated_defect_triple", 4,
     ((0, 1), (0, 2), (0, 3), (1, 2)),
     ((0, 1), (0, 3), (1, 2)),
     (0, 0, 0, 0, 4, 40, 252, 1288, 5852)),
    ("cancelled_triple", 6,
     ((3, 5), (2, 4), (1, 5), (0, 5), (1, 2), (1, 3), (3, 4), (4, 5)),
     ((3, 5), (2, 4), (1, 5)),
     (0, 0, 0, 0, 0, 0, 0, -56, -1288)),
    ("cancelled_quadruple", 5,
     ((0, 1), (0, 2), (0, 4), (1, 2), (1, 3)),
     ((0, 1), (0, 2), (0, 4), (1, 3)),
     (0, 0, 0, 0, 0, 0, 0, 28, 520)),
    ("time_sign_reversal", 5,
     ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2)),
     ((0, 1), (0, 3), (0, 4)),
     (0, 0, 0, -6, -84, -740, -5316, -34146, -204732)),
    ("binary_quartet_tree", 6,
     ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5)),
     ((0, 2), (0, 3), (1, 4), (1, 5)),
     None),
)


def fixture(name):
    for item in FIXTURES:
        if item[0] == name:
            _, n, edges, cuts, expected = item
            return graph(n, edges), cuts, expected
    raise KeyError(name)


def dense_laplacian(g, cuts=()):
    cuts = {tuple(sorted(edge)) for edge in cuts}
    matrix = [[0] * g.n for _ in range(g.n)]
    for u in range(g.n):
        for v in g.neighbors(u):
            if u < v and (u, v) not in cuts:
                matrix[u][u] += 1
                matrix[v][v] += 1
                matrix[u][v] -= 1
                matrix[v][u] -= 1
    return matrix


def multiply(left, right):
    return [[sum(a * b for a, b in zip(row, column))
             for column in zip(*right)] for row in left]


def subset_moments(g, cuts, order):
    """Full edited-subset matrices, independently of the rank-one algorithm."""
    result = [0] * (order + 1)
    for mask in range(1 << len(cuts)):
        chosen = [edge for i, edge in enumerate(cuts) if mask & (1 << i)]
        matrix = dense_laplacian(g, chosen)
        power = [[int(i == j) for j in range(g.n)] for i in range(g.n)]
        sign = (-1) ** (len(cuts) - mask.bit_count())
        for degree in range(order + 1):
            result[degree] += sign * sum(power[i][i] for i in range(g.n))
            power = multiply(power, matrix)
    return tuple(result)


def cross_moments(g, cuts, order):
    """Incidence moments using dense integer powers and oriented endpoints."""
    laplacian = dense_laplacian(g)
    power = [[int(i == j) for j in range(g.n)] for i in range(g.n)]
    result = []
    for _ in range(order + 1):
        result.append(tuple(tuple(power[u][x] - power[u][y]
                                  - power[v][x] + power[v][y]
                                  for x, y in cuts) for u, v in cuts))
        power = multiply(power, laplacian)
    return tuple(result)


def compositions(total, length):
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, length - 1):
            yield (first,) + rest


def cyclic_contributions(g, cuts, degree):
    """Directly enumerate cyclic words of one fixed total degree.

    The key is the number of defect letters, which can exceed len(cuts).
    The return value gives each letter-count contribution to the trace.
    """
    count = len(cuts)
    cross = cross_moments(g, cuts, degree)
    answer = {}
    for letters in range(count, degree + 1):
        gaps = tuple(compositions(degree - letters, letters))
        total = 0
        for labels in product(range(count), repeat=letters):
            if len(set(labels)) != count:
                continue
            for powers in gaps:
                coefficient = 1
                for i, power in enumerate(powers):
                    coefficient *= cross[power][labels[i]][labels[(i + 1) % letters]]
                    if not coefficient:
                        break
                total += coefficient
        answer[letters] = Q(degree, letters) * (-1) ** letters * total
    return answer


def record(name, order=8):
    g, cuts, expected = fixture(name)
    value = EdgeInteraction(g, [(u, v, -1) for u, v in cuts], reduce_bridges=False)
    exact = subset_moments(g, cuts, order)
    if expected is not None and exact[:len(expected)] != expected[:len(exact)]:
        raise ArithmeticError(f"Fixture changed: {name}")
    incidence = interaction_moments(value, order)
    if exact != incidence.laplacian:
        raise ArithmeticError(f"Rank-one moments differ from subset traces: {name}")
    first = next((j for j, coefficient in enumerate(exact) if coefficient), None)
    data = {"case": name, "vertices": g.n,
            "edges": [[u, v] for u in range(g.n) for v in g.neighbors(u) if u < v],
            "oriented_cut_edges": [list(edge) for edge in cuts],
            "mixed_laplacian_moments": list(exact),
            "first_nonzero_order_in_computed_range": first,
            "leading_heat_coefficient": (str(Q((-1) ** first * exact[first], factorial(first)))
                                         if first is not None else None),
            "rank_one_runtime_agrees": True,
            "runtime_work_units": incidence.work}
    check_degrees = {"repeated_defect_triple": (4,), "cancelled_triple": (6,),
                     "cancelled_quadruple": (5, 6), "binary_quartet_tree": (6,)}
    if name in check_degrees:
        grouped = {}
        for degree in check_degrees[name]:
            terms = cyclic_contributions(g, cuts, degree)
            if sum(terms.values()) != exact[degree]:
                raise ArithmeticError(f"Cyclic enumeration mismatch: {name}, degree {degree}")
            grouped[str(degree)] = {str(k): str(v) for k, v in terms.items()}
        data["cyclic_contributions_by_defect_letter_count"] = grouped
    return data


def sign_reversal_certificates(epsilon="1e-10"):
    g, cuts, _ = fixture("time_sign_reversal")
    value = EdgeInteraction(g, [(u, v, -1) for u, v in cuts])
    certificates = tuple(controlled_heat(value, time, epsilon) for time in (1, 3))
    if certificates[0].interval.lower <= 0 or certificates[1].interval.upper >= 0:
        raise ArithmeticError("The rational intervals failed to certify opposite signs")
    return certificates


def spanning_tree_data(g, cuts):
    """Build the spanning subtree by unique paths, without leaf pruning."""
    protected = {u for edge in cuts for u in edge}
    root = min(protected)
    parents, queue = {root: None}, [root]
    for u in queue:
        for v in g.neighbors(u):
            if v not in parents:
                parents[v] = u
                queue.append(v)
    selected = set()
    for u in protected:
        while parents[u] is not None:
            v = parents[u]
            selected.add(tuple(sorted((u, v))))
            u = v
    degrees = [0] * g.n
    for u, v in selected:
        degrees[u] += 1
        degrees[v] += 1
    leaves, branching = sum(d == 1 for d in degrees), 1
    for degree in degrees:
        if degree >= 2:
            branching *= factorial(degree - 1)
    return len(selected), leaves, branching


def tree_catalog_verification(max_vertices=6):
    """All nonempty cut sets of every unlabeled tree through the given size.

    Each deleted-subset trace is computed once by full integer matrices.
    A Boolean-lattice Mobius transform then yields every mixed trace.
    """
    from branch_planar_verification import edges, unlabeled_trees
    tree_counts, cut_sets, leading_orders = [], 0, {}
    positive, negative = 0, 0
    for level in unlabeled_trees(max_vertices):
        tree_counts.append({"vertices": level[0].n, "unlabeled_trees": len(level)})
        for g in level:
            all_edges = edges(g)
            order = max(1, 2 * (g.n - 1))
            table = []
            for mask in range(1 << len(all_edges)):
                cuts = [edge for i, edge in enumerate(all_edges) if mask & (1 << i)]
                matrix = dense_laplacian(g, cuts)
                power = [[int(i == j) for j in range(g.n)] for i in range(g.n)]
                traces = []
                for _ in range(order + 1):
                    traces.append(sum(power[i][i] for i in range(g.n)))
                    power = multiply(power, matrix)
                table.append(traces)
            for bit in range(len(all_edges)):
                for mask in range(1 << len(all_edges)):
                    if mask & (1 << bit):
                        lower = table[mask ^ (1 << bit)]
                        table[mask] = [a - b for a, b in zip(table[mask], lower)]
            for mask in range(1, 1 << len(all_edges)):
                cuts = tuple(edge for i, edge in enumerate(all_edges) if mask & (1 << i))
                value = EdgeInteraction(g, [(u, v, -1) for u, v in cuts])
                predicted = tree_interaction_leading(value)
                if len(cuts) == 1:
                    data, degree, coefficient = (1, 1, 1), 1, Q(2)
                else:
                    span, leaves, branching = spanning_tree_data(g, cuts)
                    degree = 2 * span - leaves
                    coefficient = Q((-1) ** (len(cuts) - leaves) * branching,
                                    factorial(degree - 1))
                    data = (span, leaves, branching)
                if data != (predicted.spanning_edges, predicted.terminal_edges,
                            predicted.branching_factor):
                    raise ArithmeticError("Path-union and pruned spanning trees differ")
                expected_moment = (-1) ** degree * factorial(degree) * coefficient
                moments = table[mask]
                if (predicted.order != degree or predicted.heat_coefficient != coefficient
                        or any(moments[:degree]) or moments[degree] != expected_moment):
                    raise ArithmeticError(f"Tree leading formula failed: {g.rows}, {cuts}")
                cut_sets += 1
                leading_orders[str(degree)] = leading_orders.get(str(degree), 0) + 1
                positive += int(coefficient > 0)
                negative += int(coefficient < 0)
    return {"tree_counts": tree_counts, "nonempty_cut_sets_verified": cut_sets,
            "leading_order_counts": leading_orders,
            "positive_leading_coefficients": positive,
            "negative_leading_coefficients": negative,
            "verification": "Independent integer subset traces and path-union spanning trees."}


def verification():
    records = [record(item[0]) for item in FIXTURES]
    certificates = sign_reversal_certificates()
    sign_data = []
    for certificate in certificates:
        sign_data.append({"time": str(certificate.time),
                          "rational_interval": certificate.interval.to_data(),
                          "decimal_interval_for_display": [float(certificate.interval.lower),
                                                           float(certificate.interval.upper)],
                          "truncation_order": certificate.steps,
                          "moment_profile": [str(a) for a in certificate.moment_profile]})
    return {"scope": "Five explicit finite graph fixtures; exact integer/rational verification.",
            "fixtures": records,
            "exhaustive_tree_leading_terms": tree_catalog_verification(),
            "time_sign_reversal_certificates": sign_data,
            "sign_reversal_conclusion": "The analytic heat interaction has a zero between t=1 and t=3."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = verification()
    text = json.dumps(data, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
