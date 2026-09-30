"""Exact finite verification of tree cut reduction and planar mixed moments.

All arithmetic is integer or rational. Tree identities are compared in the
unrooted connected-component basis and independently in rooted histograms.
Buffered planar checks concern explicit finite squares; stabilization is
reported separately from any theorem about the infinite square lattice.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction as Q
from functools import lru_cache
from itertools import combinations
from math import factorial
from pathlib import Path

from graphlocal import LocalHistogram, cycle, graph, star
from graphlocal.defects import apply_edge_edits
from graphlocal.graphs import IsoGraph, components, distances


def edges(g):
    return tuple((u, v) for u in range(g.n) for v in g.neighbors(u) if u < v)


def normalized_cuts(cuts):
    cuts = tuple(sorted(tuple(sorted(edge)) for edge in cuts))
    if len(set(cuts)) != len(cuts):
        raise ValueError("Cut edges must be distinct")
    return cuts


@lru_cache(maxsize=None)
def deleted_graph(g, cuts):
    return apply_edge_edits(g, [(u, v, -1) for u, v in cuts])[0]


def subset_graphs(g, cuts):
    """Yield (-1)^(k-|S|), G-S for every subset S of the cut set."""
    cuts = normalized_cuts(cuts)
    for mask in range(1 << len(cuts)):
        subset = tuple(edge for j, edge in enumerate(cuts) if mask & (1 << j))
        yield (-1) ** (len(cuts) - mask.bit_count()), deleted_graph(g, subset)


@lru_cache(maxsize=None)
def graph_components(g):
    return tuple(IsoGraph(component) for component in components(g))


def interaction_components(g, cuts):
    result = {}
    for sign, modified in subset_graphs(g, cuts):
        for key in graph_components(modified):
            result[key] = result.get(key, 0) + sign
    return {key: coefficient for key, coefficient in result.items() if coefficient}


@lru_cache(maxsize=None)
def graph_histogram(g, radius):
    return LocalHistogram.from_graph(g, radius)


def interaction_histogram(g, cuts, radius):
    return LocalHistogram(radius, ((key.graph, sign * coefficient)
                                   for sign, modified in subset_graphs(g, cuts)
                                   for key, coefficient in graph_histogram(modified, radius).values.items()))


def quotient_leaf_cuts(g, cuts):
    """Return pendant edges in the tree quotient formed by contracting G-F.

    This routine checks the quotient is a tree; it refuses cyclic cut sets.
    """
    cuts = normalized_cuts(cuts)
    if not g.n or len(distances(g, 0)) != g.n:
        raise ValueError("The original graph must be connected and nonempty")
    modified = deleted_graph(g, cuts)
    blocks, block_of, unseen = [], {}, set(range(g.n))
    while unseen:
        block = tuple(distances(modified, min(unseen)))
        for u in block:
            block_of[u] = len(blocks)
        unseen.difference_update(block)
        blocks.append(block)
    if len(blocks) != len(cuts) + 1:
        raise ValueError("Cut quotient is not a tree")
    degrees = [0] * len(blocks)
    for u, v in cuts:
        if block_of[u] == block_of[v]:
            raise ValueError("Cut quotient contains a loop")
        degrees[block_of[u]] += 1
        degrees[block_of[v]] += 1
    return tuple((u, v) for u, v in cuts
                 if degrees[block_of[u]] == 1 or degrees[block_of[v]] == 1)


def unlabeled_trees(max_vertices):
    """Generate trees by leaf insertion, with exact isomorphism deduplication."""
    levels, previous = [], [graph(1)]
    for n in range(1, max_vertices + 1):
        if n > 1:
            unique = {}
            for old in previous:
                for u in range(old.n):
                    candidate = graph(n, edges(old) + ((u, n - 1),))
                    unique.setdefault(IsoGraph(candidate), candidate)
            previous = sorted(unique.values(), key=lambda g: g.rows)
        levels.append(tuple(previous))
    return tuple(levels)


def verify_trees(max_vertices=7):
    totals = {"cut_sets": 0, "component_basis_equalities": 0,
              "rooted_histogram_equalities": 0, "proper_internal_cut_reductions": 0}
    counts = []
    for level in unlabeled_trees(max_vertices):
        counts.append({"vertices": level[0].n, "unlabeled_trees": len(level)})
        for g in level:
            original_edges = edges(g)
            for mask in range(1, 1 << len(original_edges)):
                cuts = tuple(edge for j, edge in enumerate(original_edges) if mask & (1 << j))
                leaves = quotient_leaf_cuts(g, cuts)
                sign = (-1) ** (len(cuts) - len(leaves))
                actual = interaction_components(g, cuts)
                expected = {key: sign * coefficient
                            for key, coefficient in interaction_components(g, leaves).items()}
                if actual != expected:
                    raise ArithmeticError(f"Tree component identity failed: {g.rows}, {cuts}")
                totals["cut_sets"] += 1
                totals["component_basis_equalities"] += 1
                totals["proper_internal_cut_reductions"] += int(len(leaves) < len(cuts))
                for radius in sorted({0, 1, min(2, g.n - 1), g.n - 1}):
                    left = interaction_histogram(g, cuts, radius)
                    right = interaction_histogram(g, leaves, radius).scale(sign)
                    if left != right:
                        raise ArithmeticError(f"Tree histogram identity failed at r={radius}")
                    totals["rooted_histogram_equalities"] += 1
    return {"tree_counts": counts, **totals,
            "identity": "C_F(G)=(-1)^(|F|-|L|) C_L(G), L the pendant edges of the contracted tree quotient.",
            "scope": "Finite exhaustive verification through the stated order; the general identity still requires proof."}


def spider(lengths):
    rows, cuts, cursor = [], [], 1
    for length in lengths:
        previous = 0
        for _ in range(length):
            rows.append((previous, cursor))
            previous, cursor = cursor, cursor + 1
        cuts.append(rows[-1])
    return graph(cursor, rows), tuple(cuts)


def root_laplacian_moments(g, root, order):
    """Exact diagonal powers by sparse integer recurrence with actual degrees."""
    values, answer = {root: 1}, [1]
    for _ in range(order):
        updated = {}
        for u, coefficient in values.items():
            updated[u] = updated.get(u, 0) + g.rows[u].bit_count() * coefficient
            for v in g.neighbors(u):
                updated[v] = updated.get(v, 0) - coefficient
        values = {u: value for u, value in updated.items() if value}
        answer.append(values.get(root, 0))
    return tuple(answer)


def mixed_laplacian_moments(g, cuts, order=8):
    """Compute sum_S (-1)^(k-|S|) tr(L_(G-S)^j), j=0,...,order.

    For nonempty cuts, roots beyond ceil(order/2) from all edited endpoints
    have identical rooted local data and cancel. Integer recurrences use the
    full finite graph, so no artificial boundary degree enters the calculation.
    """
    cuts = normalized_cuts(cuts)
    if not cuts:
        raise ValueError("A mixed difference requires at least one cut")
    radius, affected = (order + 1) // 2, set()
    for endpoint in {u for edge in cuts for u in edge}:
        affected.update(distances(g, endpoint, radius))
    result = [0] * (order + 1)
    for sign, modified in subset_graphs(g, cuts):
        for root in affected:
            for j, value in enumerate(root_laplacian_moments(modified, root, order)):
                result[j] += sign * value
    return tuple(result)


def moment_record(name, g, cuts, order=8):
    moments = mixed_laplacian_moments(g, cuts, order)
    first = next((j for j, value in enumerate(moments) if value), None)
    return {"case": name, "vertices": g.n, "edges": g.edges,
            "cut_edges": [list(edge) for edge in cuts], "cut_count": len(cuts),
            "max_moment_order": order, "mixed_laplacian_moments": list(moments),
            "first_nonzero_order": first,
            "leading_heat_coefficient": (str(Q((-1) ** first * moments[first], factorial(first)))
                                         if first is not None else None),
            "heat_coefficients": [str(Q((-1) ** j * value, factorial(j)))
                                  for j, value in enumerate(moments)]}


def square_grid(side):
    return graph(side * side,
                 [(side * y + x, side * y + x + 1) for y in range(side) for x in range(side - 1)]
                 + [(side * y + x, side * (y + 1) + x) for y in range(side - 1) for x in range(side)])


def centered_grid_cuts(side, kind):
    center = (side // 2) * side + side // 2
    if kind == "Y3":
        return ((center, center - side), (center, center - 1), (center, center + 1))
    if kind == "plaquette4":
        return ((center, center + 1), (center + 1, center + side + 1),
                (center + side + 1, center + side), (center + side, center))
    raise ValueError("Unknown planar fixture")


PAIR_FIXTURES = (
    ("perpendicular_adjacent", ((0, 0), (0, 1)), (2, 24, 226, 1980, 16826), (0, 1)),
    ("collinear_adjacent", ((1, 0), (2, 0)), (2, 24, 218, 1840, 15212), (0, -1)),
    ("opposite_square_sides", ((0, 1), (1, 1)), (0, 0, 16, 320, 4176), (1, -2)),
    ("parallel_distance_two", ((0, 2), (1, 2)), (0, 0, 0, 0, 24), (2, 2)),
    ("separated_collinear", ((3, 0), (4, 0)), (0, 0, 0, 0, 6), (2, -1)),
)


def centered_pair_edges(side, second_edge):
    """Embed oriented (0,0)--(1,0) and the specified second edge."""
    center = side // 2

    def vertex(point):
        x, y = point
        if not 0 <= center + x < side or not 0 <= center + y < side:
            raise ValueError("Grid is too small for the pair fixture")
        return (center + y) * side + center + x

    return ((vertex((0, 0)), vertex((1, 0))),
            (vertex(second_edge[0]), vertex(second_edge[1])))


def incidence_cross_moments(g, first_edge, second_edge, order):
    """Exact b^T L^j c for the two oriented incidence vectors."""
    values, result = {second_edge[0]: 1, second_edge[1]: -1}, []
    for j in range(order + 1):
        result.append(values.get(first_edge[0], 0) - values.get(first_edge[1], 0))
        if j == order:
            break
        updated = {}
        for u, coefficient in values.items():
            updated[u] = updated.get(u, 0) + g.rows[u].bit_count() * coefficient
            for v in g.neighbors(u):
                updated[v] = updated.get(v, 0) - coefficient
        values = {u: value for u, value in updated.items() if value}
    return tuple(result)


def planar_pair_records(sides=(17, 21), order=8):
    """Independently verify the planar draft's five pair moment rows."""
    records = []
    for name, second_edge, expected, first_cross in PAIR_FIXTURES:
        finite = []
        for side in sides:
            g, cuts = square_grid(side), centered_pair_edges(side, second_edge)
            record = moment_record(f"finite_square_{side}_{name}", g, cuts, order)
            record["grid_side"] = side
            cross = incidence_cross_moments(g, cuts[0], cuts[1], 4)
            first = next(j for j, value in enumerate(cross) if value)
            if tuple(record["mixed_laplacian_moments"][2:7]) != expected:
                raise ArithmeticError(f"Planar pair table disagrees for {name}, side {side}")
            if (first, cross[first]) != first_cross:
                raise ArithmeticError(f"First incidence cross moment disagrees for {name}")
            leading_order = 2 * first + 2
            leading_moment = leading_order * cross[first] ** 2
            if (record["first_nonzero_order"] != leading_order
                    or record["mixed_laplacian_moments"][leading_order] != leading_moment):
                raise ArithmeticError(f"Pair leading-order formula failed for {name}")
            record.update({"incidence_cross_moments": list(cross),
                           "first_nonzero_cross_order": first,
                           "first_nonzero_cross_value": cross[first],
                           "leading_order_formula_verified": True})
            finite.append(record)
        if any(row["mixed_laplacian_moments"] != finite[0]["mixed_laplacian_moments"] for row in finite[1:]):
            raise ArithmeticError(f"Buffered planar pair did not stabilize for {name}")
        records.append({"case": name, "first_edge_coordinates": [[0, 0], [1, 0]],
                        "second_edge_coordinates": [list(point) for point in second_edge],
                        "draft_moments_I2_through_I6": list(expected),
                        "finite_square_verifications": finite,
                        "buffered_moments_stabilized": True})
    return records


def build_report():
    trees = verify_trees(7)
    expected_counts = [1, 1, 1, 2, 3, 6, 11]
    if [row["unlabeled_trees"] for row in trees["tree_counts"]] != expected_counts:
        raise ArithmeticError("Tree generator did not recover the expected small catalog")
    stars = [moment_record(f"star_{k}_all_leaf_cuts", star(k), edges(star(k))) for k in (2, 3, 4)]
    spiders = []
    for lengths in ((2, 2, 2), (3, 2, 1), (2, 2, 2, 2)):
        g, leaves = spider(lengths)
        full = interaction_components(g, edges(g))
        sign = (-1) ** (g.edges - len(leaves))
        expected = {key: sign * coefficient for key, coefficient in interaction_components(g, leaves).items()}
        if full != expected:
            raise ArithmeticError("Branched spider parity reduction failed")
        entry = moment_record("spider_" + "_".join(map(str, lengths)), g, leaves, 12)
        entry.update({"arm_lengths": list(lengths), "full_cut_count": g.edges,
                      "leaf_cut_count": len(leaves), "full_to_leaf_sign": sign,
                      "full_component_identity_verified": True})
        spiders.append(entry)
    planar = [moment_record("triangle_all_three_edges", cycle(3), edges(cycle(3))),
              moment_record("square_all_four_edges", cycle(4), edges(cycle(4)))]
    for kind in ("Y3", "plaquette4"):
        for side in (3, 13, 17):
            cuts = centered_grid_cuts(side, kind)
            planar.append(moment_record(f"finite_square_{side}_{kind}", square_grid(side), cuts))
        a, b = planar[-2:]
        if a["mixed_laplacian_moments"] != b["mixed_laplacian_moments"]:
            raise ArithmeticError(f"Buffered {kind} moments did not stabilize")
    center_fixture = next(row for row in planar if row["case"] == "finite_square_13_Y3")
    g, cuts = square_grid(13), centered_grid_cuts(13, "Y3")
    pair_witnesses = []
    for pair in combinations(cuts, 2):
        moments = mixed_laplacian_moments(g, pair, 3)
        if moments[2] == 0 or center_fixture["mixed_laplacian_moments"][2] != 0:
            raise ArithmeticError("Expected order-two witness separating Y interaction from each pair")
        pair_witnesses.append({"pair": [list(edge) for edge in pair],
                               "second_mixed_laplacian_moment": moments[2],
                               "three_cut_second_moment": 0})
    rejected = False
    try:
        quotient_leaf_cuts(cycle(3), edges(cycle(3)))
    except ValueError:
        rejected = True
    if not rejected:
        raise ArithmeticError("Cyclic quotient was incorrectly treated as a tree")
    return {
        "arithmetic": "Exact Python integers and Fractions; rooted and unrooted isomorphism use exact backtracking.",
        "interaction_convention": "C_F(G)=sum_(S subset F) (-1)^(|F|-|S|) (G-S). Heat is the same linear combination of unnormalized traces exp(-tL).",
        "tree_verification": trees, "star_interactions": stars, "spider_interactions": spiders,
        "planar_fixtures": planar,
        "planar_two_edge_interactions": planar_pair_records(),
        "Y_cannot_equal_positive_or_negative_of_any_cut_pair": pair_witnesses,
        "triangle_tree_formula_rejected": rejected,
        "buffered_grid_stabilization": {
            "sides": [13, 17], "moment_orders": list(range(9)),
            "fixtures": ["Y3", "plaquette4"], "exactly_equal": True,
            "scope": "These are exact finite-square equalities. Infinite-lattice conclusions additionally use finite-propagation reasoning in the mathematical note.",
        },
        "conclusions_supported_by_finite_checks": [
            "Internal tree cuts reduce to the leaf-cut interaction with the predicted parity in all tested trees.",
            "Stars have genuine three- and four-cut responses: leading heat terms t^3 and t^4.",
            "A triangular cycle has leading heat term -t^3, so tree-like positive leading signs do not extend to all planar cuts.",
            "The centered square-grid Y has leading term t^3 and cannot equal either sign of any selected-edge pair interaction.",
            "The square plaquette four-cut interaction starts at t^4/3.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))
