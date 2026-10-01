"""Finite exact comparisons of regular-tree and square-lattice edge cuts.

Buffered finite graphs give stabilized local fixtures. Product checks have
small fixed powers and radii; they do not establish an entire-function
subalgebra for the planar defect. An explicit planar witness shows why the
cut-line square/sphere coordinates do not directly transfer.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import json
from pathlib import Path

from graphlocal import (LocalHistogram, SparseEdgeDifference, cartesian, complete,
                        cycle, disjoint_union, graph, path, star)
from graphlocal.graphs import ball, components, distances, integer, isomorphic
from unbounded_inverse_verification import (
    additive_coordinates, compositions, monomial_graph, series_log,
    sphere_series, square_count)


def regular_tree(degree, depth):
    """Two rooted (degree-1)-ary trees joined at their roots, labeled 0,1."""
    integer(degree, "regular degree", 2)
    integer(depth, "branch depth")
    edges, frontier, n = [(0, 1)], [0, 1], 2
    for _ in range(depth):
        following = []
        for u in frontier:
            for _ in range(degree - 1):
                edges.append((u, n))
                following.append(n)
                n += 1
        frontier = following
    return graph(n, edges)


def tree_cut(degree, radius, extra_depth=0):
    integer(radius, "radius", 1)
    integer(extra_depth, "extra depth")
    before = regular_tree(degree, 2 * radius - 1 + extra_depth)
    return SparseEdgeDifference(before, [(0, 1, -1)])


def lattice_cut(radius, extra_margin=0):
    integer(radius, "radius", 1)
    integer(extra_margin, "extra margin")
    side = 4 * radius + 5 + 2 * extra_margin
    center = side // 2
    index = lambda x, y: x * side + y
    edges = [(index(x, y), index(x + 1, y))
             for x in range(side - 1) for y in range(side)]
    edges += [(index(x, y), index(x, y + 1))
              for x in range(side) for y in range(side - 1)]
    return (SparseEdgeDifference(graph(side * side, edges),
                                 [(index(center, center), index(center + 1, center), -1)]),
            side, center)


def tree_atoms(degree, radius):
    source = tree_cut(degree, radius)
    representatives = {}
    for u, distance in distances(source.after, 0, radius - 1).items():
        representatives.setdefault(distance, u)
    return tuple(ball(source.after, representatives[j], radius) for j in range(radius)) + (
        ball(source.before, 0, radius),)


def tree_coordinates(g, radius, degree):
    """Independent generalization of the cut-line square/sphere coordinates."""
    integer(radius, "radius", 2)
    integer(degree, "regular degree", 2)
    d = g.rows[0].bit_count()
    number = Q((2 * degree - 1) * d - d*d + 2 * square_count(g), degree * (degree - 1))
    background = (Q(1),) + tuple(Q(degree * (degree - 1) ** (n-1))
                                for n in range(1, radius + 1))
    residual = [a - number*b for a, b in zip(series_log(sphere_series(g, radius)),
                                            series_log(background))]
    result = []
    for j in range(radius):
        basis = series_log((Q(1),) + tuple(Q(0) if n <= j else Q((-1) ** (n-j))
                                          for n in range(1, radius + 1)))
        coefficient = -residual[j+1]
        result.append(coefficient)
        residual = [a - coefficient*b for a, b in zip(residual, basis)]
    if any(residual):
        raise AssertionError("Tree sphere-log coordinates left a residual")
    return tuple(result) + (number - sum(result),)


def marginal_summary(histogram):
    return {"types": len(histogram.values), "variation": str(histogram.norm(0)),
            "largest_ball": max(key.graph.n for key in histogram.values),
            "terms": [{"coefficient": str(coefficient), "vertices": key.graph.n,
                       "edges": key.graph.edges, "root_degree": key.graph.rows[0].bit_count(),
                       "root_squares": square_count(key.graph),
                       "sphere_sizes": list(map(int, sphere_series(key.graph, histogram.radius)))}
                      for key, coefficient in histogram.values.items()]}


def planar_coordinate_witness():
    source, side, center = lattice_cut(3)
    index = lambda x, y: (center + x) * side + center + y
    roots = ((-1, -1), (0, -2))
    left, right = (ball(source.after, index(*root), 3) for root in roots)
    baseline = ball(source.before, index(*roots[0]), 3)
    return roots, left, right, baseline


def regular_face(g, radius):
    """All interior degrees equal the root degree; boundary degrees are free."""
    degree = g.rows[0].bit_count()
    return all(g.rows[u].bit_count() == degree
               for u, distance in distances(g, 0).items() if distance < radius)


def root_edge_graph(g):
    """Neighbors are joined exactly when no additional common neighbor exists."""
    neighbors = tuple(g.neighbors(0))
    return graph(len(neighbors), ((i, j) for i in range(len(neighbors))
                                 for j in range(i + 1, len(neighbors))
                                 if not (g.rows[neighbors[i]] & g.rows[neighbors[j]] & ~1)))


def regular_projection(histogram):
    return LocalHistogram(histogram.radius,
                          ((key, coefficient) for key, coefficient in histogram.values.items()
                           if regular_face(key.graph, histogram.radius)))


def laplacian_traces_full_matrix(g, order):
    """Exact traces from full integer matrix powers, with sparse left factor L."""
    n = g.n
    power = [[int(i == j) for j in range(n)] for i in range(n)]
    traces = [n]
    for _ in range(order):
        following = []
        for u in range(n):
            row = [g.rows[u].bit_count() * value for value in power[u]]
            for v in g.neighbors(u):
                row = [a - b for a, b in zip(row, power[v])]
            following.append(row)
        power = following
        traces.append(sum(power[u][u] for u in range(n)))
    return tuple(traces)


def verify_defect_breadth():
    checks, histograms = 0, {}

    def require(condition, label):
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1

    marginals = []
    for radius in range(1, 5):
        for degree in (3, 4):
            source = tree_cut(degree, radius)
            value = source.local(radius)
            histograms[f"B{degree}", radius] = value
            total = sum((degree - 1) ** j for j in range(radius))
            require(value.norm(0) == 4 * total, "tree-cut marginal has expected exact variation")
            require(len(value.values) == radius + 1, "tree-cut marginal has one atom at each depth and baseline")
            atoms = tree_atoms(degree, radius)
            expected = LocalHistogram(radius, tuple((atom, 2 * (degree - 1) ** j)
                                                     for j, atom in enumerate(atoms[:-1]))
                                      + ((atoms[-1], -2 * total),))
            require(value == expected, "direct sparse edit equals independently weighted tree atoms")
            if radius <= 3:
                require(value == tree_cut(degree, radius, extra_depth=1).local(radius),
                        "tree-cut histogram is unchanged by an extra generation")
            marginals.append({"family": f"B{degree}", "radius": radius,
                              "buffer_vertices": source.before.n, **marginal_summary(value)})
        source, _, _ = lattice_cut(radius)
        value = source.local(radius)
        histograms["P", radius] = value
        lens = [(x, y) for x in range(-radius, radius + 2) for y in range(-radius, radius + 1)
                if abs(x) + abs(y) <= radius and abs(x-1) + abs(y) <= radius]
        require(len(lens) == 2 * radius ** 2, "lattice endpoint-ball lens has 2r squared points")
        require(value.norm(0) == 2 * len(lens), "planar marginal variation equals twice the changed-root count")
        require(len(value.values) == 1 + radius * (radius + 1) // 2,
                "observed planar type count matches its low-radius fixture")
        require(value == lattice_cut(radius, extra_margin=1)[0].local(radius),
                "planar histogram is unchanged by a larger grid buffer")
        marginals.append({"family": "P", "radius": radius,
                          "buffer_vertices": source.before.n, **marginal_summary(value)})

    coordinate_fixtures = 0
    for degree in (3, 4):
        radius = 2
        atoms = tree_atoms(degree, radius)
        for power in range(3):
            for exponents in compositions(power, radius + 1):
                value = monomial_graph(atoms, exponents, radius)
                require(tree_coordinates(value, radius, degree) == exponents,
                        "generalized tree coordinates recover product multiplicities")
                coordinate_fixtures += 1

    products = []
    pairs = (("B3", "B3"), ("B4", "B4"), ("P", "P"), ("B3", "P"), ("B3", "B4"))
    for radius in (1, 2):
        for a, b in pairs:
            left, right = histograms[a, radius], histograms[b, radius]
            product = left.multiply(right, max_vertices=500)
            expected = left.norm(0) * right.norm(0)
            require(product.norm(0) == expected, "low-radius product attains its total-variation upper bound")
            products.append({"factors": [a, b], "radius": radius,
                             "variation": str(product.norm(0)), "upper_bound": str(expected),
                             "types": len(product.values)})
    # A planar quadratic test at larger radius; higher powers are deliberately excluded.
    planar_square = histograms["P", 3].multiply(histograms["P", 3], max_vertices=500)
    require(planar_square.norm(0) == 1296 and len(planar_square.values) == 28,
            "radius-three planar square fixture has exact norm and support size")
    products.append({"factors": ["P", "P"], "radius": 3,
                     "variation": "1296", "upper_bound": "1296", "types": 28})

    combinations = []
    for radius in (1, 2):
        for a, b, sign in (("B3", "P", 1), ("B3", "P", -1), ("B4", "P", -1)):
            value = histograms[a, radius] + histograms[b, radius].scale(sign)
            if radius == 1:
                expected = {("B3", "P", 1): 4, ("B3", "P", -1): 8,
                            ("B4", "P", -1): 0}[a, b, sign]
            else:
                expected = histograms[a, radius].norm(0) + histograms[b, radius].norm(0)
            require(value.norm(0) == expected, "mixed family marginal cancellation matches exact fixture")
            power_value = LocalHistogram(radius, ((complete(1), Q(1)),))
            for power in range(1, 4):
                power_value = power_value.multiply(value, max_vertices=500)
                require(power_value.norm(0) == expected ** power,
                        "low-radius mixed polynomial power has stated exact norm")
                combinations.append({"families": [a, b], "second_sign": sign,
                                     "radius": radius, "power": power,
                                     "variation": str(power_value.norm(0)),
                                     "types": len(power_value.values)})

    roots, left, right, baseline = planar_coordinate_witness()
    signatures = [(g.rows[0].bit_count(), square_count(g), sphere_series(g, 3))
                  for g in (left, right, baseline)]
    require(signatures[0] == signatures[1] == signatures[2],
            "two different planar defect atoms share square/sphere data with the baseline")
    require(not isomorphic(left, right, rooted=True), "explicit planar atoms are rooted nonisomorphic")
    require(left.n == right.n == baseline.n == 25 and left.edges == right.edges == 35 and baseline.edges == 36,
            "planar witness has unchanged sphere sizes and one fewer edge")
    require(additive_coordinates(left, 3) == additive_coordinates(right, 3)
            == additive_coordinates(baseline, 3) == (0, 0, 0, 2),
            "cut-line square/sphere coordinates do not separate planar positive atoms and negative baseline")

    # Independent tests of the alternative regular-face / root-edge construction.
    paw = graph(4, ((0, 1), (1, 2), (2, 0), (0, 3)))
    fixtures = (complete(1), complete(2), complete(3), cycle(4), path(3), star(3), paw)
    root_edge_pairs = 0
    radius = 2
    for g in fixtures:
        for h in fixtures:
            product = cartesian(g, h)
            for u in range(g.n):
                for v in range(h.n):
                    a, b = ball(g, u, radius), ball(h, v, radius)
                    joined = ball(product, u*h.n + v, radius)
                    require(regular_face(joined, radius)
                            == (regular_face(a, radius) and regular_face(b, radius)),
                            "regular-face indicator multiplies on arbitrary rooted fixtures")
                    require(isomorphic(root_edge_graph(joined),
                                       disjoint_union(root_edge_graph(a), root_edge_graph(b))),
                            "auxiliary root-edge graph of a product is the disjoint union")
                    root_edge_pairs += 1

    for family in ("B3", "B4", "P"):
        for radius in (2, 3, 4):
            value = histograms[family, radius]
            projected = regular_projection(value)
            require(len(projected.values) == 1 and projected.norm(0) == value.norm(0)/2,
                    "regular-face projection retains exactly the negative background")
            for key, coefficient in value.values.items():
                require(regular_face(key.graph, radius) == (coefficient < 0),
                        "every positive single-cut atom is outside the regular face")
            background_graph = next(iter(projected.values)).graph
            expected = (disjoint_union(complete(2), complete(2)) if family == "P"
                        else complete(int(family[1:])))
            require(isomorphic(root_edge_graph(background_graph), expected),
                    "background root-edge components distinguish tree degree and grid axes")

    # All coefficients are independent polynomial input, not fitted to the histograms.
    radius = 2
    coefficients = {(0, 0, 0): Q(1), (1, 0, 0): Q(-2), (0, 0, 1): Q(1, 3),
                    (1, 0, 1): Q(2), (0, 2, 0): Q(-1, 2), (1, 1, 0): Q(3, 5)}
    bases = tuple(histograms[family, radius] for family in ("B3", "B4", "P"))
    polynomial = LocalHistogram(radius)
    lower = Q(0)
    expected_background_coefficients = {}
    weights = tuple(base.norm(0)/2 for base in bases)
    for alpha, coefficient in coefficients.items():
        monomial = LocalHistogram(radius, ((complete(1), Q(1)),))
        background_coefficient = coefficient
        for base, exponent, weight in zip(bases, alpha, weights):
            for _ in range(exponent):
                monomial = monomial.multiply(base, max_vertices=500)
            background_coefficient *= (-weight) ** exponent
        polynomial += monomial.scale(coefficient)
        lower += abs(background_coefficient)
        expected_background_coefficients[alpha] = background_coefficient
    projection = regular_projection(polynomial)
    actual_background_coefficients = {}
    for key, coefficient in projection.values.items():
        component_sizes = [part.n for part in components(root_edge_graph(key.graph))]
        require(all(size in (2, 3, 4) for size in component_sizes),
                "mixed background has only the prescribed auxiliary component sizes")
        alpha = (component_sizes.count(3), component_sizes.count(4), component_sizes.count(2)//2)
        actual_background_coefficients[alpha] = coefficient
    require(actual_background_coefficients == expected_background_coefficients,
            "regular-face polynomial coefficients retain each multiindex independently")
    require(projection.norm(0) == lower <= polynomial.norm(0),
            "multivariable weighted coefficient lower bound equals projected variation")

    moments = {}
    for family, source in (("B4", tree_cut(4, 2)), ("P", lattice_cut(2)[0])):
        before = laplacian_traces_full_matrix(source.before, 4)
        after = laplacian_traces_full_matrix(source.after, 4)
        values = tuple(a-b for a, b in zip(after, before))
        expected = (0, -2, -16, -116, -832 if family == "B4" else -848)
        require(values == expected, "full-matrix Laplacian powers independently verify fourth-moment geometry")
        moments[family] = list(values)

    return {"success": True, "checks": checks, "marginals": marginals,
            "tree_coordinate_product_fixtures": coordinate_fixtures,
            "products": products, "mixed_polynomials": combinations,
            "regular_face_root_edge_pairs": root_edge_pairs,
            "mixed_coefficient_lower_bound": {"radius": radius,
                "axes": ["B3", "B4", "P"], "coefficient_norm": str(lower),
                "projected_variation": str(projection.norm(0)),
                "full_variation": str(polynomial.norm(0)),
                "terms": [{"multiindex": list(alpha), "coefficient": str(coefficient)}
                          for alpha, coefficient in coefficients.items()]},
            "full_matrix_relative_laplacian_moments": moments,
            "planar_coordinate_obstruction": {"roots_relative_to_deleted_edge": roots,
                "deleted_edge": ((0, 0), (1, 0)), "radius": 3,
                "sphere_sizes": [1, 4, 8, 12], "root_squares": 4,
                "defect_edges": [35, 35], "baseline_edges": 36,
                "equal_coordinates": [0, 0, 0, 2],
                "rooted_defect_rows": [list(left.rows), list(right.rows)],
                "interpretation": "Same sphere/square characters see these distinct signed types identically; the cut-line proof does not transfer unchanged."},
            "scope": "Exact finite-radius comparisons and witnesses; universal joint entire-function and spectral claims rely on the companion proofs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_defect_breadth()
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
