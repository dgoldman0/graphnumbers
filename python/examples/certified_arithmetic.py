"""End-to-end graph inverses and nonspectral moment certificates.

From the repository root::

    PYTHONPATH=python/src python3 python/examples/certified_arithmetic.py \
        --output python/results/certified_arithmetic.json

The reciprocal jet is checked only after a graph-algebra inverse has been
constructed. A formal reciprocal jet alone is not a graph unit certificate.
All comparisons use exact rational arithmetic; no eigenvalue calculation
or performance comparison is involved.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import json
from math import comb
from pathlib import Path

from graphlocal import (Finite, NeumannInverse, ROOT_DEGREE, certified_jet,
                        complete, cycle, joint_distribution, link_components,
                        rooted_cliques)

from cospectral_geometry import rook_graph, shrikhande_graph


def verify_certified_arithmetic():
    checks = 0

    def require(condition, description):
        nonlocal checks
        if not condition:
            raise AssertionError(description)
        checks += 1

    # Actual graph inverse: (1 + t(R-S))^(-1), with R,S normalized.
    # This small rational parameter keeps the required graph truncation at
    # degree two, within the generic exact-isomorphism search budget.
    t, epsilon = Q(1, 1024), Q(1, 100000)
    rook = Finite.from_graph(rook_graph(), normalize=True)
    shrikhande = Finite.from_graph(shrikhande_graph(), normalize=True)
    difference = rook - shrikhande
    source = -t * difference
    inverse = NeumannInverse(source, max_terms=24, max_vertices=256)
    certificate = inverse.approximation_certificate(1, 1, epsilon)
    nmax = certificate.truncation_degree
    require(inverse.mass == 1, "cospectral inverse has exact mass one")
    require(inverse.degree_bound is None, "inverse retains unbounded degree")
    require(certificate.source_error == 0, "finite source is exact")
    require(certificate.approximation.error <= epsilon, "local inverse meets tolerance")

    triangle_links = link_components(complete(3), "K3_link_components")
    cycle_links = link_components(cycle(6), "C6_link_components")
    actual = {}
    for rooted, coefficient in certificate.approximation.histogram.values.items():
        first, second = triangle_links(rooted.graph), cycle_links(rooted.graph)
        require(first % 2 == 0, "rook powers contribute pairs of triangular links")
        a, b = first // 2, second
        require(rooted.graph.rows[0].bit_count() == 6 * (a + b),
                "geometric exponents determine root degree")
        require((a, b) not in actual, "each geometric exponent pair has one rooted type")
        actual[a, b] = coefficient
    expected = {(a, n - a): Q(comb(n, a)) * (-t) ** a * t ** (n - a)
                for n in range(nmax + 1) for a in range(n + 1)}
    require(actual == expected, "runtime inverse has the full multinomial truncation")

    # Radius-one weighted norm is exactly sum_n (6n+1)(2|t|)^n.
    q = 2 * abs(t)
    full_weighted_norm = 1 / (1 - q) + 6 * q / (1 - q) ** 2
    truncated_weighted_norm = sum((Q(6 * n + 1) * q ** n
                                   for n in range(nmax + 1)), Q(0))
    exact_tail = full_weighted_norm - truncated_weighted_norm
    require(certificate.approximation.histogram.norm(1) == truncated_weighted_norm,
            "computed local norm matches disjoint-support formula")
    require(0 <= exact_tail <= certificate.tail_bound,
            "exact weighted tail is covered by the runtime certificate")
    require(inverse.variation_bound == 1 / (1 - q),
            "inverse variation certificate reaches the exact value for this family")

    # The finite target's formal reciprocal computes exact moments of this
    # already established inverse. The statistic counts K4s through a root.
    axis = rooted_cliques(4)
    element = 1 + t * difference
    reciprocal_jet = joint_distribution(element, (axis,)).jet(3).reciprocal()
    expected_moments = (Q(1), -2 * t, 8 * t ** 2 - 4 * t,
                        -48 * t ** 3 + 48 * t ** 2 - 8 * t)
    for order, expected_value in enumerate(expected_moments):
        require(reciprocal_jet.moment((order,)) == expected_value,
                f"reciprocal rooted-K4 moment of order {order}")

    # Full certificate interoperability on a smaller geometric inverse.
    # Sum_n (H/4)^n has degree n at its nth term. Jet coefficients are
    # M_j/j!, so the quadratic coefficient is 10/27, not 20/27.
    edge = Finite.from_graph(complete(2), normalize=True)
    edge_inverse = NeumannInverse(edge / 4, max_terms=40, max_vertices=128)
    edge_jet = certified_jet(edge_inverse, (ROOT_DEGREE,), 2, epsilon=epsilon)
    expected_coefficients = (Q(4, 3), Q(4, 9), Q(10, 27))
    for order, expected_value in enumerate(expected_coefficients):
        require(edge_jet.interval((order,)).contains(expected_value),
                f"actual inverse degree-jet enclosure of order {order}")
        require(edge_jet.interval((order,)).radius <= epsilon,
                f"degree-jet coefficient tolerance of order {order}")

    return {
        "success": True,
        "exact_checks": checks,
        "arithmetic": "rational coefficients, bounds, moments, and intervals",
        "cospectral_inverse": {
            "element": "(1 + t(U(rook)-U(Shrikhande)))^(-1)",
            "t": str(t),
            "epsilon": str(epsilon),
            "certificate": certificate.to_data(),
            "exact_weighted_tail": str(exact_tail),
            "exact_variation": str(1 / (1 - q)),
            "exact_radius_one_weighted_norm": str(full_weighted_norm),
            "multinomial_coefficients": [
                {"rook_power": a, "shrikhande_power": b,
                 "coefficient": str(c)}
                for (a, b), c in sorted(actual.items(), key=lambda entry:
                                       (sum(entry[0]), entry[0]))],
            "rooted_K4_reciprocal_jet": reciprocal_jet.to_data(),
            "rooted_K4_raw_moments": [str(value) for value in expected_moments],
        },
        "certified_degree_jet": {
            "element": "(1-H/4)^(-1), H=K2/2",
            "coefficient_convention": "M_j/j!",
            "expected_coefficients": [str(value) for value in expected_coefficients],
            "certificate": edge_jet.to_data(),
        },
        "scope": "End-to-end certificate checks, not a numerical speed benchmark. "
                 "The graph inverse is established independently of reciprocal moment jets.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the complete JSON report")
    args = parser.parse_args()
    report = verify_certified_arithmetic()
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
