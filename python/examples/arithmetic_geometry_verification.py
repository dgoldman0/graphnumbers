"""Exact, independent fixtures for arithmetic and nonspectral graph geometry.

Only finite graphs, integer counts and rational coefficient arithmetic are
used. Infinite inversion and character claims are proved in the accompanying
research notes; the finite checks here verify their explicit coefficients and
witnesses, without treating finite truncations as proofs of convergence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as Q
from itertools import combinations
import json
from math import comb
from pathlib import Path

from graphlocal import (cartesian, complete, cycle, disjoint_union, graph,
                       isomorphic, path, star)
from graphlocal.graphs import IsoGraph, components, induced, integer
from cospectral_geometry import rook_graph, shrikhande_graph, verify_geometry


def root_link(g, root):
    return induced(g, tuple(g.neighbors(root)))


def link_component_counts(g, root):
    return Counter(IsoGraph(part) for part in components(root_link(g, root)))


def joint_degree_triangles(g):
    """Raw vertex counts, with triangles counted once at each incident root."""
    return Counter((g.rows[root].bit_count(), root_link(g, root).edges)
                   for root in range(g.n))


def joint_convolution(left, right):
    result = Counter()
    for (d, t), a in left.items():
        for (e, s), b in right.items():
            result[d + e, t + s] += a * b
    return result


def correlation_graphs():
    paw = graph(4, ((0, 1), (1, 2), (2, 0), (0, 3)))
    return disjoint_union(complete(3), star(3)), disjoint_union(paw, path(3))


def poly_add(left, right):
    result = dict(left)
    for term, coefficient in right.items():
        result[term] = result.get(term, Q(0)) + coefficient
    return {term: Q(coefficient) for term, coefficient in result.items() if coefficient}


def poly_scale(value, scalar):
    return {term: Q(scalar) * coefficient for term, coefficient in value.items()
            if scalar * coefficient}


def poly_multiply(left, right):
    result = {}
    for (a, b), c in left.items():
        for (d, e), f in right.items():
            term = a + d, b + e
            result[term] = result.get(term, Q(0)) + c * f
    return {term: Q(coefficient) for term, coefficient in result.items() if coefficient}


def poly_power(value, exponent):
    integer(exponent, "polynomial exponent")
    result = {(0, 0): Q(1)}
    for _ in range(exponent):
        result = poly_multiply(result, value)
    return result


def degree_specialization(value):
    """Substitute T=H**2 in a polynomial in normalized K2 and K3."""
    result = Counter()
    for (a, b), coefficient in value.items():
        result[a + 2 * b] += coefficient
    return {degree: coefficient for degree, coefficient in result.items() if coefficient}


def divide_degree_kernel(value):
    """Exact polynomial division by T-H**2, monic in T."""
    work, quotient = dict(value), {}
    while any(b for (_, b) in work):
        a, b = max((key for key in work if key[1]), key=lambda key: key[1])
        coefficient = work.pop((a, b))
        quotient[a, b - 1] = quotient.get((a, b - 1), Q(0)) + coefficient
        term = a + 2, b - 1
        work[term] = work.get(term, Q(0)) + coefficient
        if not work[term]:
            del work[term]
    return ({term: coefficient for term, coefficient in quotient.items() if coefficient},
            work)


def inverse_coefficients(t, order):
    """Truncate (1+t(R-S))**-1 by total degree in independent R,S."""
    integer(order, "inverse truncation order")
    t = Q(t)
    return {(a, j - a): Q(comb(j, a)) * (-t) ** a * t ** (j - a)
            for j in range(order + 1) for a in range(j + 1)
            if t or j == 0}


def cone(link):
    return graph(link.n + 1,
                 tuple((0, u + 1) for u in range(link.n))
                 + tuple((u + 1, v + 1) for u in range(link.n)
                         for v in link.neighbors(u) if u < v))


def verify_arithmetic_geometry(inverse_order=8):
    integer(inverse_order, "inverse truncation order")
    checks = 0

    def require(condition, description):
        nonlocal checks
        if not condition:
            raise AssertionError(description)
        checks += 1

    # Explicit finite rooted checks, independent of the joint-statistic runtime.
    fixtures = (complete(1), complete(2), complete(3), path(3), cycle(4), star(3),
                graph(4, ((0, 1), (1, 2), (2, 0), (0, 3))))
    link_checks = 0
    for g in fixtures:
        for h in fixtures:
            product = cartesian(g, h)
            for u in range(g.n):
                for v in range(h.n):
                    require(isomorphic(root_link(product, u * h.n + v),
                                       disjoint_union(root_link(g, u), root_link(h, v))),
                            "Cartesian root link is the disjoint union of factor links")
                    require(link_component_counts(product, u * h.n + v)
                            == link_component_counts(g, u) + link_component_counts(h, v),
                            "each connected link-component count is additive")
                    link_checks += 1
            require(joint_degree_triangles(product)
                    == joint_convolution(joint_degree_triangles(g), joint_degree_triangles(h)),
                    "joint degree/triangle law convolves under Cartesian product")

    # The normalized K2 and K3 coordinates are independent link variables.
    k1, k2 = IsoGraph(complete(1)), IsoGraph(complete(2))
    require(all(link_component_counts(complete(2), u) == {k1: 1} for u in range(2)),
            "H=K2/2 has link-coordinate z")
    require(all(link_component_counts(complete(3), u) == {k2: 1} for u in range(3)),
            "T=K3/3 has link-coordinate w")
    fan_checks = 0
    for a in range(4):
        for b in range(4):
            g = cone(disjoint_union(*(complete(1) for _ in range(a)),
                                    *(complete(2) for _ in range(b))))
            counts = link_component_counts(g, 0)
            require(counts.get(k1, 0) == a and counts.get(k2, 0) == b
                    and sum(counts.values()) == a + b,
                    "finite cone realizes the prescribed pair of link multiplicities")
            fan_checks += 1

    kernel = {(0, 1): Q(1), (2, 0): Q(-1)}
    kernel_examples = (
        kernel,
        {(0, 2): Q(1), (4, 0): Q(-1)},
        {(1, 1): Q(1), (3, 0): Q(-1)},
        {(1, 2): Q(2), (3, 1): Q(-1), (5, 0): Q(-1)},
        {(0, 0): Q(3), (1, 2): Q(2), (3, 1): Q(-1), (5, 0): Q(-1)},
    )
    divisions = []
    for polynomial in kernel_examples:
        quotient, remainder = divide_degree_kernel(polynomial)
        require(poly_add(poly_multiply(kernel, quotient), remainder) == polynomial,
                "degree-kernel division reconstructs the input exactly")
        require({a: c for (a, b), c in remainder.items()} == degree_specialization(polynomial),
                "degree-kernel remainder is exactly the degree specialization")
        divisions.append({"input": [[a, b, str(c)] for (a, b), c in sorted(polynomial.items())],
                          "quotient": [[a, b, str(c)] for (a, b), c in sorted(quotient.items())],
                          "remainder": [[a, b, str(c)] for (a, b), c in sorted(remainder.items())]})

    left, right = correlation_graphs()
    joint_left, joint_right = joint_degree_triangles(left), joint_degree_triangles(right)
    require(left.n == right.n == 7, "joint-correlation examples have the same order")
    for axis in (0, 1):
        def marginal(joint):
            result = Counter()
            for key, count in joint.items():
                result[key[axis]] += count
            return result
        require(marginal(joint_left) == marginal(joint_right), "separate marginals coincide")
    mixed = [sum(d * t * count for (d, t), count in joint.items())
             for joint in (joint_left, joint_right)]
    require(mixed == [6, 7], "mixed degree/triangle moments distinguish the joint laws")
    require(joint_left != joint_right, "equal marginals do not determine the joint law")

    # Reuse an independently established matrix identity to certify full spectra.
    spectral = verify_geometry(max_power=4)
    checks += spectral["exact_checks"]
    rook, shrikhande = rook_graph(), shrikhande_graph()
    link_k3, link_c6 = IsoGraph(complete(3)), IsoGraph(cycle(6))
    require(all(link_component_counts(rook, u) == {link_k3: 2} for u in range(16)),
            "normalized rook graph has root-link character z**2")
    require(all(link_component_counts(shrikhande, u) == {link_c6: 1} for u in range(16)),
            "normalized Shrikhande graph has root-link character w")
    witnesses = []
    for t, z, z_squared, w in ((Q(1, 2), "i", -1, 1),
                               (Q(-1, 2), "1", 1, -1)):
        x_value = z_squared - w
        require(1 + t * x_value == 0, "bounded link character annihilates endpoint family")
        witnesses.append({"t": str(t), "z": z, "w": str(w),
                          "X_character": str(x_value), "family_character": "0"})

    inverse_reports = []
    for t in (Q(1, 4), Q(-1, 4), Q(1, 3), Q(0)):
        family = {(0, 0): Q(1)}
        if t:
            family.update({(1, 0): t, (0, 1): -t})
        for order in range(inverse_order + 1):
            approximation = inverse_coefficients(t, order)
            product = poly_multiply(family, approximation)
            residual = poly_scale(poly_power({(1, 0): Q(1), (0, 1): Q(-1)}, order + 1),
                                  -(-t) ** (order + 1))
            require(product == poly_add({(0, 0): Q(1)}, residual),
                    "truncated Neumann inverse has precisely the expected residual")
            variation = sum(abs(coefficient) for coefficient in approximation.values())
            require(variation == sum((2 * abs(t)) ** j for j in range(order + 1)),
                    "inverse truncation has the exact binomial total variation")
        q = 2 * abs(t)
        tail = q ** (inverse_order + 1) / (1 - q)
        # This is the exact infinite coefficient-tail formula, algebraically
        # checked by its recurrence rather than approximated numerically.
        require(tail == q ** (inverse_order + 1) + q * tail,
                "geometric coefficient-tail formula satisfies its recurrence")
        inverse_reports.append({"t": str(t), "through_degree": inverse_order,
                                "retained_total_variation": str(variation),
                                "coefficient_tail_total_variation": str(tail)})

    # An arbitrary rooted law need not obey the mass-transport balance required
    # of graph-number arrays: concentrate only on the center-rooted P3 ball.
    center_only = star(2)
    forward = sum(center_only.rows[v].bit_count() == 1 for v in center_only.neighbors(0))
    reverse = sum(center_only.rows[v].bit_count() == 2 for v in center_only.neighbors(0))
    require((forward, reverse) == (2, 0), "center-only P3 law violates degree2-to-degree1 balance")

    return {
        "success": True, "arithmetic": "integer and exact rational",
        "exact_checks": checks,
        "root_link_product_vertex_pairs": link_checks,
        "joint_convolution_graph_pairs": len(fixtures) ** 2,
        "finite_link_cone_realizations": fan_checks,
        "degree_kernel": {"generator": "T-H^2, H=K2/2, T=K3/3", "divisions": divisions},
        "joint_correlation": {
            "graphs": ["K3 disjoint K1,3", "paw disjoint P3"],
            "degree_marginal": [[1, 3], [2, 3], [3, 1]],
            "root_triangle_marginal": [[0, 4], [1, 3]],
            "joint_counts": [[[d, t, c] for (d, t), c in sorted(joint.items())]
                             for joint in (joint_left, joint_right)],
            "raw_mixed_moments": mixed,
            "normalized_mixed_moments": [str(Q(value, 7)) for value in mixed]},
        "cospectral_family": {
            "definition": "1+tX, X=U(rook)-U(Shrikhande)",
            "full_spectrum_certificate": spectral["spectra_certified_by"],
            "common_laplacian_spectrum": spectral["common_laplacian_spectrum"],
            "root_link_character": "X=z^2-w; z counts K3 link components, w counts C6",
            "endpoint_nonunit_witnesses": witnesses,
            "inverse_truncations": inverse_reports},
        "balance_obstruction": {"rooted_law": "center-rooted P3 with mass one",
                                "degree2_to_degree1_mass": forward,
                                "degree1_to_degree2_mass": reverse},
        "scope": "Finite exact verification accompanies proofs; finite checks alone do not establish infinite convergence.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inverse-order", type=int, default=8)
    args = parser.parse_args()
    report = verify_arithmetic_geometry(args.inverse_order)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
