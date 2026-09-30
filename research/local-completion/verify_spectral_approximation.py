#!/usr/bin/env python3
"""Exact examples and certificates for the spectral/approximation notes.

Python 3.10+, standard library only. Infinite Fourier limits, spectral
invariance, and general approximation statements have separate proofs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from fractions import Fraction as Q
from functools import lru_cache
from itertools import product
from pathlib import Path

from reconstruct_local import (
    BudgetExceeded, OutOfSpan, catalog, encode_result, reconstruct,
)
from verify_local_algebra import (
    TYPES, ball, cart, clean, complete, convolution, cycle, graph, hist,
    linear_hist, path, star,
)
from verify_multiplication import hom_cumulant, multi
from verify_representation import cut_degree, subtract, weighted_norm


def gaussian_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def gaussian_mul(a, b):
    return (a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0])


def phase(exponent):
    return ((Q(1), Q(0)), (Q(0), Q(1)),
            (Q(-1), Q(0)), (Q(0), Q(-1)))[exponent % 4]


def main():
    checks = Counter()

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] += 1

    # A finite exact Fourier inversion over fourth roots of unity. Degree
    # alone collides here; additional rooted cumulants separate the fixtures.
    fixtures = [graph(1), path(2), path(3), star(2), complete(3)]
    types = [TYPES.register(g) for g in fixtures]
    patterns = [multi(path(2)), multi(path(3)), multi(complete(3))]

    @lru_cache(None)
    def code(type_id):
        values = [hom_cumulant(f, TYPES.reps[type_id]) for f in patterns]
        if any(v.denominator != 1 for v in values):
            raise ArithmeticError("Fixture cumulant is nonintegral")
        return tuple(int(v) % 4 for v in values)

    check("fourier_fixture_separation", len({code(t) for t in types}) == len(types))
    left = dict(zip(types, map(Q, ["1/3", "-2", "4/7", "5", "-3/2"])))
    right = dict(zip(types, map(Q, ["-1", "2/5", "3", "-1/7", "2"])))
    conv = convolution(left, right, 2)

    def character(array, parameter):
        result = (Q(0), Q(0))
        for t, value in array.items():
            z = phase(sum(u*v for u, v in zip(parameter, code(t))))
            result = gaussian_add(result, (value*z[0], value*z[1]))
        return result

    parameters = list(product(range(4), repeat=3))
    transforms = {t: character(left, t) for t in parameters}
    for t in parameters:
        check("exact_phase_character_products",
              character(conv, t) == gaussian_mul(character(left, t), character(right, t)))
        # Phase characters separate elements but miss this nonunit's zero
        # at the interior degree character z=1/2.
        edge_phase = phase(sum(u*v for u, v in zip(t, code(types[1]))))
        value = (1 - 2*edge_phase[0], -2*edge_phase[1])
        check("phase_family_misses_interior_zero", value != (0, 0))
    check("interior_character_detects_nonunit", 1 - 2*Q(1, 2) == 0)
    for type_id in types:
        value = (Q(0), Q(0))
        for t in parameters:
            inverse_phase = phase(-sum(u*v for u, v in zip(t, code(type_id))))
            value = gaussian_add(value, gaussian_mul(transforms[t], inverse_phase))
        value = tuple(v / len(parameters) for v in value)
        check("exact_fourier_coefficient_recovery", value == (left[type_id], Q(0)))

    # Compare direct Cartesian-power balls with the polynomial growth bound.
    for g in fixtures:
        power = graph(1)
        delta = max(row.bit_count() for row in g.rows)
        for n in range(1, 5):
            power = cart(power, g)
            for r in range(1, 4):
                observed = ball(power, 0, r)[0].n
                bound = sum((n*delta)**j for j in range(r+1))
                check("cartesian_power_ball_growth", observed <= bound)

    # Signed histograms test the new moment-controlled degree-filter rate.
    terms = [(Q(2, 3), star(4)), (Q(-3, 5), path(6)),
             (Q(1, 7), complete(4)), (Q(-2, 9), cycle(7))]
    for d, r, k, q in product(range(1, 5), range(3), (1, 2), (1, 2)):
        filtered = [(c, cut_degree(g, d)) for c, g in terms]
        error = weighted_norm(subtract(linear_hist(filtered, r), linear_hist(terms, r)), k)
        upper = 2*Q(1, d**q)*weighted_norm(linear_hist(terms, r+1), k+q)
        check("moment_cutoff_rate", error <= upper)

    # Exact full-catalog primal/dual certificates; check the certificates
    # afresh using graph histograms, without the solver's matrix operations.
    examples = []
    settings = [(1, 3, None, 5), (1, 4, 2, 1),
                (2, 5, None, 9), (2, 6, 2, 1)]
    for radius, limit, degree_limit, expected_cost in settings:
        graphs = catalog(limit, degree_limit)
        target = {TYPES.register(ball(path(2*radius+1), radius, radius)[0]): Q(1)}
        result = reconstruct(target, graphs, radius)
        recovered = linear_hist(list(zip(result["coefficients"], graphs)), radius)
        check("certified_reconstruction", recovered == target)
        primal_cost = sum(g.n*abs(c) for g, c in zip(graphs, result["coefficients"]))
        dual_value = sum(result["dual"].get(t, 0)*v for t, v in target.items())
        check("certified_optimal_cost", primal_cost == dual_value == expected_cost)
        for g in graphs:
            observable = sum(result["dual"].get(t, 0)*v for t, v in hist(g, radius).items())
            check("dual_constraints", abs(observable) <= g.n)
        check("negative_mass_identity",
              result["negative_mass"] == (primal_cost-1)/2)
        examples.append({"max_vertices": limit, "max_degree": degree_limit,
                         **encode_result(result, graphs, target, radius)})

    # Universal line formulas over a wider range than the enumerated catalogs.
    for r in range(1, 6):
        target = {TYPES.register(ball(path(2*r+1), r, r)[0]): Q(1)}
        check("path_difference_line_identity",
              linear_hist([(1, path(2*r+1)), (-1, path(2*r))], r) == target)
        check("positive_cycle_line_identity",
              linear_hist([(Q(1, 2*r+2), cycle(2*r+2))], r) == target)
        check("sharp_path_coefficient_mass", (2*r+1) + 2*r == 4*r+1)

    # An unbalanced target must produce a checked catalog-span separator.
    graphs = catalog(5, 2)
    outside = {TYPES.register(path(3)): Q(1)}
    try:
        reconstruct(outside, graphs, 2)
    except OutOfSpan as error:
        for g in graphs:
            check("span_separator_annihilates_catalog",
                  sum(error.witness.get(t, 0)*v for t, v in hist(g, 2).items()) == 0)
        check("span_separator_detects_target", error.value != 0 and
              error.value == sum(error.witness.get(t, 0)*v for t, v in outside.items()))
    else:
        raise AssertionError("Unbalanced target unexpectedly reconstructed")
    for job in (lambda: catalog(4, label_budget=1),
                lambda: reconstruct({TYPES.register(graph(1)): Q(1)}, graphs, 2,
                                    search_budget=1)):
        try:
            job()
        except BudgetExceeded:
            check("budget_exhaustion_is_explicit", True)
        else:
            raise AssertionError("Expected explicit budget exhaustion")

    # Quantitative lower bounds for a finite prefix of the non-measure family.
    sequence_terms = []
    coefficients = [Q(1), Q(-2), Q(3, 2)]
    for j, c in enumerate(coefficients, 1):
        n = 3**j
        sequence_terms.extend([(c/(2*n), cycle(2*n)), (-c/n, cycle(n))])
        variation = weighted_norm(linear_hist(sequence_terms, n), 0)
        cost = sum(abs(a)*g.n for a, g in sequence_terms)
        check("cycle_prefix_variation_lower_bound",
              variation == cost == 2*sum(abs(x) for x in coefficients[:j]))

    output = {"all_passed": True, "total_checks": sum(checks.values()),
              "checks": dict(sorted(checks.items())), "reconstruction_examples": examples,
              "scope": "Exact finite regressions and catalog certificates; universal proofs are separate."}
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("spectral_approximation_results.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("all_passed", "total_checks", "checks")}, indent=2))


if __name__ == "__main__":
    main()
