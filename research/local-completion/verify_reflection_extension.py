#!/usr/bin/env python3
"""Exact finite checks for REFLECTION_EXTENSION.md, Python 3.10+.

The universal extension and continuity assertions have separate proofs.
No finite search here decides whether a full automorphism exists.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction as Q
from functools import lru_cache
from pathlib import Path

from reconstruct_local import catalog
from verify_intrinsic_structure import add_leaf, edges, join
from verify_local_algebra import (
    TYPES, ball, cart, complete, cycle, graph, induced, isomorphic, linear_hist,
    path, star, union,
)
from verify_representation import weighted_norm


def parity_cut(g):
    return graph(g.n, [(u, v) for u, v in edges(g)
                       if g.rows[u].bit_count() % 2 == g.rows[v].bit_count() % 2])


@lru_cache(None)
def reflect_graph(g):
    cut = parity_cut(g)
    unseen = set(range(g.n))
    output = []
    while unseen:
        first = min(unseen)
        component, pending = {first}, [first]
        while pending:
            u = pending.pop()
            for v in cut.neighbors(u):
                if v not in component:
                    component.add(v)
                    pending.append(v)
        unseen -= component
        output.append((Q((-1)**g.rows[first].bit_count()),
                       induced(cut, sorted(component))))
    return tuple(output)


def reflect(terms):
    return [(c*d, h) for c, g in terms for d, h in reflect_graph(g)]


@lru_cache(None)
def positive_preimage(g):
    odd = [u for u, row in enumerate(g.rows) if row.bit_count() % 2]
    decorated = graph(g.n+len(odd), edges(g)+[(u, g.n+i) for i, u in enumerate(odd)])
    return union(decorated, graph(len(odd)))


@lru_cache(None)
def negative_preimage(g):
    even = [u for u, row in enumerate(g.rows) if row.bit_count() % 2 == 0]
    new_edges = edges(g)
    for i, u in enumerate(even):
        middle, last = g.n+2*i, g.n+2*i+1
        new_edges.extend([(u, middle), (middle, last)])
    return graph(g.n+2*len(even), new_edges)


def apply_graph_map(terms, operation):
    return [(c, operation(g)) for c, g in terms]


def multiply(left, right):
    return [(c*d, cart(g, h)) for c, g in left for d, h in right]


def scale(value, terms):
    return [(value*c, g) for c, g in terms]


REPS = []


@lru_cache(None)
def graph_type(g):
    for i, other in enumerate(REPS):
        if isomorphic(g, other):
            return i
    REPS.append(g)
    return len(REPS)-1


def coefficients(terms):
    result = Counter()
    for c, g in terms:
        # Inputs to this checker are connected graph terms. The graph
        # preimage constructions can be disconnected, so decompose first.
        unseen = set(range(g.n))
        while unseen:
            first = min(unseen)
            seen, pending = {first}, [first]
            while pending:
                u = pending.pop()
                for v in g.neighbors(u):
                    if v not in seen:
                        seen.add(v)
                        pending.append(v)
            unseen -= seen
            result[graph_type(induced(g, sorted(seen)))] += c
    return {t: c for t, c in result.items() if c}


def equal(left, right):
    return coefficients(left) == coefficients(right)


def degree_polynomial(terms, triangle_sign=False):
    result = Counter()
    for c, g in terms:
        for u in range(g.n):
            neighbors = g.neighbors(u)
            triangles = sum(bool(g.rows[v] & (1 << w))
                            for i, v in enumerate(neighbors) for w in neighbors[i+1:])
            result[len(neighbors)] += c*((-1)**triangles if triangle_sign else 1)
    return {n: c for n, c in result.items() if c}


def poly_product(a, b):
    result = Counter()
    for n, c in a.items():
        for m, d in b.items():
            result[n+m] += c*d
    return {n: c for n, c in result.items() if c}


@lru_cache(None)
def cube(n):
    result = graph(1)
    for _ in range(n):
        result = cart(result, path(2))
    return result


def section(polynomial):
    return [(c/Q(2**n), cube(n)) for n, c in polynomial.items()]


def linear_involution(terms):
    polynomial = degree_polynomial(terms)
    correction = {n: ((-1)**n-1)*c for n, c in polynomial.items() if n % 2}
    return terms + section(correction)


def main():
    checks = Counter()

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] += 1

    fixtures = catalog(4)
    unit, edge = graph(1), path(2)
    for g in fixtures:
        r, o = positive_preimage(g), negative_preimage(g)
        check("positive_right_inverse", equal(reflect([(1, r)]), [(1, g)]))
        check("positive_preimage_of_negative", equal(reflect([(1, o)]), [(-1, g)]))
        check("positive_preimage_size", r.n <= 3*g.n)
        check("negative_preimage_size", o.n <= 3*g.n)
        if len({row.bit_count() % 2 for row in g.rows}) == 1:
            sign = (-1)**g.rows[0].bit_count()
            check("uniform_parity_reflection", equal(reflect([(1, g)]), [(sign, g)]))
            check("uniform_parity_involution", equal(reflect(reflect([(1, g)])), [(1, g)]))
        check("linear_involution_square", equal(linear_involution(linear_involution([(1, g)])), [(1, g)]))
        for h in fixtures:
            product = cart(g, h)
            check("parity_cut_cartesian_identity", parity_cut(product) == cart(parity_cut(g), parity_cut(h)))
            check("reflection_multiplicativity",
                  equal(reflect([(1, product)]), multiply(reflect([(1, g)]), reflect([(1, h)]))))
            check("triangle_retraction_multiplicativity",
                  degree_polynomial([(1, product)], True)
                  == poly_product(degree_polynomial([(1, g)], True), degree_polynomial([(1, h)], True)))

    check("edge_reflection", equal(reflect([(Q(1, 2), edge)]), [(-Q(1, 2), edge)]))
    kernel_element = [(1, path(3)), (1, unit)]
    check("positive_nonzero_kernel", bool(coefficients(kernel_element)) and not coefficients(reflect(kernel_element)))
    check("reflection_not_involutive_on_all_graphs",
          not equal(reflect(reflect([(1, path(3))])), [(1, path(3))]))

    # Signed inputs include exact local cancellations, not just positive
    # graph inputs where a vertex-count bound would suffice.
    signed_inputs = [
        [(Q(2, 3), path(3)), (-Q(3, 5), star(3)), (Q(7, 11), complete(3))],
        [(1, path(3)), (-1, edge), (-Q(1, 4), cycle(4))],
        [(Q(1, 7), cycle(7)), (-Q(1, 11), cycle(11))],
        [(1, path(4)), (-1, path(3)), (-1, unit)],
    ]
    for terms in signed_inputs:
        for radius in (1, 2, 3):
            for k in (1, 2):
                source = weighted_norm(linear_hist(terms, radius), k)
                larger = weighted_norm(linear_hist(terms, radius+1), k)
                check("signed_reflection_bound",
                      weighted_norm(linear_hist(reflect(terms), radius), k) <= larger)
                check("signed_positive_section_bound",
                      weighted_norm(linear_hist(apply_graph_map(terms, positive_preimage), radius), k)
                      <= (2**(k+1)+1)*source)
                check("signed_negative_section_bound",
                      weighted_norm(linear_hist(apply_graph_map(terms, negative_preimage), radius), k)
                      <= 3**(k+1)*source)

    # The natural linear involution has a nonzero, explicitly predicted
    # multiplicativity defect, witnessed independently as graph coefficients.
    h = [(Q(1, 2), edge)]
    j = [(1, path(3)), (-1, edge), (-Q(1, 4), cycle(4))]
    check("degree_kernel_witness", degree_polynomial(j) == {} and bool(linear_hist(j, 2)))
    hj = multiply(h, j)
    defect = linear_involution(hj) + scale(-1, multiply(linear_involution(h), linear_involution(j)))
    check("linear_involution_product_defect", equal(defect, scale(2, hj)) and bool(coefficients(defect)))
    check("linear_involution_path_image", equal(linear_involution([(1, path(3))]), [(1, path(3)), (-2, edge)]))

    # Two distinct retractions agree on the complete edge subalgebra.
    for n in range(5):
        check("triangle_retraction_section", degree_polynomial(section({n: Q(1)}), True) == {n: Q(1)})
    alternative_kernel = [(1, complete(3)), (-Q(3, 4), cycle(4))]
    check("distinct_retraction_kernels", degree_polynomial(alternative_kernel) == {}
          and degree_polynomial(alternative_kernel, True) == {2: Q(-6)})

    witnesses = []
    for n in (5, 7, 11):
        prism = cart(edge, cycle(n))
        joined = join(prism, cycle(2*n))
        punctured = induced(prism, list(range(1, prism.n)))
        expected = [(1, path(2*n-1)), (-1, punctured)]
        actual = reflect([(1, joined)])
        check("joined_graph_image_formula", equal(actual, expected))
        check("joined_graph_image_zero_mass", sum(c*g.n for c, g in actual) == 0)
        marginal = linear_hist(scale(Q(1, joined.n), actual), 1)
        expected_coefficients = {
            TYPES.register(star(1)): Q(1, 2*n),
            TYPES.register(star(2)): Q(1, 2)-Q(3, 2*n),
            TYPES.register(star(3)): -Q(1, 2)+Q(1, n),
        }
        check("joined_graph_exact_signed_marginal", marginal == expected_coefficients)
        check("joined_graph_mixed_signs", min(marginal.values()) < 0 < max(marginal.values()))
        input_error = [(Q(1, 4*n), joined), (-Q(1, 4*n), prism), (-Q(1, 4*n), cycle(2*n))]
        for radius in (1, 2):
            for k in (1, 2):
                check("joined_graph_weighted_limit_bound",
                      weighted_norm(linear_hist(reflect(input_error), radius), k)
                      <= weighted_norm(linear_hist(input_error, radius+1), k))
        witnesses.append({"cycle_length": n, "vertices": joined.n,
                          "radius_one_star_coefficients": {
                              str(degree): str(expected_coefficients[TYPES.register(star(degree))])
                              for degree in (1, 2, 3)}})

    discontinuity_witnesses = []
    cube_ball = TYPES.register(ball(cube(3), 0, 2)[0])
    for n in (7, 11, 17):
        prism = cart(edge, cycle(n))
        pendant = add_leaf(prism)
        # Formula (14) on the bridge prime, combined with theta(H)=-H
        # and theta(C_n)=C_n. Compute the prime degree polynomial afresh.
        polynomial = degree_polynomial([(Q(1, pendant.n), pendant)])
        correction = {d: ((-1)**d-1)*c for d, c in polynomial.items() if d % 2}
        image = [(Q(1, pendant.n), pendant), (Q(1, prism.n), prism)] + section(correction)
        marginal = linear_hist(image, 2)
        expected = -Q(2*(2*n-1), 2*n+1)
        check("algebraic_involution_discontinuity_coordinate", marginal[cube_ball] == expected)
        check("algebraic_involution_image_stays_separated", abs(marginal[cube_ball]) > 1)
        original = [(Q(1, pendant.n), pendant), (-Q(1, prism.n), prism)]
        discontinuity_witnesses.append({
            "cycle_length": n,
            "input_radius_two_weight_one": str(weighted_norm(linear_hist(original, 2), 1)),
            "image_hypercube_ball_coefficient": str(marginal[cube_ball]),
            "image_radius_two_weight_one": str(weighted_norm(marginal, 1)),
        })

    result = {
        "status": "passed",
        "total_checks": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "connected_catalog_size": len(fixtures),
        "mixed_sign_witnesses": witnesses,
        "algebraic_involution_discontinuity": discontinuity_witnesses,
        "scope": "Exact finite examples. A full continuous automorphism reflecting H remains unconstructed and unexcluded.",
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(result, indent=2)+"\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
