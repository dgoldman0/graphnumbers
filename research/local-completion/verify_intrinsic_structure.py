#!/usr/bin/env python3
"""Exact finite checks for INTRINSIC_GRAPH_STRUCTURE.md.

Python 3.10+, standard library only. Infinite limits and rigidity have
separate proofs; these checks use directly constructed finite graphs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction as Q
from pathlib import Path

from verify_local_algebra import (
    TYPES, ball, cart, cycle, distances, graph, isomorphic,
    linear_hist, path, star, union,
)
from verify_representation import weighted_norm


def edges(g):
    return [(u, v) for u in range(g.n) for v in g.neighbors(u) if u < v]


def add_leaf(g, root=0):
    return graph(g.n + 1, edges(g) + [(root, g.n)])


def join(g, h):
    disjoint = union(g, h)
    return graph(disjoint.n, edges(disjoint) + [(0, g.n)])


def on_four_cycle(g, u, v):
    return any(x != v and y != u and x != y and (g.rows[x] & (1 << y))
               for x in g.neighbors(u) for y in g.neighbors(v))


def degree_character(terms, t):
    return sum(c * sum(t ** row.bit_count() for row in g.rows)
               for c, g in terms)


def main():
    checks = Counter()

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] += 1

    fixtures = [graph(1), path(2), path(3), cycle(3), star(3)]

    # Directly test the graph-theoretic bridge obstruction. No Cartesian
    # factorization routine is used to label the pendant examples prime.
    for g in fixtures[1:]:
        for h in fixtures[1:]:
            product = cart(g, h)
            check("nontrivial_product_edges_on_squares",
                  all(on_four_cycle(product, u, v) for u, v in edges(product)))

    errors = []
    for g in fixtures:
        for n in (7, 11, 17):
            product = cart(g, cycle(n))
            pendant = add_leaf(product)
            check("pendant_graph_connected", len(distances(pendant, 0)) == pendant.n)
            without_leaf_edge = graph(pendant.n, edges(product))
            check("new_leaf_edge_is_bridge", without_leaf_edge.component_count() == 2)
            check("leaf_edge_not_on_square", not on_four_cycle(pendant, 0, product.n))

            # The direct radius-one formula is much sharper than the
            # general degree-bound estimate, and detects normalization errors.
            if g == path(2):
                sample = {"cycle_length": n, "errors": {}}
                for k in (1, 2, 3):
                    difference = linear_hist([(Q(1, pendant.n), pendant),
                                              (-Q(1, product.n), product)], 1)
                    observed = weighted_norm(difference, k)
                    expected = Q(2**k + 5**k + 2*4**k, 2*n+1)
                    check("exact_sign_change_input_error", observed == expected)
                    sample["errors"][str(k)] = str(observed)
                # The algebraic sign change fixes these bridge primes and
                # the cycles, but negates the K2 factor of the product.
                before_vertex = Q(pendant.n, pendant.n)-Q(product.n, product.n)
                after_vertex = Q(pendant.n, pendant.n)+Q(product.n, product.n)
                check("sign_change_separated_by_vertex_character",
                      before_vertex == 0 and after_vertex == 2)
                sample["image_vertex_value"] = str(after_vertex)
                errors.append(sample)

            delta = max(row.bit_count() for row in pendant.rows)
            for radius in (1, 2):
                b = sum(delta**j for j in range(radius+1))
                difference = linear_hist([(Q(1, pendant.n), pendant),
                                          (-Q(1, product.n), product)], radius)
                # Independently compare balls outside the claimed affected set.
                ds = distances(product, 0)
                unchanged = [u for u in range(product.n) if ds[u] > radius]
                check("leaf_change_locality",
                      all(isomorphic(ball(product, u, radius)[0],
                                     ball(pendant, u, radius)[0], roots=(0, 0))
                          for u in unchanged))
                for k in (1, 2):
                    bound = Q(2*(b+1)*b**k, pendant.n)
                    check("weighted_leaf_change_bound",
                          weighted_norm(difference, k) <= bound)

    # Equal-size bridge-joined components implement the diagonal-rigidity
    # mixture. Test the weighted estimate and independent edge observables.
    for g in fixtures[1:]:
        for n in (7, 11):
            product = cart(g, cycle(n))
            ring = cycle(g.n*n)
            joined = join(product, ring)
            check("mixture_graph_connected", joined.component_count() == 1)
            check("mixture_equal_part_sizes", product.n == ring.n)
            check("mixture_edge_observable",
                  Q(joined.edges, joined.n)
                  == 1+Q(g.edges, 2*g.n)+Q(1, joined.n))
            check("product_and_line_observables_independent",
                  Q(product.edges, product.n)-Q(ring.edges, ring.n)
                  == Q(g.edges, g.n) > 0)
            delta = max(row.bit_count() for row in joined.rows)
            for radius in (1, 2):
                b = sum(delta**j for j in range(radius+1))
                difference = linear_hist([(Q(1, joined.n), joined),
                                          (-Q(1, 2*product.n), product),
                                          (-Q(1, 2*ring.n), ring)], radius)
                for k in (1, 2):
                    check("weighted_bridge_mixture_bound",
                          weighted_norm(difference, k) <= Q(4*b**(k+1), joined.n))

    for radius in range(1, 5):
        short, long = cycle(2*radius+2), cycle(2*radius+3)
        terms = [(Q(1, short.n), short), (-Q(1, long.n), long)]
        check("strict_filtration_zero_at_radius", not linear_hist(terms, radius))
        check("strict_filtration_visible_next_radius", bool(linear_hist(terms, radius+1)))

    # A positive graph has negative value under a continuous real character.
    edge = path(2)
    check("positive_edge_outside_closed_sums_of_squares",
          degree_character([(1, edge)], Q(-1)) == -2)
    square_terms = [(1, graph(1)), (-2, edge), (1, cart(edge, edge))]
    square_hist = linear_hist(square_terms, 1)
    edge_type = TYPES.register(edge)
    check("square_outside_local_positive_cone", square_hist[edge_type] == -4)
    for t in map(Q, ["-1", "-1/2", "0", "1/2", "1"]):
        check("square_degree_character_identity",
              degree_character(square_terms, t) == (1-2*t)**2)

    # Sender/receiver transport is enumerated on P3 independently of the
    # compact formula 2t-2t^2, including both signs and the exceptional t=0,1.
    p3 = path(3)
    for t in map(Q, ["-2", "-1", "-1/2", "0", "1/2", "1", "2"]):
        outgoing = incoming = Q(0)
        for u in range(p3.n):
            for v in p3.neighbors(u):
                if p3.rows[u].bit_count() == 1 and p3.rows[v].bit_count() == 2:
                    outgoing += t**p3.rows[u].bit_count()
                    incoming += t**p3.rows[v].bit_count()
        check("degree_twist_transport_formula", outgoing-incoming == 2*t-2*t*t)
        check("degree_twist_balance_obstruction", (outgoing == incoming) == (t in (0, 1)))

    result = {
        "status": "passed",
        "total_checks": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "sign_change_witnesses": errors,
        "scope": "Exact finite examples and estimates; universal rigidity is proved in INTRINSIC_GRAPH_STRUCTURE.md.",
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
