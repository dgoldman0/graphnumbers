#!/usr/bin/env python3
"""Exact finite checks for REPRESENTATION_THEOREM.md; Python 3.10+.

Uses the existing graph/ball/isomorphism primitives, with independent
injective-embedding enumeration for the inclusion-exclusion expansion.
These checks do not replace the universal proofs.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction as Q
from functools import lru_cache
from itertools import combinations
from pathlib import Path

from verify_local_algebra import (
    TYPES, ball, clean, cycle, distances, graph, hist, induced, isomorphic,
    linear_hist, path, star,
)


def edges(g):
    return [(u, v) for u in range(g.n) for v in range(u + 1, g.n)
            if g.rows[u] & (1 << v)]


def degree(g, u):
    return g.rows[u].bit_count()


def max_degree(g):
    return max((degree(g, u) for u in range(g.n)), default=0)


@lru_cache(None)
def inj_root(source, target):
    """Injective edge-preserving maps fixing vertex zero, by backtracking."""
    if source.n > target.n:
        return 0
    if degree(source, 0) > degree(target, 0):
        return 0
    order = sorted(range(1, source.n), key=lambda u: -degree(source, u))
    mapping = {0: 0}
    used = {0}

    def visit(i):
        if i == len(order):
            return 1
        u = order[i]
        total = 0
        for v in range(target.n):
            if v in used or degree(target, v) < degree(source, u):
                continue
            if any(source.rows[u] & (1 << x)
                   and not target.rows[v] & (1 << y)
                   for x, y in mapping.items()):
                continue
            mapping[u] = v
            used.add(v)
            total += visit(i + 1)
            used.remove(v)
            del mapping[u]
        return total

    return visit(0)


def rooted(g, u):
    return induced(g, (u,) + tuple(v for v in range(g.n) if v != u))


def inj(source, target):
    return sum(inj_root(source, rooted(target, u)) for u in range(target.n))


def subsets(items):
    for mask in range(1 << len(items)):
        yield [item for j, item in enumerate(items) if mask & (1 << j)]


@lru_cache(None)
def indicator_expansion(h, radius, bound):
    """Compile equation (1), discarding terms of degree above the bound."""
    interior = [u for u, dist in distances(h, 0).items() if dist < radius]
    missing = [(u, v) for u, v in combinations(range(h.n), 2)
               if not h.rows[u] & (1 << v)]
    automorphisms = inj_root(h, h)
    result = defaultdict(Q)
    for added in subsets(missing):
        base_edges = edges(h) + added
        base = graph(h.n, base_edges)
        if max_degree(base) > bound:
            continue
        capacity = {u: bound - degree(base, u) for u in interior}
        # Every new vertex consumes at least one remaining incidence.
        for t in range(sum(capacity.values()) + 1):
            new_edges = []

            def extend(j, selected):
                if j == t:
                    f = graph(h.n + t, base_edges + new_edges)
                    result[f] += Q((-1) ** (len(added) + selected),
                                   automorphisms * math.factorial(t))
                    return
                available = [u for u in interior if capacity[u]]
                for attach in subsets(available):
                    if not attach or len(attach) > bound:
                        continue
                    for u in attach:
                        capacity[u] -= 1
                        new_edges.append((u, h.n + j))
                    extend(j + 1, selected + len(attach))
                    for u in reversed(attach):
                        new_edges.pop()
                        capacity[u] += 1

            extend(0, 0)
    return clean(result)


def cut_degree(g, bound):
    return graph(g.n, ((u, v) for u, v in edges(g)
                      if degree(g, u) <= bound and degree(g, v) <= bound))


def leaf_transport_divergence(g, u):
    outgoing = sum(degree(g, u) == 1 and degree(g, v) == 2
                   for v in g.neighbors(u))
    incoming = sum(degree(g, v) == 1 and degree(g, u) == 2
                   for v in g.neighbors(u))
    return outgoing - incoming


def weighted_norm(h, power):
    return sum((TYPES.reps[t].n ** power * abs(value)
                for t, value in h.items()), Q(0))


def subtract(a, b):
    result = defaultdict(Q, a)
    for key, value in b.items():
        result[key] -= value
    return clean(result)


def normalized_cycle_hist(n, r):
    # Vertex transitivity makes the normalized counting histogram a point mass.
    return {TYPES.register(ball(cycle(n), 0, r)[0]): Q(1)}


def main():
    checks = Counter()

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] += 1

    labelled = []
    for n in range(1, 5):
        choices = list(combinations(range(n), 2))
        labelled.extend(graph(n, chosen) for chosen in subsets(choices))
    extras = [path(n) for n in range(5, 9)]
    extras += [cycle(n) for n in range(5, 9)]
    extras += [star(4), star(5)]
    fixtures = labelled + extras

    # Equation (1) is tested against direct rooted-ball isomorphism, including
    # complete small components where excluding external edges matters.
    expansion_summary = []
    for bound, radius in [(1, 0), (1, 1), (1, 2),
                          (2, 1), (2, 2), (2, 3), (3, 1)]:
        targets = [g for g in fixtures if max_degree(g) <= bound]
        types = sorted({TYPES.register(ball(g, u, radius)[0])
                        for g in targets for u in range(g.n)})
        term_count = 0
        for type_id in types:
            h = TYPES.reps[type_id]
            expansion = indicator_expansion(h, radius, bound)
            term_count += len(expansion)
            size_bound = (bound + 1) * sum(bound ** j for j in range(radius + 1))
            for pattern in expansion:
                check("expansion_pattern_bounds",
                      max_degree(pattern) <= bound and pattern.n <= size_bound
                      and len(distances(pattern, 0)) == pattern.n)
            for g in targets:
                for u in range(g.n):
                    target = rooted(g, u)
                    actual = Q(isomorphic(ball(g, u, radius)[0], h, (0, 0)))
                    expanded = sum((coefficient * inj_root(pattern, target)
                                    for pattern, coefficient in expansion.items()), Q(0))
                    check("rooted_indicator_expansions", expanded == actual)
        expansion_summary.append({"degree_bound": bound, "radius": radius,
                                  "ball_types": len(types),
                                  "nonzero_labelled_pattern_terms": term_count})

    # Independently exercise the finite triangular injective-count matrix.
    representatives = []
    for g in labelled:
        if len(distances(g, 0)) != g.n:
            continue
        if not any(isomorphic(g, h) for h in representatives):
            representatives.append(g)
    representatives.sort(key=lambda g: (g.n, g.edges))
    for i, target in enumerate(representatives):
        for j, pattern in enumerate(representatives):
            count = inj(pattern, target)
            check("injective_matrix_triangularity",
                  (j <= i or count == 0) and (j != i or count > 0))

    # Whole-graph degree filtering must be determined by one extra layer.
    for g in fixtures:
        for bound in range(1, 4):
            cut = cut_degree(g, bound)
            for radius in range(3):
                for u in range(g.n):
                    parent_ball = ball(g, u, radius + 1)[0]
                    from_local = ball(cut_degree(parent_ball, bound), 0, radius)[0]
                    direct = ball(cut, u, radius)[0]
                    check("degree_cutoff_locality", isomorphic(from_local, direct, (0, 0)))

    # Check the tail bound against signed histograms after cancellation.
    terms = [(Q((-1) ** j, j % 4 + 1), g) for j, g in enumerate(fixtures)]
    for bound in range(1, 6):
        transformed = [(coefficient, cut_degree(g, bound)) for coefficient, g in terms]
        for radius in range(3):
            parent = linear_hist(terms, radius + 1)
            push = defaultdict(Q)
            for type_id, coefficient in parent.items():
                local = ball(cut_degree(TYPES.reps[type_id], bound), 0, radius)[0]
                push[TYPES.register(local)] += coefficient
            actual = linear_hist(transformed, radius)
            check("signed_cutoff_pushforward", clean(push) == actual)
            difference = subtract(actual, linear_hist(terms, radius))
            for power in range(1, 4):
                tail = sum((abs(value) * TYPES.reps[t].n ** power
                            for t, value in parent.items()
                            if TYPES.reps[t].n > bound), Q(0))
                check("signed_weighted_cutoff_tail_bound",
                      weighted_norm(difference, power) <= 2 * tail)

    for g in fixtures:
        divergences = [leaf_transport_divergence(g, u) for u in range(g.n)]
        check("finite_graph_transport_balance", sum(divergences) == 0)
        for u, value in enumerate(divergences):
            check("transport_locality",
                  value == leaf_transport_divergence(ball(g, u, 2)[0], 0))
    p3 = path(3)
    check("nonmember_endpoint_path", leaf_transport_divergence(p3, 0) == 1)
    check("uniform_root_path_balance",
          sum(Q(leaf_transport_divergence(p3, u), 3) for u in range(3)) == 0)

    # Cycle formulas use actual induced balls, with the cancellation predicted
    # from the inequalities in the proof.
    cycle_sizes = [3 ** j for j in range(1, 5)]
    for n in cycle_sizes:
        g = cycle(n)
        for radius in [0, 1, 2, 3]:
            b = ball(g, 0, radius)[0]
            check("cycle_root_transitivity",
                  isomorphic(b, ball(g, n // 2, radius)[0], (0, 0)))
        for radius in range(8):
            if n > 2 * radius + 1:
                check("cycle_pair_local_cancellation",
                      normalized_cycle_hist(2 * n, radius)
                      == normalized_cycle_hist(n, radius))
    for coefficients in [(Q(1), Q(1), Q(1)), (Q(-7, 3), Q(0), Q(5))]:
        for m in range(1, 4):
            radius = cycle_sizes[m - 1]
            h = defaultdict(Q)
            for j, coefficient in enumerate(coefficients):
                n = cycle_sizes[j]
                for sign, size in [(1, 2 * n), (-1, n)]:
                    for key, value in normalized_cycle_hist(size, radius).items():
                        h[key] += sign * coefficient * value
            h = clean(h)
            check("cycle_total_variation_growth",
                  weighted_norm(h, 0) == 2 * sum(abs(c) for c in coefficients[:m]))
            for power in [1, 2, 3]:
                expected = sum(abs(coefficients[j])
                               * ((2 * cycle_sizes[j]) ** power + cycle_sizes[j] ** power)
                               for j in range(m))
                check("cycle_weighted_norms", weighted_norm(h, power) == expected)
            key = next(iter(normalized_cycle_hist(2 * radius, radius)))
            check("cycle_coefficient_recovery", h.get(key, Q(0)) == coefficients[m - 1])

    result = {
        "status": "passed",
        "arithmetic": "exact integers and fractions",
        "scope": "finite checks of the separate universal proofs",
        "labelled_nonempty_graphs_up_to_four_vertices": len(labelled),
        "additional_graph_fixtures": len(extras),
        "connected_unlabelled_triangular_matrix_size": len(representatives),
        "indicator_expansion_cases": expansion_summary,
        "checks": dict(sorted(checks.items())),
        "total_checks": sum(checks.values()),
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("representation_results.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
