#!/usr/bin/env python3
"""Exact finite checks for MULTIPLICATION_AND_UNITS.md; Python 3.10+.

The coalgebra is checked symbolically, independently of homomorphism counts.
Counts, cumulants, convolutions, and reciprocal coefficients are exact.
Universal domain and summability assertions require the separate proofs.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from itertools import combinations, permutations, product
from pathlib import Path

from verify_local_algebra import (
    TYPES, ball, cart, clean, complete, convolution, cycle, distances, graph,
    linear_hist, path, truncated_type_product,
)


@dataclass(frozen=True)
class MultiGraph:
    n: int
    edges: tuple[tuple[int, int], ...]


def multi(g):
    return MultiGraph(g.n, tuple((u, v) for u in range(g.n)
                                for v in range(u + 1, g.n)
                                if g.rows[u] & (1 << v)))


@lru_cache(None)
def canonical(f):
    """Brute-force rooted canonical form, preserving parallel occurrences."""
    candidates = []
    for tail in permutations(range(1, f.n)):
        mapping = (0,) + tail
        edges = tuple(sorted(tuple(sorted((mapping[u], mapping[v])))
                             for u, v in f.edges))
        candidates.append(edges)
    return MultiGraph(f.n, min(candidates))


def quotient(f, coloring, color):
    """Contract all other colors; a retained loop invalidates the term."""
    parent = list(range(f.n))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for (u, v), label in zip(f.edges, coloring):
        if label != color:
            parent[find(u)] = find(v)
    classes = [find(u) for u in range(f.n)]
    order = [classes[0]] + sorted(set(classes) - {classes[0]})
    mapping = {old: new for new, old in enumerate(order)}
    kept = []
    for (u, v), label in zip(f.edges, coloring):
        if label == color:
            a, b = mapping[classes[u]], mapping[classes[v]]
            if a == b:
                return None
            kept.append(tuple(sorted((a, b))))
    return canonical(MultiGraph(len(order), tuple(sorted(kept))))


@lru_cache(None)
def coproduct(f, arity, onto=False):
    result = Counter()
    for coloring in product(range(arity), repeat=len(f.edges)):
        if onto and len(set(coloring)) != arity:
            continue
        factors = tuple(quotient(f, coloring, i) for i in range(arity))
        if all(x is not None for x in factors):
            result[factors] += 1
    return result


@lru_cache(None)
def hom(f, target):
    """Direct root-preserving assignments into a simple target."""
    neighbors = [set() for _ in range(f.n)]
    for u, v in f.edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    mapping = {0: 0}

    def visit():
        if len(mapping) == f.n:
            return 1
        u = max((u for u in range(f.n) if u not in mapping),
                key=lambda u: (len(neighbors[u] & mapping.keys()),
                               len(neighbors[u])))
        fixed = neighbors[u] & mapping.keys()
        candidates = range(target.n)
        if fixed:
            candidates = target.neighbors(mapping[min(fixed)])
        total = 0
        for v in candidates:
            if all(target.rows[v] & (1 << mapping[x]) for x in fixed):
                mapping[u] = v
                total += visit()
                del mapping[u]
        return total

    return visit()


@lru_cache(None)
def hom_cumulant(f, target):
    value = Q(0)
    for arity in range(1, len(f.edges) + 1):
        value += Q((-1) ** (arity - 1), arity) * sum(
            multiplicity * math.prod(hom(q, target) for q in factors)
            for factors, multiplicity in coproduct(f, arity, True).items())
    return value


def reconstruct_hom(f, target):
    if not f.edges:
        return Q(1)
    return sum((Q(1, math.factorial(arity)) * sum(
        multiplicity * math.prod(hom_cumulant(q, target) for q in factors)
        for factors, multiplicity in coproduct(f, arity, True).items())
        for arity in range(1, len(f.edges) + 1)), Q(0))


@lru_cache(None)
def walk_moments(g, length, closed=False):
    counts = [int(u == 0) for u in range(g.n)]
    result = [1]
    for _ in range(length):
        counts = [sum(counts[v] for v in g.neighbors(u)) for u in range(g.n)]
        result.append(counts[0] if closed else sum(counts))
    return tuple(result)


def cumulants(moments):
    result = [0]
    for n in range(1, len(moments)):
        result.append(moments[n] - sum(math.comb(n - 1, j - 1)
                      * result[j] * moments[n - j] for j in range(1, n)))
    return tuple(result)


def root_degree(type_id):
    return TYPES.reps[type_id].rows[0].bit_count()


def add_arrays(a, b):
    result = defaultdict(Q, a)
    for key, value in b.items():
        result[key] += value
    return clean(result)


def truncate_degree(a, bound):
    return {t: c for t, c in a.items() if root_degree(t) <= bound and c}


def main():
    checks = Counter()

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] += 1

    small = set()
    for n in range(1, 5):
        possible = list(combinations(range(n), 2))
        for mask in range(1 << len(possible)):
            g = graph(n, (e for i, e in enumerate(possible) if mask & (1 << i)))
            if len(distances(g, 0)) == n:
                small.add(TYPES.register(g))
    patterns = {canonical(multi(TYPES.reps[t])) for t in small}
    patterns.update([
        MultiGraph(2, ((0, 1), (0, 1))),
        MultiGraph(3, ((0, 1), (0, 1), (1, 2))),
        MultiGraph(3, ((0, 1), (0, 1), (0, 2), (1, 2))),
    ])

    for f in patterns:
        left, right = Counter(), Counter()
        for (a, b), multiplicity in coproduct(f, 2).items():
            check("coproduct_edge_grading", len(a.edges) + len(b.edges) == len(f.edges))
            for (a1, a2), inner in coproduct(a, 2).items():
                left[(a1, a2, b)] += multiplicity * inner
            for (b1, b2), inner in coproduct(b, 2).items():
                right[(a, b1, b2)] += multiplicity * inner
        check("symbolic_coassociativity", left == right == coproduct(f, 3))
        check("symbolic_cocommutativity", coproduct(f, 2) == Counter(
            {(b, a): c for (a, b), c in coproduct(f, 2).items()}))

    radius_summary = []
    for radius in (1, 2, 3):
        types = sorted(t for t in small
                       if max(distances(TYPES.reps[t], 0).values()) <= radius)
        sources = sorted((f for f in patterns
                          if max(distances(graph(f.n, f.edges), 0).values()) <= radius),
                         key=lambda f: (f.n, f.edges))
        keys = {}
        for t in types:
            g = TYPES.reps[t]
            walks = walk_moments(g, radius)
            key = cumulants(walks)[1:] + tuple(hom_cumulant(f, g) for f in sources)
            keys[t] = key
            check("walk_size_bound", g.n <= sum(walks))
            for f in sources:
                check("log_exp_reconstruction", reconstruct_hom(f, g) == hom(f, g))
        check("small_ball_separation", len(set(keys.values())) == len(types))
        for a, b in product(types, repeat=2):
            g, h = TYPES.reps[a], TYPES.reps[b]
            t = truncated_type_product(a, b, radius)
            target = TYPES.reps[t]
            check("axis_embedding_size_bound", target.n >= g.n + h.n - 1)
            ca = cumulants(walk_moments(g, radius))
            cb = cumulants(walk_moments(h, radius))
            cp = cumulants(walk_moments(target, radius))
            check("open_walk_cumulant_additivity",
                  cp == tuple(x + y for x, y in zip(ca, cb)))
            length = 2 * radius + 1
            ca = cumulants(walk_moments(g, length, True))
            cb = cumulants(walk_moments(h, length, True))
            cp = cumulants(walk_moments(target, length, True))
            check("closed_walk_cumulant_additivity",
                  cp == tuple(x + y for x, y in zip(ca, cb)))
            for f in sources:
                expanded = sum(c * hom(q1, g) * hom(q2, h)
                               for (q1, q2), c in coproduct(f, 2).items())
                check("hom_cartesian_coproduct", hom(f, target) == expanded)
                check("hom_cumulant_additivity", hom_cumulant(f, target)
                      == hom_cumulant(f, g) + hom_cumulant(f, h))
        for shift in range(len(types)):
            aa = {t: Q((-1) ** j * (j + 1)) for j, t in enumerate(types)
                  if (j + shift) % 3 != 0}
            bb = {t: Q((-1) ** (j + shift), j + 1) for j, t in enumerate(types)
                  if (2 * j + shift) % 4 != 0}
            if aa and bb:
                a, b = min(aa, key=keys.get), min(bb, key=keys.get)
                target = truncated_type_product(a, b, radius)
                check("leading_coefficient", convolution(aa, bb, radius)[target]
                      == aa[a] * bb[b])
        radius_summary.append({"radius": radius, "ball_types": len(types),
                               "source_patterns": len(sources)})

    one = TYPES.register(graph(1))
    for radius, generators in [(1, [complete(2), complete(3)]),
                               (2, [complete(2), path(3)])]:
        bound = 5
        support = [TYPES.register(ball(g, 0, radius)[0]) for g in generators]
        a = {one: Q(3), support[0]: Q(2), support[1]: Q(-1)}
        universe = {one}
        frontier = {one}
        for _ in range(bound):
            frontier = {truncated_type_product(t, g, radius)
                        for t in frontier for g in support}
            frontier = {t for t in frontier if root_degree(t) <= bound}
            universe.update(frontier)
        reciprocal = {one: Q(1, 3)}
        for target in sorted(universe - {one}, key=lambda t: (root_degree(t), t)):
            total = sum((a[c] * reciprocal.get(d, Q(0))
                         for c in support for d in universe
                         if root_degree(d) < root_degree(target)
                         and truncated_type_product(c, d, radius) == target), Q(0))
            reciprocal[target] = -total / a[one]
        z = {t: -coefficient / a[one] for t, coefficient in a.items() if t != one}
        geometric, power = {}, {one: Q(1)}
        for _ in range(bound + 1):
            geometric = add_arrays(geometric, {t: c / a[one] for t, c in power.items()})
            power = truncate_degree(convolution(power, z, radius), bound)
        check("formal_reciprocal_recursion", clean(reciprocal) == geometric)
        check("formal_reciprocal_product",
              truncate_degree(convolution(a, reciprocal, radius), bound) == {one: Q(1)})

    hypercube = graph(1)
    for dimension in range(8):
        for radius in range(1, 4):
            size = ball(hypercube, 0, radius)[0].n
            expected = sum(math.comb(dimension, j)
                           for j in range(min(radius, dimension) + 1))
            check("hypercube_ball_size", size == expected)
            check("hypercube_polynomial_bound", size <= (dimension + 1) ** radius)
        if dimension < 7:
            hypercube = cart(hypercube, complete(2))

    odd_cycle_summary = []
    for n in (3, 5, 7, 9, 11, 13):
        radius = (n - 1) // 2
        short, long = cycle(n), cycle(2 * n)
        q_short = cumulants(walk_moments(short, n, True))[n]
        q_long = cumulants(walk_moments(long, n, True))[n]
        check("odd_cycle_cumulants", (q_short, q_long) == (2, 0))
        for g in (short, long):
            check("closed_walk_locality", walk_moments(g, n, True)
                  == walk_moments(ball(g, 0, radius)[0], n, True))
        terms = [(Q(1), graph(1)), (Q(-1, 4 * n), long), (Q(1, 2 * n), short)]
        value = Q(0)
        for t, coefficient in linear_hist(terms, radius).items():
            q = cumulants(walk_moments(TYPES.reps[t], n, True))[n]
            check("character_phase_is_real_in_example", q % 2 == 0)
            value += coefficient * (-1) ** (q // 2)
        check("nonunit_character_vanishes", value == 0)
        degree_series = defaultdict(Q)
        for coefficient, g in terms:
            for row in g.rows:
                degree_series[row.bit_count()] += coefficient
        check("all_degree_characters_miss_obstruction", clean(degree_series) == {0: Q(1)})
        for r in range((n - 1) // 2):
            check("nonunits_eventually_equal_identity_locally",
                  linear_hist(terms, r) == {one: Q(1)})
        odd_cycle_summary.append({"odd_length": n, "detecting_radius": radius,
                                  "short_cumulant": q_short, "long_cumulant": q_long})

    report = {
        "status": "passed", "arithmetic": "exact integers and fractions",
        "total_checks": sum(checks.values()), "checks": dict(sorted(checks.items())),
        "small_connected_rooted_types": len(small),
        "multigraph_patterns": len(patterns), "radius_cases": radius_summary,
        "odd_cycle_nonunits": odd_cycle_summary,
        "scope": "Finite regressions support, and do not replace, the universal proofs in MULTIPLICATION_AND_UNITS.md.",
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
