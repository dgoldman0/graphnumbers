"""Exact finite witnesses for the cut-line entire-function subalgebra.

Rooted graph constructions and rational formal series are independent of the
specialized cut-line exponential runtime. These checks verify examples and
coefficients; the infinite spectrum, topology, and convergence assertions
require the proof in UNBOUNDED_VARIATION_INVERSION.md.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
from itertools import combinations
import json
from math import factorial
from pathlib import Path

from graphlocal import (CutLineDefect, LocalHistogram, cartesian, complete,
                        cycle, graph, path, star)
from graphlocal.graphs import ball, induced, integer, isomorphic


def square_count(g, root=0):
    """All simple four-cycles through root, including cycles with chords."""
    return sum((g.rows[u] & g.rows[v] & ~(1 << root)).bit_count()
               for u, v in combinations(g.neighbors(root), 2))


def sphere_series(g, radius):
    """Independent breadth-first distance counts through a fixed radius."""
    integer(radius, "radius")
    result, visited, frontier = [Q(1)], {0}, {0}
    for _ in range(radius):
        following = {v for u in frontier for v in g.neighbors(u)} - visited
        result.append(Q(len(following)))
        visited.update(following)
        frontier = following
    return tuple(result)


def series_product(left, right, order):
    return tuple(sum((left[k] * right[n-k] for k in range(n + 1)), Q(0))
                 for n in range(order + 1))


def series_log(value):
    if value[0] != 1:
        raise ValueError("Formal logarithm requires constant one")
    order = len(value) - 1
    remainder = (Q(0),) + tuple(value[1:])
    power = (Q(1),) + (Q(0),) * order
    result = [Q(0)] * (order + 1)
    for k in range(1, order + 1):
        power = series_product(power, remainder, order)
        for j in range(order + 1):
            result[j] += Q((-1) ** (k + 1), k) * power[j]
    return tuple(result)


def additive_coordinates(g, radius):
    """Return (c_0,...,c_(r-1),c_L) using square and sphere-log invariants."""
    integer(radius, "radius", 2)
    degree = g.rows[0].bit_count()
    number = Q(3 * degree - degree ** 2 + 2 * square_count(g), 2)
    line_log = series_log((Q(1),) + (Q(2),) * radius)
    sphere_log = series_log(sphere_series(g, radius))
    residual = [a - number * b for a, b in zip(sphere_log, line_log)]
    coordinates = []
    for j in range(radius):
        # ell_j = log(1-z**(j+1)/(1+z)), independently from path graphs.
        base = (Q(1),) + tuple(Q(0) if n <= j else Q((-1) ** (n - j))
                                    for n in range(1, radius + 1))
        basis = series_log(base)
        coordinate = -residual[j + 1]
        coordinates.append(coordinate)
        residual = [a - coordinate * b for a, b in zip(residual, basis)]
    if any(residual):
        raise AssertionError("Triangular sphere-log expansion left a residual")
    return tuple(coordinates) + (number - sum(coordinates),)


def phase_turn(coordinates, turns):
    """Exact phase modulo one turn, avoiding floating complex arithmetic."""
    return sum((a * Q(b) for a, b in zip(coordinates, turns)), Q(0)) % 1


def path_atoms(radius):
    integer(radius, "radius", 2)
    atoms = []
    for short in range(radius):
        g = path(radius + short + 1)
        atoms.append(induced(g, [short] + [v for v in range(g.n) if v != short]))
    g = path(2 * radius + 1)
    atoms.append(induced(g, [radius] + [v for v in range(g.n) if v != radius]))
    return tuple(atoms)


def compositions(total, width):
    if width == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for following in compositions(total - first, width - 1):
                yield (first,) + following


def monomial_graph(atoms, exponents, radius):
    value = complete(1)
    for atom, count in zip(atoms, exponents):
        for _ in range(count):
            # Full finite Cartesian construction, followed by truncation.
            value = ball(cartesian(value, atom), 0, radius)
    return value


def monomial_coefficient(exponents, radius):
    degree = sum(exponents)
    multinomial = factorial(degree)
    for exponent in exponents:
        multinomial //= factorial(exponent)
    return Q(multinomial * 2 ** degree * (-radius) ** exponents[-1])


def exponential_tail_majorant(time, radius, weight, cutoff):
    """Rational geometric majorant for the documented weighted series tail."""
    time = abs(Q(time))
    integer(radius, "radius", 2)
    integer(weight, "weight", 1)
    integer(cutoff, "cutoff")
    lam, degree = 4 * radius * time, radius * weight
    ratio = lam / (cutoff + 2) * Q(2 * cutoff + 5, 2 * cutoff + 3) ** degree
    if ratio >= 1:
        raise ValueError("Geometric tail ratio must be below one")
    first = (radius + 1) ** weight * (2 * cutoff + 3) ** degree
    first *= lam ** (cutoff + 1) / factorial(cutoff + 1)
    return first / (1 - ratio)


def verify_unbounded_inverse():
    checks = 0

    def require(condition, label):
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1

    paw = graph(4, ((0, 1), (1, 2), (2, 0), (0, 3)))
    fixtures = (complete(1), complete(2), complete(3), cycle(4), path(3),
                paw, star(3))
    rooted_pairs = 0
    for radius in (2, 3):
        for g in fixtures:
            for h in fixtures:
                product = cartesian(g, h)
                for u in range(g.n):
                    for v in range(h.n):
                        left = ball(g, u, radius)
                        right = ball(h, v, radius)
                        joined = ball(product, u * h.n + v, radius)
                        d, e = left.rows[0].bit_count(), right.rows[0].bit_count()
                        require(square_count(joined) == square_count(left) + square_count(right) + d * e,
                                "root square count obeys its Cartesian correction")
                        require(sphere_series(joined, radius)
                                == series_product(sphere_series(left, radius),
                                                  sphere_series(right, radius), radius),
                                "root sphere series multiplies after truncation")
                        a, b = additive_coordinates(left, radius), additive_coordinates(right, radius)
                        c = additive_coordinates(joined, radius)
                        require(c == tuple(x + y for x, y in zip(a, b)),
                                "square/sphere-log coordinates are additive")
                        turns = tuple(Q(j + 1, j + 4) for j in range(radius + 1))
                        require(phase_turn(c, turns) == (phase_turn(a, turns) + phase_turn(b, turns)) % 1,
                                "rational phase characters multiply exactly")
                        rooted_pairs += 1

    monomial_rows, phase_rows, exponential_rows = [], [], []
    gaussian = ((1, 0), (0, 1), (-1, 0), (0, -1))
    for radius, max_power in ((2, 3), (3, 2)):
        atoms = path_atoms(radius)
        for j, atom in enumerate(atoms):
            require(additive_coordinates(atom, radius)
                    == tuple(Q(int(i == j)) for i in range(radius + 1)),
                    "independently constructed path atoms have unit coordinates")
        source = LocalHistogram(radius, ((atom, Q(2) if j < radius else Q(-2 * radius))
                                         for j, atom in enumerate(atoms)))
        require(source == CutLineDefect().local(radius), "explicit path atoms equal cut-line marginal")
        powers = [LocalHistogram(radius, ((complete(1), Q(1)),))]
        all_coordinate_vectors = set()
        for power in range(max_power + 1):
            terms, graphs = [], []
            for exponents in compositions(power, radius + 1):
                g = monomial_graph(atoms, exponents, radius)
                coordinates = additive_coordinates(g, radius)
                require(coordinates == exponents, "coordinates recover all monomial exponents")
                require(coordinates not in all_coordinate_vectors,
                        "different monomials, including different powers, have distinct coordinates")
                all_coordinate_vectors.add(coordinates)
                require(all(not isomorphic(g, previous, rooted=True) for previous in graphs),
                        "explicit same-power rooted monomials are pairwise nonisomorphic")
                graphs.append(g)
                terms.append((g, monomial_coefficient(exponents, radius)))
            expanded = LocalHistogram(radius, terms)
            if power:
                powers.append(powers[-1].multiply(source))
            require(expanded == powers[power], "multinomial coefficients agree with independent histogram convolution")
            require(expanded.norm(0) == (4 * radius) ** power, "cut-line power total variation is exact")
            for weight in (1, 2):
                require(expanded.norm(weight)
                        <= (radius + 1) ** weight * (1 + 2 * power) ** (radius * weight) * (4 * radius) ** power,
                        "weighted local power bound dominates the explicitly constructed graph sum")
            monomial_rows.append({"radius": radius, "power": power,
                                  "types": len(expanded.values),
                                  "variation": str(expanded.norm(0))})

        for u_index, u in enumerate(gaussian):
            for v_index, v in enumerate(gaussian):
                turns = (Q(u_index, 4),) * radius + (Q(v_index, 4),)
                value = [Q(0), Q(0)]
                for key, coefficient in source.values.items():
                    turn = phase_turn(additive_coordinates(key.graph, radius), turns)
                    phase = gaussian[int(4 * turn)]
                    value = [x + coefficient * y for x, y in zip(value, phase)]
                expected = [2 * radius * (x-y) for x, y in zip(u, v)]
                require(value == expected, "fourth-root phase witness gives 2r(u-v)")
                require(sum(x*x for x in value) <= (4 * radius) ** 2,
                        "phase witness lies in the stated local spectral disk")
                phase_rows.append({"radius": radius, "atom_quarter_turns": u_index,
                                   "line_quarter_turns": v_index,
                                   "value_real": str(value[0]), "value_imaginary": str(value[1])})

        for time in (Q(-1, 4), Q(1, 4)):
            partial = LocalHistogram(radius)
            for n, power_histogram in enumerate(powers):
                partial += power_histogram.scale(time ** n / factorial(n))
            exact_variation = sum((Q((4 * radius) ** n) * abs(time) ** n / factorial(n)
                                   for n in range(max_power + 1)), Q(0))
            require(partial.norm(0) == exact_variation,
                    "truncated exponential has no cross-power cancellation")
            exponential_rows.append({"radius": radius, "time": str(time),
                                     "cutoff": max_power, "variation": str(exact_variation)})

    tail_rows = []
    for radius, weight, time, cutoff in ((2, 1, Q(1, 4), 8), (3, 1, Q(1, 4), 10),
                                        (2, 2, Q(-1, 4), 12)):
        bound = exponential_tail_majorant(time, radius, weight, cutoff)
        lam = 4 * radius * abs(time)
        finite_tail = sum((Q((radius + 1) ** weight * (1 + 2*n) ** (radius * weight))
                           * lam ** n / factorial(n) for n in range(cutoff + 1, cutoff + 41)), Q(0))
        require(finite_tail <= bound, "rational tail majorant dominates forty independently summed terms")
        tail_rows.append({"radius": radius, "weight": weight, "time": str(time),
                          "cutoff": cutoff, "bound": str(bound),
                          "finite_terms_checked": 40})

    for time in (Q(-2, 3), Q(0), Q(1, 4)):
        for n in range(13):
            coefficient = sum((time ** j * (-time) ** (n-j) / (factorial(j) * factorial(n-j))
                               for j in range(n + 1)), Q(0))
            require(coefficient == int(n == 0), "formal exponential inverse coefficients cancel exactly")

    return {"success": True, "checks": checks, "rooted_cartesian_pairs": rooted_pairs,
            "monomial_fixtures": monomial_rows, "phase_witnesses": phase_rows,
            "truncated_exponentials": exponential_rows, "weighted_tail_fixtures": tail_rows,
            "scope": "Exact finite witnesses; infinite spectrum, entire-function identification, and convergence rely on proofs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_unbounded_inverse()
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
