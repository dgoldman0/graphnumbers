#!/usr/bin/env python3
"""Exact finite regressions for local_graph_completion.tex (Python >= 3.10).

No external packages, spectral approximations, or probabilistic isomorphism
hashes are used. The isomorphism checker performs exact backtracking after
joint color refinement. This is a small research verifier, not a large-graph
canonicalization engine. Universal proofs are in the accompanying note.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations, permutations, product
from pathlib import Path


@dataclass(frozen=True)
class Graph:
    rows: tuple[int, ...]

    def __post_init__(self):
        n = len(self.rows)
        for u, row in enumerate(self.rows):
            assert 0 <= row < (1 << n) and not (row & (1 << u))
            for v in range(n):
                assert bool(row & (1 << v)) == bool(self.rows[v] & (1 << u))

    @property
    def n(self):
        return len(self.rows)

    def neighbors(self, u):
        return [v for v in range(self.n) if self.rows[u] & (1 << v)]

    @property
    def edges(self):
        return sum(row.bit_count() for row in self.rows) // 2

    @property
    def isolated(self):
        return sum(row == 0 for row in self.rows)

    @property
    def triangles(self):
        return sum(all(self.rows[u] & (1 << v) for u, v in combinations(t, 2))
                   for t in combinations(range(self.n), 3))

    def component_count(self):
        unseen = set(range(self.n))
        count = 0
        while unseen:
            count += 1
            unseen -= set(distances(self, min(unseen)))
        return count


def graph(n, edges=()):
    rows = [0] * n
    for u, v in edges:
        rows[u] |= 1 << v
        rows[v] |= 1 << u
    return Graph(tuple(rows))


def complete(n):
    return graph(n, combinations(range(n), 2))


def cycle(n):
    assert n >= 3
    return graph(n, ((i, (i + 1) % n) for i in range(n)))


def path(n):
    return graph(n, ((i, i + 1) for i in range(n - 1)))


def star(leaves):
    return graph(leaves + 1, ((0, i) for i in range(1, leaves + 1)))


def union(*gs):
    rows, offset = [], 0
    for g in gs:
        rows.extend(row << offset for row in g.rows)
        offset += g.n
    return Graph(tuple(rows))


@lru_cache(None)
def cart(a, b):
    rows = []
    for u in range(a.n):
        for v in range(b.n):
            row = sum(1 << (x * b.n + v) for x in a.neighbors(u))
            row |= sum(1 << (u * b.n + y) for y in b.neighbors(v))
            rows.append(row)
    return Graph(tuple(rows))


def distances(g, root):
    ds, queue = {root: 0}, deque([root])
    while queue:
        u = queue.popleft()
        for v in g.neighbors(u):
            if v not in ds:
                ds[v] = ds[u] + 1
                queue.append(v)
    return ds


def induced(g, order):
    return graph(len(order), ((i, j) for i in range(len(order))
                             for j in range(i + 1, len(order))
                             if g.rows[order[i]] & (1 << order[j])))


@lru_cache(None)
def ball(g, root, radius):
    ds = distances(g, root)
    order = (root,) + tuple(sorted(u for u in ds if u != root and ds[u] <= radius))
    return induced(g, order), order


ISO_CALLS = 0


def isomorphic(a, b, roots=None):
    """Exact isomorphism with optional distinguished roots."""
    global ISO_CALLS
    ISO_CALLS += 1
    if a.n != b.n or a.edges != b.edges:
        return False
    if a.n == 0:
        return True
    ra, rb = roots if roots is not None else (None, None)
    ka = [(int(u == ra), a.rows[u].bit_count()) for u in range(a.n)]
    kb = [(int(u == rb), b.rows[u].bit_count()) for u in range(b.n)]
    palette = {key: i for i, key in enumerate(sorted(set(ka + kb)))}
    ca, cb = [palette[k] for k in ka], [palette[k] for k in kb]
    while True:
        ka = [(ca[u], tuple(sorted(ca[v] for v in a.neighbors(u)))) for u in range(a.n)]
        kb = [(cb[u], tuple(sorted(cb[v] for v in b.neighbors(u)))) for u in range(b.n)]
        palette = {key: i for i, key in enumerate(sorted(set(ka + kb)))}
        na, nb = [palette[k] for k in ka], [palette[k] for k in kb]
        if Counter(na) != Counter(nb):
            return False
        stable = len(set(na + nb)) == len(set(ca + cb))
        ca, cb = na, nb
        if stable:
            break
    mapping = {} if roots is None else {ra: rb}
    used = set(mapping.values())

    def candidates(u):
        return [v for v in range(b.n) if v not in used and ca[u] == cb[v]
                and all(bool(a.rows[u] & (1 << x)) == bool(b.rows[v] & (1 << y))
                        for x, y in mapping.items())]

    def dfs():
        if len(mapping) == a.n:
            return True
        choices = [(candidates(u), u) for u in range(a.n) if u not in mapping]
        opts, u = min(choices, key=lambda item: len(item[0]))
        for v in opts:
            mapping[u] = v
            used.add(v)
            if dfs():
                return True
            del mapping[u]
            used.remove(v)
        return False

    return dfs()


class RootTypes:
    def __init__(self):
        self.reps = []
        self.buckets = defaultdict(list)

    @lru_cache(None)
    def register(self, g):
        ds = distances(g, 0)
        signature = (g.n, g.edges, g.rows[0].bit_count(),
                     tuple(sorted((ds[u], g.rows[u].bit_count()) for u in range(g.n))))
        for i in self.buckets[signature]:
            if isomorphic(g, self.reps[i], (0, 0)):
                return i
        i = len(self.reps)
        self.reps.append(g)
        self.buckets[signature].append(i)
        return i


TYPES = RootTypes()


@lru_cache(None)
def hist(g, r):
    return Counter(TYPES.register(ball(g, u, r)[0]) for u in range(g.n))


def clean(h):
    return {key: value for key, value in h.items() if value}


def linear_hist(terms, r):
    h = defaultdict(F)
    for coefficient, g in terms:
        for key, count in hist(g, r).items():
            h[key] += coefficient * count
    return clean(h)


def seminorm(terms, r, k):
    return sum((TYPES.reps[key].n ** k * abs(c)
                for key, c in linear_hist(terms, r).items()), F(0))


def multiply(x, y):
    return [(a * b, cart(g, h)) for a, g in x for b, h in y]


def observable(terms, name):
    def val(g):
        return g.component_count() if name == 'components' else getattr(g, name)
    return sum((c * val(g) for c, g in terms), F(0))


@lru_cache(None)
def truncated_type_product(i, j, r):
    return TYPES.register(ball(cart(TYPES.reps[i], TYPES.reps[j]), 0, r)[0])


def convolution(a, b, r):
    result = defaultdict(F)
    for i, ci in a.items():
        for j, cj in b.items():
            result[truncated_type_product(i, j, r)] += ci * cj
    return clean(result)


def edge_structure(g, labels):
    vs = frozenset(labels)
    es = frozenset(frozenset((labels[u], labels[v]))
                   for u in range(g.n) for v in g.neighbors(u) if u < v)
    return vs, es


def map_count(source, target, kind='hom', rooted=False):
    if source.n == 0:
        return 1
    tail = product(range(target.n), repeat=source.n - int(rooted))
    count = 0
    for t in tail:
        images = (0,) + t if rooted else t
        if kind != 'hom' and len(set(images)) != len(images):
            continue
        good = True
        for u, v in combinations(range(source.n), 2):
            se = bool(source.rows[u] & (1 << v))
            te = bool(target.rows[images[u]] & (1 << images[v]))
            if (se and not te) or (kind == 'induced' and se != te):
                good = False
                break
        count += good
    return count


def charpoly(g):
    """Adjacency characteristic polynomial by exact Newton identities."""
    n = g.n
    a = [[F(bool(g.rows[i] & (1 << j))) for j in range(n)] for i in range(n)]
    power = [[F(i == j) for j in range(n)] for i in range(n)]
    traces = [F(0)]
    for _ in range(n):
        power = [[sum((power[i][h] * a[h][j] for h in range(n)), F(0))
                  for j in range(n)] for i in range(n)]
        traces.append(sum((power[i][i] for i in range(n)), F(0)))
    coeffs = [F(1)]
    for k in range(1, n + 1):
        coeffs.append(-sum((coeffs[k - j] * traces[j] for j in range(1, k + 1)), F(0)) / k)
    assert all(c.denominator == 1 for c in coeffs)
    return [int(c) for c in coeffs]


def main():
    empty, one, two, p3, k3, c4, s4 = graph(0), complete(1), complete(2), path(3), complete(3), cycle(4), star(4)
    samples = [one, two, p3, k3, c4, s4]
    checks = Counter()

    # Exact isomorphism grouping and separation of every unlabelled graph <=4 vertices.
    reps = []
    labelled = 0
    for n in range(5):
        potential = list(combinations(range(n), 2))
        for mask in range(1 << len(potential)):
            g = graph(n, (edge for j, edge in enumerate(potential) if mask & (1 << j)))
            labelled += 1
            if not any(isomorphic(g, h) for h in reps):
                reps.append(g)
    assert len(reps) == 19 and labelled == 76
    signatures = {tuple(sorted(hist(g, 4).items())) for g in reps}
    assert len(signatures) == len(reps)
    checks['distinct_graphs_up_to_four_vertices'] = len(reps)
    for g in reps:
        for order in permutations(range(g.n)):
            h = induced(g, order)
            assert isomorphic(g, h)
            for r in range(3):
                assert hist(g, r) == hist(h, r)
                checks['relabeling_histograms'] += 1

    # Independent explicit vertex labels check the product-ball identity.
    for a in samples:
        for b in samples:
            ab = cart(a, b)
            for u in range(a.n):
                for v in range(b.n):
                    for r in range(4):
                        direct, order = ball(ab, u * b.n + v, r)
                        direct_labels = [(z // b.n, z % b.n) for z in order]
                        ba, oa = ball(a, u, r)
                        bb, ob = ball(b, v, r)
                        truncated, ot = ball(cart(ba, bb), 0, r)
                        truncated_labels = [(oa[z // bb.n], ob[z % bb.n]) for z in ot]
                        assert edge_structure(direct, direct_labels) == edge_structure(truncated, truncated_labels)
                        assert direct.n <= ba.n * bb.n
                        checks['product_ball_identity_and_size_bound'] += 1

    for a in samples + [empty, union(one, two), union(two, two)]:
        for b in samples + [empty]:
            for r in range(3):
                assert hist(cart(a, b), r) == convolution(hist(a, r), hist(b, r), r)
                checks['graph_histogram_convolution'] += 1

    rng = random.Random(20260930)
    for _ in range(40):
        x = [(F(rng.randint(-3, 3), rng.randint(1, 4)), rng.choice(samples)) for _ in range(3)]
        y = [(F(rng.randint(-3, 3), rng.randint(1, 4)), rng.choice(samples)) for _ in range(3)]
        xy = multiply(x, y)
        assert observable(xy, 'n') == observable(x, 'n') * observable(y, 'n')
        assert observable(xy, 'edges') == observable(x, 'edges') * observable(y, 'n') + observable(x, 'n') * observable(y, 'edges')
        assert observable(xy, 'isolated') == observable(x, 'isolated') * observable(y, 'isolated')
        checks['signed_observable_product_identities'] += 3
        for r in range(3):
            assert linear_hist(xy, r) == convolution(linear_hist(x, r), linear_hist(y, r), r)
            checks['signed_histogram_convolution'] += 1
            for k in range(1, 4):
                assert seminorm(x + y, r, k) <= seminorm(x, r, k) + seminorm(y, r, k)
                assert seminorm(xy, r, k) <= seminorm(x, r, k) * seminorm(y, r, k)
                checks['signed_triangle_and_product_bound'] += 2

    for a in [F(-7, 3), F(0), F(1, 5), F(5, 2)]:
        for r in range(4):
            for k in range(1, 4):
                assert seminorm([(a, one)], r, k) == abs(a)
                assert seminorm([(a, empty)], r, k) == 0
                checks['scalar_and_zero'] += 2
    for g in samples:
        for r in range(3):
            assert hist(cart(g, one), r) == hist(g, r)
            assert hist(union(g, g), r) == {t: 2 * n for t, n in hist(g, r).items()}
            assert seminorm([(F(1), g), (F(-1), g)], r, 1) == 0
            checks['unit_repeated_components_and_cancellation'] += 3

    for source in [one, two, p3, k3]:
        radius = max(distances(source, u)[v] for u in range(source.n) for v in range(source.n))
        k = max(1, source.n - 1)
        for g in samples:
            for kind in ['hom', 'injective', 'induced']:
                direct = map_count(source, g, kind)
                local = sum(c * map_count(source, TYPES.reps[t], kind, rooted=True)
                            for t, c in hist(g, radius).items())
                assert direct == local
                assert direct <= seminorm([(F(1), g)], radius, k)
                checks['connected_pattern_local_formula_and_bound'] += 1

    lattice_rows = []
    for r in range(4):
        n = 2 * r + 3
        a, b = cycle(n), cycle(n + 2)
        ha, hb = linear_hist([(F(1, n), a)], r), linear_hist([(F(1, n + 2), b)], r)
        assert ha == hb and len(ha) == 1
        only = next(iter(ha))
        expected = TYPES.register(ball(path(2 * r + 1), r, r)[0])
        assert only == expected
        for k in range(1, 4):
            assert seminorm([(F(1, n), a)], r, k) == (2 * r + 1) ** k
            checks['cycle_stabilization'] += 1
        grid_hist = linear_hist([(F(1, n * (n + 2)), cart(a, b))], r)
        assert grid_hist == convolution(ha, hb, r) and len(grid_hist) == 1
        grid_type = next(iter(grid_hist))
        assert TYPES.reps[grid_type].n == 1 + 2 * r * (r + 1)
        assert observable([(F(1, n * (n + 2)), cart(a, b))], 'edges') == 2
        checks['normalized_cartesian_grid'] += 1
        lattice_rows.append({'radius': r, 'cycle_size': 2 * r + 1, 'square_grid_size': TYPES.reps[grid_type].n})

    for n in [7, 9, 11, 13]:
        z = [(F(1), cycle(2 * n)), (F(-2), cycle(n))]
        assert observable(z, 'components') == -1
        assert observable(z, 'n') == observable(z, 'edges') == 0
        for r in range(5):
            if n > 2 * r + 1:
                assert linear_hist(z, r) == {}
                checks['component_count_discontinuity_example'] += 1

    triangle_rows = []
    for n in [3, 4, 8, 16]:
        y = [(F(1, n ** 3), complete(n))]
        assert seminorm(y, 0, 1) == F(1, n ** 2)
        assert seminorm(y, 1, 1) == F(1, n)
        assert seminorm(y, 1, 2) == 1
        assert observable(y, 'triangles') == F(math.comb(n, 3), n ** 3)
        checks['triangle_counterexample_and_refinement'] += 1
        triangle_rows.append({'n': n, 'p_0_1': str(seminorm(y, 0, 1)),
                              'p_1_1': str(seminorm(y, 1, 1)), 'p_1_2': str(seminorm(y, 1, 2)),
                              'triangle_count': str(observable(y, 'triangles'))})

    spectral_a, spectral_b = union(c4, one, one), union(star(4), one)
    assert spectral_a.n == spectral_b.n == 6 and spectral_a.edges == spectral_b.edges == 4
    polynomial_a, polynomial_b = charpoly(spectral_a), charpoly(spectral_b)
    assert polynomial_a == polynomial_b == [1, 0, -4, 0, 0, 0, 0]
    assert not isomorphic(spectral_a, spectral_b)
    separation = seminorm([(F(1), spectral_a), (F(-1), spectral_b)], 1, 1)
    assert separation > 0
    checks['same_size_cospectral_pair_separated'] += 1

    return {
        'status': 'all checks passed',
        'candidate': 'A_loc v0.1; all integer powers of rooted-neighborhood size',
        'arithmetic': 'disjoint union and Cartesian product',
        'coefficient_arithmetic': 'exact fractions',
        'isomorphism_method': 'exact backtracking after joint color refinement',
        'checks': dict(checks),
        'total_recorded_checks': sum(checks.values()),
        'isomorphism_calls': ISO_CALLS,
        'registered_rooted_types': len(TYPES.reps),
        'lattice_neighborhoods': lattice_rows,
        'triangle_counterexample': triangle_rows,
        'cospectral_regression': {
            'graphs': ['C4 disjoint_union 2 K1', 'K1,4 disjoint_union K1'],
            'characteristic_polynomial_coefficients': polynomial_a,
            'p_1_1_of_difference': str(separation)},
        'limits': 'Finite checks support the implementation and examples. Universal proofs, completion, and continuity are in the note.'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = main()
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
