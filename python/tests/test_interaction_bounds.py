"""Geometry bounds checked against independent full-matrix expansions."""
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from itertools import combinations
from math import factorial
import unittest

from graphlocal import (EdgeInteraction, apply_edge_edits, controlled_heat, cycle,
                        disjoint_union, graph, heat_return, moment_profile, path, star)
from graphlocal.graphs import BudgetExceeded
from graphlocal.interaction_bounds import GeometricInteraction, interaction_geometry, interaction_heat_bound


def mixed_returns(before, edits, degree, steps):
    result = [Q(0)] * (steps + 1)
    for size in range(len(edits) + 1):
        for subset in combinations(edits, size):
            after, _ = apply_edge_edits(before, subset)
            matrix = [[Q(int(u == v)) - Q(after.rows[u].bit_count() if u == v
                       else -int(bool(after.rows[u] & (1 << v))), degree)
                       for v in range(after.n)] for u in range(after.n)]
            power = [[Q(int(u == v)) for v in range(after.n)] for u in range(after.n)]
            for j in range(steps + 1):
                result[j] += (-1) ** (len(edits) - size) * sum(power[u][u] for u in range(after.n))
                if j < steps:
                    power = [[sum(power[u][w] * matrix[w][v] for w in range(after.n))
                              for v in range(after.n)] for u in range(after.n)]
    return result


class InteractionBoundsTests(unittest.TestCase):
    def test_tour_geometry_and_bridge_reduction(self):
        cuts = [(0, 1, -1), (6, 7, -1), (11, 12, -1)]
        raw = interaction_geometry(EdgeInteraction(path(13), cuts, reduce_bridges=False))
        reduced = interaction_geometry(EdgeInteraction(path(13), cuts))
        self.assertEqual(raw.support_distances, ((0, 5, 10), (5, 0, 4), (10, 4, 0)))
        self.assertEqual(raw.tour_costs, ((19, 2),))
        self.assertEqual(raw.vanishing_order, 22)
        self.assertEqual(reduced.tour_costs, ((20, 1),))
        self.assertEqual(reduced.vanishing_order, 22)
        spider = graph(7, [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6)])
        geometry = interaction_geometry(EdgeInteraction(spider, [(1, 2, -1), (3, 4, -1), (5, 6, -1)]))
        self.assertEqual(geometry.tour_costs, ((6, 2),))
        self.assertEqual(geometry.vanishing_order, 9)

    def test_exact_matrix_moments_and_profiles(self):
        fixtures = [(star(3), [(0, i, -1) for i in range(1, 4)]),
                    (path(6), [(0, 1, -1), (2, 3, -1), (4, 5, -1)]),
                    (cycle(5), [(0, 1, -1), (2, 3, -1), (0, 2, 1)])]
        for before, edits in fixtures:
            value = EdgeInteraction(before, edits, reduce_bridges=False)
            geometry = interaction_geometry(value)
            moments = mixed_returns(before, edits, value.degree_bound, 8)
            for j, moment in enumerate(moments):
                bound = geometry.moment_bound(j)
                self.assertLessEqual(abs(moment), bound)
                profile = sum(a * Q(factorial(j), factorial(j - r)) / value.degree_bound ** r
                              for r, a in enumerate(geometry.moment_profile) if r <= j)
                self.assertLessEqual(bound, profile)
                if j < geometry.vanishing_order:
                    self.assertEqual(moment, 0)
            if before == cycle(5):
                self.assertEqual(geometry.support_distances[0][2], 0)
        scaled = interaction_geometry(EdgeInteraction(star(3), [(0, i, -1) for i in range(1, 4)], normalize=True))
        self.assertEqual(scaled.moment_profile, (0, 0, 0, 2))

    def test_exact_heat_and_tail_enclosures(self):
        value = EdgeInteraction(star(3), [(0, i, -1) for i in range(1, 4)])
        moments = mixed_returns(value.before, value.edits, 3, 5)
        with localcontext() as context:
            context.prec = 70
            time = Q(1, 4)
            z = (-Decimal(1) / 4).exp()
            heat = z * (1 - z) ** 3
            full = interaction_heat_bound(value, time, "1e-20")
            bound = Decimal(full.magnitude_bound.numerator) / Decimal(full.magnitude_bound.denominator)
            self.assertLessEqual(abs(heat), bound)
            tail = interaction_heat_bound(value, time, "1e-20", after_step=5)
            numerator = sum(Q(3, 4) ** j * moment / factorial(j) for j, moment in enumerate(moments))
            retained = (Decimal(-3) / 4).exp() * Decimal(numerator.numerator) / Decimal(numerator.denominator)
            tail_bound = Decimal(tail.magnitude_bound.numerator) / Decimal(tail.magnitude_bound.denominator)
            self.assertLessEqual(abs(heat - retained), tail_bound)
            self.assertLess(tail.magnitude_bound, full.magnitude_bound)
        self.assertLessEqual(tail.bound_enclosure_error, Q("1e-20"))

    def test_separation_certificate_without_local_extraction(self):
        value = EdgeInteraction(path(13), [(0, 1, -1), (6, 7, -1), (11, 12, -1)], reduce_bridges=False)
        value.local = lambda radius: (_ for _ in ()).throw(AssertionError("local extraction forbidden"))
        bound = interaction_heat_bound(value, "1/4")
        self.assertLess(bound.magnitude_bound, Q("1e-18"))
        self.assertLess(bound.magnitude_bound, 8 * Q(1, 4) ** 3)
        self.assertEqual(bound.geometry.vanishing_order, 22)
        self.assertIn("moment_profile", bound.geometry.to_data())

    def test_disconnected_empty_and_budgets(self):
        disconnected = EdgeInteraction(disjoint_union(path(2), path(2)), [(0, 1, -1), (2, 3, -1)])
        geometry = interaction_geometry(disconnected, max_cycles=0)
        self.assertTrue(geometry.identically_zero)
        self.assertEqual(geometry.moment_bound(100), 0)
        self.assertEqual(interaction_heat_bound(geometry, 5).magnitude_bound, 0)
        self.assertEqual(interaction_heat_bound(EdgeInteraction(path(2), []), 5).magnitude_bound, 0)
        value = EdgeInteraction(star(3), [(0, i, -1) for i in range(1, 4)])
        with self.assertRaises(BudgetExceeded):
            interaction_geometry(value, max_cycles=1)
        with self.assertRaises(BudgetExceeded):
            interaction_heat_bound(value, 5, max_steps=0)
        with self.assertRaises(TypeError):
            interaction_geometry(path(2))
        with self.assertRaises(ValueError):
            interaction_heat_bound(value, -1)
        self.assertEqual(interaction_heat_bound(value, 0, max_steps=0).magnitude_bound, 0)

    def test_compositional_wrapper(self):
        source = EdgeInteraction(path(4), [(0, 1, -1), (2, 3, -1)])
        value = GeometricInteraction(source)
        self.assertIs(value.source, source)
        self.assertEqual(value.moment_profile, (0, 0, 0, 0, Q(8, 3)))
        self.assertEqual(value.edit_bound, source.edit_bound)
        self.assertEqual(value.local(2), source.local(2))
        self.assertEqual(value.norm_bound(1, 1), source.norm_bound(1, 1))
        self.assertEqual(value.finite(), source.finite())
        product = value * value
        self.assertEqual(moment_profile(product), (0,) * 8 + (Q(64, 9),))
        for expression in (value, product):
            certified = controlled_heat(expression, "1/4", "1e-9")
            independent = heat_return(expression.finite(), "1/4", "1e-10")
            self.assertLessEqual(certified.interval.lower, independent.interval.upper)
            self.assertGreaterEqual(certified.interval.upper, independent.interval.lower)
        # A profile proved at degree 2 also controls the same moments at degree 5.
        moments = mixed_returns(source.before, source.edits, 5, 7)
        for j, moment in enumerate(moments):
            bound = sum(a * Q(factorial(j), factorial(j-r)) / 5 ** r
                        for r, a in enumerate(value.moment_profile) if r <= j)
            self.assertLessEqual(abs(moment), bound)


if __name__ == "__main__":
    unittest.main()
