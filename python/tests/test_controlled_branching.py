"""Certificates for bridge reductions and products of signed defects."""
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from itertools import combinations
from math import comb
import unittest

from graphlocal import (BudgetExceeded, CutLineDefect, EdgeInteraction, Element,
                        Finite, Line, LocalApproximation, LocalHistogram, PreparedLocal,
                        SparseEdgeDifference, apply_edge_edits, bridge_cut_reduction,
                        controlled_heat, cycle, exp, graph, heat_return,
                        moment_profile, path, relative_heat, star)
from graphlocal.heat import lazy_returns


def decimal(value):
    value = Q(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


def raw_interaction(before, edits):
    terms = []
    for size in range(len(edits) + 1):
        for subset in combinations(edits, size):
            after, _ = apply_edge_edits(before, subset)
            terms.append(((-1) ** (len(edits) - size), after))
    return Finite(terms)


class InexactControlled(Element):
    def __init__(self, value, outside_degree=False):
        self.value = value
        self.outside_degree = outside_degree
        self.degree_bound = value.degree_bound
        self.moment_profile = moment_profile(value)

    def approximate(self, radius, k=1, epsilon="1e-8"):
        error = Q(epsilon) / 2
        if self.outside_degree and radius:
            noise = star(self.degree_bound + 1)
            histogram = LocalHistogram(radius, [(noise, error / noise.n ** k)])
            return LocalApproximation(self.value.local(radius) + histogram, k, error)
        return LocalApproximation(self.value.local(radius) + Finite.scalar(error).local(radius), k, error)


class ControlledBranchTests(unittest.TestCase):
    def test_bridge_reduction_and_work_budget(self):
        before = path(9)
        edits = [(i, i + 1, -1) for i in range(8)]
        value = EdgeInteraction(before, edits, max_edits=2)
        self.assertTrue(value.bridge_reduction_applies)
        self.assertEqual(len(value.active_edits), 2)
        self.assertEqual(value.reduction_sign, 1)
        self.assertEqual(value.edit_bound, 4)
        self.assertEqual(value.finite(), raw_interaction(before, edits))
        with self.assertRaises(BudgetExceeded):
            EdgeInteraction(before, edits, reduce_bridges=False, max_edits=2)
        blocks = graph(12, [(3*i+j, 3*i+(j+1)%3) for i in range(4) for j in range(3)]
                       + [(2, 3), (5, 6), (8, 9)])
        cuts = [(2, 3, -1), (5, 6, -1), (8, 9, -1)]
        value = EdgeInteraction(blocks, cuts)
        self.assertEqual(value.reduction_sign, -1)
        self.assertEqual(len(value.active_edits), 2)
        self.assertEqual(value.finite(), raw_interaction(blocks, cuts))

    def test_cycles_mixed_edits_and_normalization(self):
        for before, edits in [(cycle(3), [(0, 1, -1), (1, 2, -1), (0, 2, -1)]),
                              (cycle(6), [(0, 1, -1), (0, 3, 1), (3, 4, -1)])]:
            value = EdgeInteraction(before, edits)
            self.assertFalse(value.bridge_reduction_applies)
            explicit = raw_interaction(before, edits)
            self.assertEqual(value.finite(), explicit)
            for radius in (0, 1, 2):
                self.assertEqual(value.local(radius), explicit.local(radius))
            scaled = EdgeInteraction(before, edits, normalize=True)
            self.assertEqual(scaled.local(2), value.local(2).scale(Q(1, before.n)))
            self.assertEqual(moment_profile(scaled), (Q(0), Q(0), Q(0), Q(8, before.n)))
        self.assertEqual(EdgeInteraction(cycle(3), []).local(2).norm(0), 0)

    def test_branch_and_cycle_heat_signs(self):
        with localcontext() as ctx:
            ctx.prec = 70
            for leaves in (3, 4):
                value = EdgeInteraction(star(leaves), [(0, v, -1) for v in range(1, leaves + 1)])
                for t in (Q(1, 4), Q(1)):
                    z = (-decimal(t)).exp()
                    expected = z * (1 - z) ** leaves
                    result = controlled_heat(value, t, "1e-10")
                    self.assertLessEqual(decimal(result.interval.lower), expected)
                    self.assertGreaterEqual(decimal(result.interval.upper), expected)
            triangle = EdgeInteraction(cycle(3), [(0, 1, -1), (1, 2, -1), (0, 2, -1)])
            result = controlled_heat(triangle, 1, "1e-10")
            expected = -(1 - Decimal(-1).exp()) ** 3
            self.assertLessEqual(decimal(result.interval.lower), expected)
            self.assertGreaterEqual(decimal(result.interval.upper), expected)
            self.assertLess(result.interval.upper, 0)

    def test_crossing_cut_variation_types_and_moments(self):
        cut = CutLineDefect()
        for power in (2, 3):
            for radius in range(1, 4):
                local = (cut ** power).local(radius)
                self.assertEqual(local.norm(0), (4 * radius) ** power)
                self.assertEqual(len(local.values), comb(radius + power, power))
        crossing = cut * cut
        self.assertIsNone(crossing.edit_bound)
        self.assertIsNone(crossing.variation_bound)
        self.assertEqual(moment_profile(crossing), (0, 0, 4))
        local, steps = crossing.local(3), 6
        moments = [Q(0)] * (steps + 1)
        for key, c in local.values.items():
            for j, m in enumerate(lazy_returns(key.graph, 4, steps)):
                moments[j] += c * m
        self.assertEqual(moments, [0, 0, Q(1, 2), 0, Q(1, 2), 0, Q(1, 2)])
        with self.assertRaises(ValueError):
            relative_heat(crossing, "1/2")

    def test_controlled_products_and_inexact_source(self):
        crossing = CutLineDefect() ** 2
        finite_cut = SparseEdgeDifference(path(2), [(0, 1, -1)])
        with localcontext() as ctx:
            ctx.prec = 70
            for t in (Q(1, 10), Q(1, 2)):
                expected = ((1 - (-4 * decimal(t)).exp()) / 2) ** 2
                for source in (crossing, InexactControlled(crossing), InexactControlled(crossing, True)):
                    result = controlled_heat(source, t, "1e-8")
                    self.assertLessEqual(decimal(result.interval.lower), expected)
                    self.assertGreaterEqual(decimal(result.interval.upper), expected)
                    self.assertLessEqual(result.interval.radius, Q("1e-8"))
            value = finite_cut ** 3
            result = controlled_heat(value, "1/2", "1e-10")
            expected = (1 - Decimal(-1).exp()) ** 3
            self.assertLessEqual(decimal(result.interval.lower), expected)
            self.assertGreaterEqual(decimal(result.interval.upper), expected)
        self.assertEqual(moment_profile(3 * crossing + 2 * CutLineDefect() - Line()), (1, 4, 12))
        self.assertEqual(moment_profile(PreparedLocal(crossing, 2)), (0, 0, 4))

    def test_zero_degree_zero_time_and_contracts(self):
        for scalar in (-3, 0, 2):
            result = controlled_heat(Finite.scalar(scalar), 100, "1e-10")
            self.assertEqual(result.interval.lower, scalar)
            self.assertEqual(result.interval.upper, scalar)
        self.assertEqual(controlled_heat(CutLineDefect() ** 2, 0, max_steps=0).interval.lower, 0)
        self.assertEqual(controlled_heat(Finite.from_graph(cycle(3)), 0).interval.lower, 3)
        with self.assertRaises(ValueError):
            controlled_heat(exp(Line()), 1)
        with self.assertRaises(TypeError):
            controlled_heat(Line(), 0.5)
        with self.assertRaises(BudgetExceeded):
            controlled_heat(Line(), 1, max_steps=0)
        for source in (Line(), Finite.from_graph(cycle(4)), CutLineDefect()):
            result = controlled_heat(source, "1/2", "1e-9")
            other = (relative_heat(source, "1/2", "1e-10") if source.edit_bound is not None
                     else heat_return(source, "1/2", "1e-10"))
            self.assertLessEqual(result.interval.lower, other.interval.upper)
            self.assertGreaterEqual(result.interval.upper, other.interval.lower)


if __name__ == "__main__":
    unittest.main()
