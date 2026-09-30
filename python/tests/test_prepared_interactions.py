"""Independent finite-cut, combinatorial-moment and prepared-query checks."""
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from itertools import combinations
from math import comb, factorial
import unittest

from graphlocal import (BudgetExceeded, CutInteraction, CutLineDefect, Element,
                        LineCutDefect, LocalApproximation, LocalHistogram,
                        PreparedLocal, PreparedRelativeHeat, SparseEdgeDifference,
                        TwoCutLineDefect, connected_cut_interaction, cycle,
                        relative_heat)
from graphlocal.heat import lazy_returns


def decimal(x):
    x = Q(x)
    return Decimal(x.numerator) / Decimal(x.denominator)


def image_sum(ell, time):
    """Independent positive Bessel image series at high decimal precision."""
    t, total = decimal(time), Decimal(0)
    for k in range(1, 50):
        order = 2 * ell * k
        term = t ** order / Decimal(factorial(order))
        value = term
        for j in range(1, 100):
            term *= t * t / (j * (order + j))
            value += term
        total += value
    return 2 * ell * (-2 * t).exp() * total


class CountedCut(Element):
    degree_bound, edit_bound, mass = 2, Q(1), Q(0)

    def __init__(self):
        self.calls = 0

    def local(self, radius):
        self.calls += 1
        return CutLineDefect().local(radius)


class InexactCut(CountedCut):
    def approximate(self, radius, k=1, epsilon="1e-8"):
        from graphlocal import Finite
        error = Q(epsilon) / 2
        return LocalApproximation(self.local(radius) + Finite.scalar(error).local(radius), k, error)


class PreparedInteractionTests(unittest.TestCase):
    def test_prepared_geometry_reuses_source_and_truncates(self):
        source = CountedCut()
        prepared = PreparedLocal(source, 8)
        for r in (8, 4, 2, 4, 0, 8):
            self.assertEqual(prepared.local(r), CutLineDefect().local(r))
        self.assertEqual(source.calls, 1)
        with self.assertRaises(BudgetExceeded):
            prepared.local(9)

    def test_prepared_heat_matches_separate_queries(self):
        source = CountedCut()
        prepared = PreparedRelativeHeat(source, 2)
        for time in (Q(0), Q(1, 16), Q(1, 4), Q(1, 2), Q(1), Q(2)):
            result = prepared.evaluate(time)
            single = relative_heat(CutLineDefect(), time)
            self.assertEqual(result.interval, single.interval)
            self.assertEqual(result.returns, single.returns)
        self.assertEqual(source.calls, 1)
        with self.assertRaises(ValueError):
            prepared.evaluate(3)
        with self.assertRaises(ValueError):
            prepared.evaluate(-1)
        with self.assertRaises(TypeError):
            prepared.evaluate(0.5)
        with self.assertRaises(BudgetExceeded):
            PreparedRelativeHeat(source, 2, max_steps=0)

    def test_prepared_inexact_and_zero(self):
        source = InexactCut()
        prepared = PreparedRelativeHeat(source, 2, "1e-10")
        with localcontext() as ctx:
            ctx.prec = 70
            for t in (Q(1, 16), Q(1, 2), Q(2)):
                expected = (1 - (-4 * decimal(t)).exp()) / 2
                result = prepared.at(t)
                self.assertLessEqual(decimal(result.interval.lower), expected)
                self.assertGreaterEqual(decimal(result.interval.upper), expected)
        self.assertEqual(source.calls, 1)
        zero = PreparedRelativeHeat(LineCutDefect([]), 10000, max_steps=0)
        self.assertEqual(zero.at(9999).interval.lower, 0)
        self.assertEqual(PreparedRelativeHeat(source, 0).at(0).interval.upper, 0)

    def test_two_cut_limits_against_explicit_edits(self):
        for ell in range(1, 8):
            for r in range(5):
                for value in (TwoCutLineDefect(ell), CutInteraction(ell)):
                    self.assertEqual(value.local(r), value.finite_at_radius(r).local(r))

    def test_exact_interaction_support_and_variation(self):
        for ell in range(1, 13):
            value = CutInteraction(ell)
            for r in range(ell // 2 + 1):
                self.assertEqual(value.local(r), LocalHistogram(r))
            self.assertGreater(value.local(ell // 2 + 1).norm(0), 0)
            for r in (ell, ell + 1):
                self.assertEqual(value.local(r).norm(0), 4 * r)

    def test_interaction_moments_from_binomial_walks(self):
        for ell in range(1, 7):
            steps = 2 * ell + 4
            local = CutInteraction(ell).local((steps + 1) // 2)
            actual = [Q(0)] * (steps + 1)
            for key, coefficient in local.values.items():
                for j, moment in enumerate(lazy_returns(key.graph, 2, steps)):
                    actual[j] += coefficient * moment
            expected = [Q(0) if j % 2 else Q(2 * ell, 2 ** j) * sum(
                comb(j, j // 2 + ell * k) for k in range(1, j // (2 * ell) + 1))
                        for j in range(steps + 1)]
            self.assertEqual(actual, expected)
            self.assertTrue(all(x == 0 for x in actual[:2 * ell]))
            self.assertEqual(actual[2 * ell], Q(2 * ell, 2 ** (2 * ell)))

    def test_positive_heat_interactions_from_image_sum(self):
        with localcontext() as ctx:
            ctx.prec = 80
            for ell in (1, 2, 4, 8):
                for t in (Q(1, 4), Q(1), Q(3)):
                    expected = image_sum(ell, t)
                    result = relative_heat(CutInteraction(ell), t, "1e-10")
                    self.assertLess(expected, Decimal("0.5"))
                    self.assertGreater(expected, 0)
                    self.assertLessEqual(decimal(result.interval.lower), expected)
                    self.assertGreaterEqual(decimal(result.interval.upper), expected)

    def test_many_cuts_and_connected_reduction(self):
        for positions in ((0,), (0, 1), (0, 4), (0, 1, 3), (0, 2, 5, 7), (0, 1, 2, 3, 4)):
            value = LineCutDefect(positions)
            n = positions[-1] + 15
            before = cycle(n)
            edges = [(0, n - 1, -1)] + [(p - 1, p, -1) for p in positions[1:]]
            for r in range(5):
                self.assertEqual(value.local(r), SparseEdgeDifference(before, edges).local(r))
                total = LocalHistogram(r)
                for size in range(1, len(edges) + 1):
                    for subset in combinations(edges, size):
                        total = total + SparseEdgeDifference(before, subset).local(r).scale(
                            (-1) ** (len(edges) - size))
                self.assertEqual(total, connected_cut_interaction(positions).local(r))
        self.assertEqual(LineCutDefect([7, 0, 5]).positions, (0, 5, 7))
        self.assertEqual(LineCutDefect([-7, 0, -2]).positions, (0, 5, 7))
        self.assertEqual(LineCutDefect([]).local(3), LocalHistogram(3))
        with self.assertRaises(ValueError):
            LineCutDefect([1, 1])
        with self.assertRaises(TypeError):
            LineCutDefect([True])
        with self.assertRaises(ValueError):
            connected_cut_interaction([])


if __name__ == "__main__":
    unittest.main()
