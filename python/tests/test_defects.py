"""Exact local-defect identities and independent relative-heat checks."""
from decimal import Decimal, localcontext
from fractions import Fraction as Q
import itertools
import unittest

from graphlocal import (CutLineDefect, Element, Finite, Line, LocalApproximation,
                        LocalHistogram, SparseEdgeDifference, apply_edge_edits,
                        catalog, complete, cycle, graph, heat_return, path,
                        relative_heat)
from graphlocal.heat import lazy_returns


def decimal(value):
    value = Q(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


class PerturbedDefect(Element):
    def __init__(self, value):
        self.value = value
        self.degree_bound, self.edit_bound, self.mass = value.degree_bound, value.edit_bound, 0

    def approximate(self, radius, k=1, epsilon="1e-8"):
        error = Q(epsilon) / 2
        return LocalApproximation(self.value.local(radius) + Finite.scalar(error).local(radius), k, error)


class DefectTests(unittest.TestCase):
    def test_all_small_single_edge_local_changes(self):
        cases = 0
        for before in catalog(4):
            for u, v in itertools.combinations(range(before.n), 2):
                sign = -1 if before.rows[u] & (1 << v) else 1
                for normalize in (False, True):
                    defect = SparseEdgeDifference(before, [(u, v, sign)], normalize)
                    explicit = defect.finite()
                    self.assertEqual(defect.edit_bound, Q(1, before.n) if normalize else Q(1))
                    for radius in range(3):
                        self.assertEqual(defect.local(radius), explicit.local(radius))
                        cases += 1
        self.assertEqual(cases, 258)

    def test_mixed_edits_and_validation(self):
        source = SparseEdgeDifference(cycle(12), [(0, 11, -1), (2, 7, 1), (4, 5, -1)])
        for radius in range(4):
            self.assertEqual(source.local(radius), source.finite().local(radius))
        self.assertEqual(source.edit_bound, 3)
        for edits in [[(0, 1, 1)], [(0, 2, -1)], [(0, 1, -1), (1, 0, 1)], [(0, 0, 1)]]:
            with self.assertRaises(ValueError):
                apply_edge_edits(cycle(6), edits)

    def test_cut_limit_and_growing_variation(self):
        defect = CutLineDefect()
        self.assertEqual(defect.local(0), LocalHistogram(0))
        for radius in range(1, 8):
            self.assertEqual(defect.local(radius), defect.finite_at_radius(radius).local(radius))
            self.assertEqual(defect.local(radius).norm(0), 4 * radius)
            for k in (1, 2):
                expected = 2 * sum((radius + j + 1) ** k for j in range(radius)) + 2 * radius * (2 * radius + 1) ** k
                self.assertEqual(defect.local(radius).norm(k), expected)
        with self.assertRaises(ValueError):
            heat_return(defect, 1)

    def test_cut_moments_independent_closed_form(self):
        for degree in (2, 3, 4):
            for steps in (1, 4, 9, 14):
                local = CutLineDefect().local((steps + 1) // 2)
                moments = [Q(0)] * (steps + 1)
                for key, coefficient in local.values.items():
                    for j, value in enumerate(lazy_returns(key.graph, degree, steps)):
                        moments[j] += coefficient * value
                self.assertEqual(moments, [(1 - (1 - Q(4, degree)) ** j) / 2 for j in range(steps + 1)])

    def test_relative_heat_closed_form_and_inexact_input(self):
        with localcontext() as ctx:
            ctx.prec = 70
            for t in (Q(1, 10), Q(1, 2), Q(2), Q(5)):
                expected = (1 - (-4 * decimal(t)).exp()) / 2
                for source in (CutLineDefect(), PerturbedDefect(CutLineDefect())):
                    result = relative_heat(source, t, "1e-10")
                    self.assertLessEqual(decimal(result.interval.lower), expected)
                    self.assertGreaterEqual(decimal(result.interval.upper), expected)
                    self.assertLessEqual(result.interval.radius, Q("1e-10"))

    def test_signed_product_and_sum_certificates(self):
        cut, line = CutLineDefect(), Line()
        self.assertEqual((cut * line).edit_bound, 1)
        self.assertEqual((cut * (3 * line - 2)).edit_bound, 5)
        self.assertEqual((3 * cut - cut).edit_bound, 4)
        self.assertIsNone((cut * cut).edit_bound)
        with self.assertRaises(ValueError):
            relative_heat(cut * cut, "1/2")
        with localcontext() as ctx:
            ctx.prec = 70
            t = Decimal("0.5")
            term = series = Decimal(1)
            for j in range(1, 100):
                term *= t * t / (j * j)
                series += term
            line_heat = (-2 * t).exp() * series
            cut_heat = (1 - (-4 * t).exp()) / 2
            for source, expected in [(cut * line, cut_heat * line_heat),
                                     (cut * (3 * line - 2), cut_heat * (3 * line_heat - 2)),
                                     (-2 * cut, -2 * cut_heat)]:
                result = relative_heat(source, "1/2", "1e-8")
                self.assertLessEqual(decimal(result.interval.lower), expected)
                self.assertGreaterEqual(decimal(result.interval.upper), expected)

    def test_finite_normalization_and_tail(self):
        defect = SparseEdgeDifference(cycle(6), [(0, 5, -1)])
        scaled = SparseEdgeDifference(cycle(6), [(0, 5, -1)], normalize=True)
        for t in (Q(1, 2), Q(2)):
            a, b = relative_heat(defect, t), relative_heat(scaled, t)
            conventional = heat_return(defect.finite(), t, "1e-10").interval
            self.assertLessEqual(a.interval.lower, conventional.upper)
            self.assertGreaterEqual(a.interval.upper, conventional.lower)
            self.assertLessEqual(a.interval.lower / 6, b.interval.upper)
            self.assertGreaterEqual(a.interval.upper / 6, b.interval.lower)

    def test_separated_defects_and_interaction(self):
        before = cycle(60)
        first, second = [(0, 59, -1)], [(29, 30, -1)]
        together = SparseEdgeDifference(before, first + second)
        separately = SparseEdgeDifference(before, first) + SparseEdgeDifference(before, second)
        for radius in range(1, 8):
            self.assertEqual(together.local(radius), separately.local(radius))
        nearby = SparseEdgeDifference(before, [(0, 59, -1), (0, 1, -1)])
        single_sum = SparseEdgeDifference(before, [(0, 59, -1)]) + SparseEdgeDifference(before, [(0, 1, -1)])
        self.assertNotEqual(nearby.local(1), single_sum.local(1))

    def test_zero_and_input_contracts(self):
        source = SparseEdgeDifference(complete(12), [])
        self.assertEqual(relative_heat(source, 1000, max_steps=0).interval.lower, 0)
        self.assertEqual(relative_heat(CutLineDefect(), 0, max_steps=0).interval.upper, 0)
        with self.assertRaises(ValueError):
            relative_heat(Line(), 1)
        with self.assertRaises(ValueError):
            relative_heat(CutLineDefect(), -1)
        with self.assertRaises(TypeError):
            relative_heat(CutLineDefect(), 0.5)


if __name__ == "__main__":
    unittest.main()
