"""Independent coefficients, exact variation, and cut-line exponential tails."""
from fractions import Fraction as Q
from math import factorial
import unittest

from graphlocal import BudgetExceeded, CutLineDefect, Finite, exp
from graphlocal.defect_exponential import (
    CutLineExponential, _poisson_polynomial, _weighted_tail)


def factor_count(g):
    neighbors = list(g.neighbors(0))
    sets = {u: set(g.neighbors(u)) - {0} for u in neighbors}
    squares = sum(len(sets[u] & sets[v])
                  for i, u in enumerate(neighbors) for v in neighbors[i + 1:])
    degree = len(neighbors)
    return (3 * degree - degree ** 2 + 2 * squares) // 2


class DefectExponentialTests(unittest.TestCase):
    def test_radius_one_closed_form_coefficients(self):
        t = Q(1, 10)
        certificate = CutLineExponential(t).approximation_certificate(1, 1, "1e-8")
        histogram = certificate.approximation.histogram
        actual = {key.graph.rows[0].bit_count(): c for key, c in histogram.values.items()}
        for degree in range(min(6, certificate.degree) + 1):
            # exp(2*t*z - 2*t*z**2), using its independent factorization.
            expected = sum(((2 * t) ** (degree - 2 * b) * (-2 * t) ** b
                            / (factorial(degree - 2 * b) * factorial(b))
                            for b in range(degree // 2 + 1)), Q(0))
            self.assertEqual(actual.get(degree, Q(0)), expected)
        self.assertEqual(histogram.mass, 1)
        self.assertLessEqual(certificate.tail_bound, Q("1e-8"))

    def test_radius_two_separates_powers_and_exact_variation(self):
        t = Q(-1, 1000)
        certificate = CutLineExponential(t).approximation_certificate(2, 1, "1e-6")
        variation_by_power = {}
        for key, coefficient in certificate.approximation.histogram.values.items():
            n = factor_count(key.graph)
            self.assertGreaterEqual(n, 0)
            variation_by_power[n] = variation_by_power.get(n, Q(0)) + abs(coefficient)
        parameter = 8 * abs(t)
        expected = {n: parameter ** n / factorial(n) for n in range(certificate.degree + 1)}
        self.assertEqual(variation_by_power, expected)
        self.assertEqual(certificate.approximation.histogram.norm(0), sum(expected.values()))
        self.assertEqual(certificate.poisson_parameter, parameter)
        self.assertGreater(certificate.unweighted_tail_bound, 0)

    def test_inverse_equation_and_generic_exponential(self):
        value = CutLineExponential(Q(1, 1000))
        inverse = value.inverse()
        self.assertEqual(inverse.parameter, -value.parameter)
        left = value.approximate(2, 1, "1e-5")
        right = inverse.approximate(2, 1, "1e-5")
        product = left.multiply(right)
        difference = product.histogram - Finite.scalar(1).local(2)
        self.assertLessEqual(difference.norm(1), product.error)
        generic = exp(CutLineDefect() * value.parameter).approximate(1, 2, "1e-7")
        specialized = value.approximate(1, 2, "1e-7")
        self.assertLessEqual((generic.histogram - specialized.histogram).norm(2),
                             generic.error + specialized.error)

    def test_polynomial_majorants_and_tail_ratio(self):
        x = Q(2, 5)
        self.assertEqual(_poisson_polynomial(x, 0), 1)
        self.assertEqual(_poisson_polynomial(x, 1), 1 + 2 * x)
        self.assertEqual(_poisson_polynomial(x, 2), 1 + 8 * x + 4 * x ** 2)
        value = CutLineExponential(Q(1, 1000))
        approximation = value.approximate(2, 2, "1e-5")
        self.assertLessEqual(approximation.histogram.norm(2), value.norm_bound(2, 2))
        n, exponent = 3, 4
        next_term = x ** (n + 1) / factorial(n + 1)
        bound, ratio = _weighted_tail(x, n, exponent, next_term)
        partial_tail = sum(((1 + 2 * j) ** exponent * x ** j / factorial(j)
                            for j in range(n + 1, 35)), Q(0))
        self.assertLessEqual(partial_tail, bound)
        self.assertLess(ratio, 1)
        self.assertIsNone(_weighted_tail(Q(100), 0, 1, Q(100))[0])

    def test_scalar_metadata_zero_radius_and_budgets(self):
        zero = CutLineExponential(0, max_terms=0)
        result = zero.approximation_certificate(4, 3, "1e-30")
        self.assertEqual(result.approximation.histogram, Finite.scalar(1).local(4))
        self.assertEqual(result.tail_bound, 0)
        self.assertEqual(zero.degree_bound, 0)
        self.assertEqual(zero.variation_bound, 1)
        self.assertTrue(zero.positive)
        value = CutLineExponential(Q(1, 2))
        self.assertIsNone(value.degree_bound)
        self.assertIsNone(value.variation_bound)
        self.assertFalse(value.positive)
        self.assertEqual(value.mass, 1)
        self.assertEqual(value.approximate(0).histogram, Finite.scalar(1).local(0))
        self.assertEqual(value.norm_bound(0, 7), 1)
        self.assertEqual(result.to_data(True)["histogram"], result.approximation.histogram.to_data())
        with self.assertRaises(BudgetExceeded):
            CutLineExponential(1, max_terms=0).approximate(1)
        with self.assertRaises(BudgetExceeded):
            CutLineExponential(1, max_vertices=2).approximate(1)
        with self.assertRaises(ValueError):
            value.approximate(1, epsilon=0)
        with self.assertRaises(TypeError):
            CutLineExponential(0.1)


if __name__ == "__main__":
    unittest.main()
