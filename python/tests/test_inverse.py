"""Independent power coefficients and inexact-source checks for graph inverses."""
from fractions import Fraction as Q
import unittest

from graphlocal import (BudgetExceeded, Element, Finite, Line, LocalApproximation,
                        LocalHistogram, complete, graph, star)
from graphlocal.inverse import NeumannInverse, _power_majorant


class NoisySource(Element):
    degree_bound, variation_bound, mass, positive = 1, Q(1, 4), Q(1, 4), True

    def approximate(self, radius, k=1, epsilon="1e-6"):
        noise = star(2) if radius else graph(1)
        error = Q(epsilon) / 2
        local = (Finite.from_graph(complete(2), normalize=True) / 4).local(radius)
        return LocalApproximation(local + LocalHistogram(radius, [(noise, error / noise.n ** k)]), k, error)


class InverseTests(unittest.TestCase):
    def test_hypercube_coefficients_and_exact_remainder(self):
        h = Finite.from_graph(complete(2), normalize=True)
        inverse = NeumannInverse(h / 4)
        result = inverse.approximation_certificate(1, 2, "1e-7")
        by_degree = {key.graph.rows[0].bit_count(): c for key, c in result.approximation.histogram.values.items()}
        self.assertEqual(by_degree, {n: Q(1, 4) ** n for n in range(result.truncation_degree + 1)})
        # At radius one a hypercube n-ball has n+1 vertices exactly.
        q, n = Q(1, 4), result.truncation_degree
        total = (1 + q) / (1 - q) ** 3
        exact_tail = total - sum((i + 1) ** 2 * q ** i for i in range(n + 1))
        self.assertLessEqual(exact_tail, result.tail_bound)
        self.assertEqual(inverse.mass, Q(4, 3))
        self.assertIsNone(inverse.degree_bound)

    def test_inverse_equation_with_local_error(self):
        y = (Finite.from_graph(complete(2), normalize=True) / 8
             - Finite.from_graph(complete(3), normalize=True) / 16)
        inverse = NeumannInverse(y)
        approximation = ((1 - y) * inverse).approximate(1, 1, "1e-6")
        difference = approximation.histogram - Finite.scalar(1).local(1)
        self.assertLessEqual(difference.norm(1), approximation.error)
        self.assertLessEqual(approximation.error, Q("1e-6"))

    def test_inexact_source_and_degree_projection(self):
        result = NeumannInverse(NoisySource()).approximation_certificate(1, 1, "1e-6")
        q = Q(1, 4)
        expected = {n: q ** n for n in range(result.truncation_degree + 1)}
        actual = {key.graph.rows[0].bit_count(): c for key, c in result.approximation.histogram.values.items()}
        self.assertEqual(actual, expected)
        self.assertGreater(result.source_error, 0)
        self.assertGreater(result.stability_bound, 0)
        self.assertLessEqual(result.approximation.error, Q("1e-6"))

    def test_scalar_zero_and_contracts(self):
        for scalar in (Q(0), Q(1, 4), Q(-1, 3)):
            value = NeumannInverse(scalar, max_terms=0)
            result = value.approximate(3, 2, "1e-20")
            self.assertEqual(result.histogram, Finite.scalar(1 / (1 - scalar)).local(3))
            self.assertEqual(result.error, 0)
        with self.assertRaises(ValueError):
            NeumannInverse(Line())
        with self.assertRaises(BudgetExceeded):
            NeumannInverse(Finite.from_graph(complete(2), True) / 2, max_terms=0).approximate(1)
        with self.assertRaises(ValueError):
            NeumannInverse(0).approximate(1, epsilon=0)
        invalid = NoisySource()
        invalid.mass = Q(1)
        with self.assertRaises(ValueError):
            NeumannInverse(invalid)
        zero = NoisySource()
        zero.mass, zero.variation_bound = None, Q(0)
        self.assertEqual(NeumannInverse(zero).mass, Q(1))
        self.assertEqual(_power_majorant(Q(1, 4), 1, 0, 1), Q(4, 3))
        self.assertEqual(_power_majorant(Q(1, 4), 1, 0, 1, derivative=True), Q(16, 9))


if __name__ == "__main__":
    unittest.main()
