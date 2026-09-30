"""Residual inversion checks in a single weighted local Banach algebra."""
from fractions import Fraction as Q
import unittest

from graphlocal import (BudgetExceeded, CutLineDefect, CutLineExponential, Element, Finite,
                        LocalApproximation, LocalHistogram, complete, graph, star)
from graphlocal.local_inverse import (LocalInverseCertificate, UncertifiedLocalInverse,
                                     local_inverse_certificate, refine_local_inverse)


def unit(radius=1):
    return LocalHistogram(radius, [(graph(1), Q(1))])


def by_degree(histogram):
    return {key.graph.rows[0].bit_count(): c for key, c in histogram.values.items()}


class NoisyLocalSource(Element):
    """Exact A=1-H/4 with a certified finite high-degree perturbation."""
    def __init__(self):
        self.value = 1 - Finite.from_graph(complete(2), normalize=True) / 4
        self.requests = []

    def approximate(self, radius, k=1, epsilon="1e-8"):
        self.requests.append((radius, k, Q(epsilon)))
        error = Q(epsilon) / 2
        noise = star(3) if radius else graph(1)
        histogram = LocalHistogram(radius, [(noise, error / noise.n ** k)])
        return LocalApproximation(self.value.local(radius) + histogram, k, error)


class LocalInverseTests(unittest.TestCase):
    def test_scalar_candidates_and_exact_zero_residual(self):
        for scalar, candidate in ((Q(2), Q(1, 3)), (Q(-3), Q(-1, 4))):
            local_candidate = unit(2).scale(candidate)
            certificate = local_inverse_certificate(scalar, local_candidate, k=2)
            expected_error = abs(1 / scalar - candidate)
            self.assertEqual(certificate.approximation.error, expected_error)
            self.assertEqual(certificate.inverse_norm_bound, abs(1 / scalar))
            refined = refine_local_inverse(scalar, local_candidate, k=2, epsilon="1e-12")
            actual_error = abs(refined.approximation.histogram.mass - 1 / scalar)
            self.assertEqual(actual_error, refined.tail_bound)
            self.assertEqual(refined.stability_bound, 0)
            self.assertLessEqual(refined.approximation.error, Q("1e-12"))
        exact = refine_local_inverse(2, unit(1).scale(Q(1, 2)), max_terms=0)
        self.assertEqual(exact.approximation.error, 0)
        self.assertEqual(exact.truncation_degree, 0)
        self.assertEqual(exact.residual_norm, 0)
        self.assertEqual(exact.to_data(include_histogram=True)["histogram"],
                         exact.approximation.histogram.to_data())

    def test_hypercube_inverse_coefficients_and_exact_tail(self):
        h = Finite.from_graph(complete(2), normalize=True)
        source = 1 - h / 4
        certificate = local_inverse_certificate(source, unit(1))
        self.assertEqual(certificate.residual_norm, Q(1, 2))
        self.assertEqual(certificate.residual_bound, Q(1, 2))
        self.assertEqual(certificate.inverse_norm_bound, 2)
        refined = refine_local_inverse(source, unit(1), epsilon="1e-9")
        q, n = Q(1, 4), refined.truncation_degree
        self.assertEqual(by_degree(refined.approximation.histogram), {j: q ** j for j in range(n + 1)})
        exact_tail = 1 / (1 - q) ** 2 - sum((j + 1) * q ** j for j in range(n + 1))
        self.assertLessEqual(exact_tail, refined.approximation.error)
        self.assertLessEqual(refined.approximation.error, Q("1e-9"))
        residual = unit(1) - source.local(1).multiply(refined.approximation.histogram)
        self.assertLessEqual(residual.norm(1), source.local(1).norm(1) * refined.approximation.error)

    def test_better_candidate_passes_a_stronger_weight(self):
        h = Finite.from_graph(complete(2), normalize=True)
        source = 1 - h / 4
        with self.assertRaises(UncertifiedLocalInverse):
            local_inverse_certificate(source, unit(1), k=2)
        candidate = (1 + h / 4).local(1)
        certificate = local_inverse_certificate(source, candidate, k=2)
        self.assertEqual(certificate.candidate_norm, 2)
        self.assertEqual(certificate.residual_norm, Q(9, 16))
        refined = refine_local_inverse(source, candidate, k=2, epsilon="1e-8")
        last = 2 * refined.truncation_degree + 1
        q = Q(1, 4)
        self.assertEqual(by_degree(refined.approximation.histogram), {j: q ** j for j in range(last + 1)})
        exact_tail = (1 + q) / (1 - q) ** 3 - sum((j + 1) ** 2 * q ** j for j in range(last + 1))
        self.assertLessEqual(exact_tail, refined.approximation.error)
        self.assertLessEqual(refined.approximation.error, Q("1e-8"))

    def test_inexact_source_stability_against_full_known_inverse(self):
        source = NoisyLocalSource()
        result = refine_local_inverse(source, unit(1), epsilon="1e-8", source_epsilon="1e-4")
        self.assertEqual(len(source.requests), 2)
        self.assertLess(source.requests[1][2], source.requests[0][2])
        self.assertGreater(result.source_error, 0)
        self.assertGreater(result.stability_bound, 0)
        actual = by_degree(result.approximation.histogram)
        last, q = max(actual), Q(1, 4)
        finite_error = sum((j + 1) * abs(actual.get(j, 0) - q ** j) for j in range(last + 1))
        infinite_tail = 1 / (1 - q) ** 2 - sum((j + 1) * q ** j for j in range(last + 1))
        self.assertLessEqual(finite_error + infinite_tail, result.approximation.error)
        self.assertLessEqual(result.approximation.error, Q("1e-8"))
        self.assertEqual(result.approximation.error, result.stability_bound + result.tail_bound)
        self.assertLess(result.residual_bound, 1)
        self.assertLessEqual(result.residual_bound, (1 + result.initial_residual_bound) / 2)

    def test_radius_one_certificate_has_only_a_local_claim(self):
        # The nonunit result for 1-tE is proved separately in the research
        # notes. These computations certify only the radius-one marginal.
        source = 1 - CutLineDefect() / 16
        certificate = local_inverse_certificate(source, unit(1))
        self.assertEqual(certificate.residual_norm, Q(5, 8))
        self.assertEqual(certificate.inverse_norm_bound, Q(8, 3))
        self.assertIsInstance(certificate, LocalInverseCertificate)
        self.assertNotIsInstance(certificate, Element)
        self.assertIn("specified weighted local marginal only", certificate.to_data()["scope"])
        refined = refine_local_inverse(source, unit(1), epsilon="1e-6")
        coefficients = by_degree(refined.approximation.histogram)
        # Independent generating-function recurrence for 1/(1-z/8+z^2/8).
        expected = [Q(1), Q(1, 8)]
        for degree in range(2, refined.truncation_degree + 1):
            expected.append((expected[-1] - expected[-2]) / 8)
        for degree, coefficient in enumerate(expected):
            self.assertEqual(coefficients.get(degree, Q(0)), coefficient)
        self.assertLessEqual(refined.approximation.error, Q("1e-6"))

    def test_signed_exponential_unit_without_global_variation_bound(self):
        source = CutLineExponential(Q(1, 65536))
        self.assertIsNone(source.variation_bound)
        self.assertIsNone(source.degree_bound)
        for radius in (1, 2):
            refined = refine_local_inverse(source, unit(radius), epsilon="1e-6")
            specialized = source.inverse().approximate(radius, 1, "1e-9")
            difference = refined.approximation.histogram - specialized.histogram
            self.assertLessEqual(difference.norm(1), refined.approximation.error + specialized.error)
            self.assertLessEqual(refined.approximation.error, Q("1e-6"))

    def test_failed_test_budgets_and_invalid_source_contracts(self):
        with self.assertRaisesRegex(UncertifiedLocalInverse, "does not prove"):
            local_inverse_certificate(2, unit(1))
        with self.assertRaises(UncertifiedLocalInverse):
            local_inverse_certificate(1, LocalHistogram(1))
        h = Finite.from_graph(complete(2), normalize=True)
        with self.assertRaises(BudgetExceeded):
            refine_local_inverse(1 - h / 4, unit(1), epsilon="1e-12", max_terms=0)
        with self.assertRaises(BudgetExceeded):
            refine_local_inverse(1 - h / 4, unit(1), epsilon="1e-5", max_vertices=2)
        with self.assertRaises(ValueError):
            local_inverse_certificate(1, unit(1), source_epsilon=0)
        with self.assertRaises(TypeError):
            refine_local_inverse(1, unit(1), epsilon=0.5)
        with self.assertRaises(TypeError):
            local_inverse_certificate(1, Finite.scalar(1))

        class InvalidSource(Element):
            def __init__(self, mode):
                self.mode = mode

            def approximate(self, radius, k=1, epsilon="1e-8"):
                if self.mode == "type":
                    return unit(radius)
                if self.mode == "radius":
                    return LocalApproximation(unit(radius + 1), k)
                if self.mode == "weight":
                    return LocalApproximation(unit(radius), 1)
                return LocalApproximation(unit(radius), k, Q(epsilon) * 2)

        for mode in ("type", "radius", "weight", "error"):
            with self.assertRaises((TypeError, ValueError)):
                local_inverse_certificate(InvalidSource(mode), unit(1), k=2)


if __name__ == "__main__":
    unittest.main()
