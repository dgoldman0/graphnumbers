"""Finite independent checks of arithmetic and nonspectral geometry fixtures."""
from fractions import Fraction as Q
from pathlib import Path
import sys
import unittest

from graphlocal import cartesian, complete, path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from arithmetic_geometry_verification import (
    correlation_graphs, degree_specialization, divide_degree_kernel,
    inverse_coefficients, joint_convolution, joint_degree_triangles,
    poly_add, poly_multiply, verify_arithmetic_geometry)


class ArithmeticGeometryVerificationTests(unittest.TestCase):
    def test_exact_verification_report(self):
        report = verify_arithmetic_geometry(inverse_order=5)
        self.assertTrue(report["success"])
        self.assertEqual(report["root_link_product_vertex_pairs"], 441)
        self.assertEqual(report["joint_convolution_graph_pairs"], 49)
        self.assertEqual(report["finite_link_cone_realizations"], 16)
        self.assertEqual(report["joint_correlation"]["raw_mixed_moments"], [6, 7])
        self.assertEqual([row["family_character"] for row in
                          report["cospectral_family"]["endpoint_nonunit_witnesses"]], ["0", "0"])

    def test_joint_correlation_is_preserved_by_product(self):
        g, h = correlation_graphs()
        factor = complete(3)
        for value in (g, h):
            self.assertEqual(joint_degree_triangles(cartesian(value, factor)),
                             joint_convolution(joint_degree_triangles(value),
                                               joint_degree_triangles(factor)))
        self.assertNotEqual(joint_degree_triangles(g), joint_degree_triangles(h))

    def test_kernel_division_with_nonzero_remainder(self):
        kernel = {(0, 1): Q(1), (2, 0): Q(-1)}
        polynomial = {(1, 3): Q(2), (0, 1): Q(-3), (7, 0): Q(1), (0, 0): Q(5)}
        quotient, remainder = divide_degree_kernel(polynomial)
        self.assertEqual(poly_add(poly_multiply(kernel, quotient), remainder), polynomial)
        self.assertEqual(degree_specialization(polynomial), {7: Q(3), 2: Q(-3), 0: Q(5)})
        self.assertEqual(remainder, {(7, 0): Q(3), (2, 0): Q(-3), (0, 0): Q(5)})

    def test_inverse_coefficients_and_exact_total_variation(self):
        value = inverse_coefficients(Q(1, 4), 2)
        self.assertEqual(value, {(0, 0): Q(1), (1, 0): Q(-1, 4), (0, 1): Q(1, 4),
                                 (2, 0): Q(1, 16), (1, 1): Q(-1, 8), (0, 2): Q(1, 16)})
        self.assertEqual(sum(map(abs, value.values())), Q(7, 4))
        self.assertEqual(inverse_coefficients(0, 3), {(0, 0): Q(1)})
        for invalid in (-1, True):
            with self.assertRaises(ValueError):
                inverse_coefficients(Q(1, 4), invalid)


if __name__ == "__main__":
    unittest.main()
