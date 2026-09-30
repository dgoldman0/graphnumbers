"""Exact checks for square/sphere coordinates and cut-line power witnesses."""
from fractions import Fraction as Q
from pathlib import Path
import sys
import unittest

from graphlocal import complete, cycle, path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from unbounded_inverse_verification import (
    additive_coordinates, exponential_tail_majorant, monomial_coefficient,
    path_atoms, series_log, sphere_series, square_count, verify_unbounded_inverse)


class UnboundedInverseVerificationTests(unittest.TestCase):
    def test_complete_exact_verifier(self):
        report = verify_unbounded_inverse()
        self.assertTrue(report["success"])
        self.assertEqual(report["rooted_cartesian_pairs"], 882)
        self.assertEqual(len(report["phase_witnesses"]), 32)
        self.assertEqual([row["variation"] for row in report["monomial_fixtures"]],
                         ["1", "8", "64", "512", "1", "12", "144"])

    def test_square_count_includes_chorded_cycles(self):
        self.assertEqual(square_count(cycle(4)), 1)
        self.assertEqual(square_count(complete(4)), 3)
        self.assertEqual(square_count(path(4)), 0)

    def test_path_coordinate_basis_and_formal_log(self):
        for radius in (2, 3):
            for j, atom in enumerate(path_atoms(radius)):
                self.assertEqual(additive_coordinates(atom, radius),
                                 tuple(Q(int(i == j)) for i in range(radius + 1)))
        self.assertEqual(series_log((Q(1), Q(1), Q(0), Q(0))),
                         (Q(0), Q(1), Q(-1, 2), Q(1, 3)))
        self.assertEqual(sphere_series(complete(1), 3), (1, 0, 0, 0))
        self.assertEqual(monomial_coefficient((1, 1, 1), 2), -96)
        with self.assertRaises(ValueError):
            additive_coordinates(path(3), 1)

    def test_rational_tail_contract(self):
        self.assertEqual(exponential_tail_majorant(0, 2, 1, 0), 0)
        self.assertEqual(exponential_tail_majorant(Q(-1, 4), 2, 1, 8),
                         exponential_tail_majorant(Q(1, 4), 2, 1, 8))
        with self.assertRaises(ValueError):
            exponential_tail_majorant(1, 2, 1, 0)


if __name__ == "__main__":
    unittest.main()
