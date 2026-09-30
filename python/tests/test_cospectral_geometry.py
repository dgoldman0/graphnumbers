"""Exact checks for the local-geometry versus scalar-spectrum example."""
import sys
import unittest
from pathlib import Path

from graphlocal import complete, path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from cospectral_geometry import count_cliques, diagonal_moments, verify_geometry


class CospectralGeometryTests(unittest.TestCase):
    def test_exact_geometry_certificate(self):
        report = verify_geometry()
        self.assertTrue(report["success"])
        self.assertEqual(report["graphs"]["rook"]["K4_count"], 8)
        self.assertEqual(report["graphs"]["shrikhande"]["K4_count"], 0)
        self.assertEqual(report["normalized_difference"]["radius_one_p_1_1"], "14")
        self.assertEqual(report["cartesian_product_check"]["product_K4_counts"],
                         {"rook": 16, "shrikhande": 0})

    def test_integer_moments_on_irregular_graph(self):
        self.assertEqual(diagonal_moments(path(3), 2),
                         ((1, 1, 1), (0, 0, 0), (1, 2, 1)))
        self.assertEqual(diagonal_moments(path(3), 2, laplacian=True),
                         ((1, 1, 1), (1, 2, 1), (2, 6, 2)))

    def test_clique_counter(self):
        self.assertEqual(count_cliques(complete(5), 4), 5)
        self.assertEqual(count_cliques(path(5), 3), 0)
        self.assertEqual(count_cliques(path(5), 1), 5)

    def test_argument_validation(self):
        with self.assertRaises(ValueError):
            diagonal_moments(path(2), -1)
        with self.assertRaises(ValueError):
            count_cliques(path(2), 0)
        with self.assertRaises(ValueError):
            verify_geometry(True)


if __name__ == "__main__":
    unittest.main()
