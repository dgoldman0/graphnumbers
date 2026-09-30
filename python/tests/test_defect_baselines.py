"""Optional numerical comparator checks, independent of rational certificates."""
import importlib.util
import sys
import unittest
from pathlib import Path

from graphlocal import cycle, graph, path


NUMERIC_AVAILABLE = (importlib.util.find_spec("numpy") is not None
                     and importlib.util.find_spec("scipy") is not None)
if NUMERIC_AVAILABLE:
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from defect_baselines import (_krylov_projection, _laplacian, dense_heat,
                                 defect_krylov_heat, sparse_uniformized_heat)


@unittest.skipUnless(NUMERIC_AVAILABLE, "Optional NumPy/SciPy benchmark dependencies")
class DefectBaselineTests(unittest.TestCase):
    def test_sparse_poisson_tail_against_dense(self):
        g = path(13)
        dense, _ = dense_heat(g, 0.5, normalize=False)
        sparse, details = sparse_uniformized_heat(g, 0.5, 7, normalize=False)
        self.assertLessEqual(sparse, dense + 1e-12)
        self.assertLessEqual(dense - sparse, details["truncation_bound"] + 1e-12)

    def test_krylov_cut_cycle_and_normalization(self):
        before, after = cycle(27), path(27)
        estimate, details = defect_krylov_heat(
            before, after, 0.5, 12, [(0, 26, -1)], normalize=False)
        dense_before, _ = dense_heat(before, 0.5, normalize=False)
        dense_after, _ = dense_heat(after, 0.5, normalize=False)
        self.assertAlmostEqual(estimate, dense_after - dense_before, places=11)
        normalized, _ = defect_krylov_heat(before, after, 0.5, 12)
        self.assertAlmostEqual(27 * normalized, estimate, places=13)
        self.assertLess(details["basis_orthogonality_error"], 1e-12)

    def test_mixed_edits_polynomial_trace_exactness(self):
        before = cycle(9)
        edges = [(u, (u + 1) % 9) for u in range(8)] + [(1, 5), (2, 6)]
        after = graph(9, edges)
        a, b, details = _krylov_projection(before, after, 2)
        full_a, full_b = _laplacian(before).toarray(), _laplacian(after).toarray()
        for degree in range(1, 5):
            exact = np.trace(np.linalg.matrix_power(full_b, degree)
                             - np.linalg.matrix_power(full_a, degree))
            reduced = np.trace(np.linalg.matrix_power(b, degree)
                               - np.linalg.matrix_power(a, degree))
            self.assertAlmostEqual(exact, reduced, delta=1e-9)
        self.assertEqual(details["moment_exact_through_degree"], 4)

    def test_invariant_subspace_is_complete_for_edge_addition(self):
        before, after = graph(8), graph(8, [(2, 5)])
        value, details = defect_krylov_heat(before, after, 2, 5, normalize=False)
        self.assertAlmostEqual(value, np.exp(-4) - 1, places=14)
        self.assertEqual(details["subspace_dimension"], 1)
        self.assertTrue(details["numerical_breakdown"])

    def test_identity_and_zero_time(self):
        g = cycle(11)
        value, details = defect_krylov_heat(g, g, 3, 4, normalize=False)
        self.assertEqual(value, 0)
        self.assertEqual(details["subspace_dimension"], 0)
        self.assertEqual(defect_krylov_heat(g, path(11), 0, 3)[0], 0)

    def test_edit_list_and_degree_validation(self):
        with self.assertRaises(ValueError):
            defect_krylov_heat(cycle(8), path(8), 1, 3, [])
        with self.assertRaises(ValueError):
            defect_krylov_heat(cycle(8), path(8), 1, 3, [(0, 7, +1)])
        with self.assertRaises(ValueError):
            sparse_uniformized_heat(cycle(8), 1, 5, degree_bound=1)


if __name__ == "__main__":
    unittest.main()
