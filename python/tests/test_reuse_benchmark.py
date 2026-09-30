"""Independent checks of the optional reusable numerical/motif comparators."""
import importlib.util
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path

from graphlocal import SparseEdgeDifference, complete, cycle
from graphlocal.graphs import induced

AVAILABLE = (importlib.util.find_spec("numpy") is not None
             and importlib.util.find_spec("scipy") is not None)
if AVAILABLE:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from defect_baselines import dense_heat
    from reuse_benchmark import (OBSERVABLES, clique_chain, evaluate_observable,
                                 extract_terms, krylov_query, prepare_krylov,
                                 root_four_cliques, root_triangles)


@unittest.skipUnless(AVAILABLE, "Optional NumPy/SciPy benchmark dependencies")
class ReuseBenchmarkTests(unittest.TestCase):
    def test_complete_graph_motif_normalization(self):
        g = complete(5)
        rooted = [induced(g, [u] + [v for v in range(5) if v != u]) for u in range(5)]
        self.assertEqual(sum(Q(root_triangles(b), 3) for b in rooted), 10)
        self.assertEqual(sum(Q(root_four_cliques(b), 4) for b in rooted), 5)

    def test_affected_root_motifs_capture_clique_deletion(self):
        defect = SparseEdgeDifference(clique_chain(16), [(0, 1, -1), (0, 5, +1)])
        functions = dict(OBSERVABLES)
        self.assertEqual(evaluate_observable(extract_terms(defect, 2), functions["four_cliques"]), -1)
        self.assertEqual(evaluate_observable(extract_terms(defect, 2), functions["triangles"]), -2)

    def test_reused_krylov_projection_at_distinct_times(self):
        defect = SparseEdgeDifference(cycle(19), [(0, 18, -1)])
        prepared = prepare_krylov(defect, 12)
        for t in (0, Q(1, 16), Q(1, 2), 2):
            before, _ = dense_heat(defect.before, t, normalize=False)
            after, _ = dense_heat(defect.after, t, normalize=False)
            self.assertAlmostEqual(krylov_query(prepared, t), after - before, delta=1e-11)


if __name__ == "__main__":
    unittest.main()
