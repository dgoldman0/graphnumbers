"""Independent low-radius comparisons of branching and planar defects."""
from pathlib import Path
import sys
import unittest

from graphlocal.graphs import isomorphic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from defect_breadth_verification import (
    laplacian_traces_full_matrix, lattice_cut, planar_coordinate_witness,
    regular_face, regular_tree, root_edge_graph, tree_cut,
    verify_defect_breadth)
from graphlocal import complete, cycle, disjoint_union, path


class DefectBreadthVerificationTests(unittest.TestCase):
    def test_exact_report(self):
        report = verify_defect_breadth()
        self.assertTrue(report["success"])
        self.assertEqual(report["tree_coordinate_product_fixtures"], 20)
        for family, values in (("B3", ["4", "12", "28", "60"]),
                               ("B4", ["4", "16", "52", "160"]),
                               ("P", ["4", "16", "36", "64"])):
            self.assertEqual([row["variation"] for row in report["marginals"]
                              if row["family"] == family], values)
        self.assertEqual(report["planar_coordinate_obstruction"]["equal_coordinates"], [0, 0, 0, 2])
        self.assertEqual(report["regular_face_root_edge_pairs"], 441)
        self.assertEqual(report["full_matrix_relative_laplacian_moments"],
                         {"B4": [0, -2, -16, -116, -832], "P": [0, -2, -16, -116, -848]})

    def test_degree_four_backgrounds_agree_only_at_first_radius(self):
        self.assertEqual(tree_cut(4, 1).local(1), lattice_cut(1)[0].local(1))
        self.assertNotEqual(tree_cut(4, 2).local(2), lattice_cut(2)[0].local(2))

    def test_planar_same_spheres_do_not_imply_rooted_isomorphism(self):
        _, left, right, baseline = planar_coordinate_witness()
        self.assertFalse(isomorphic(left, right, rooted=True))
        self.assertEqual((left.n, right.n, baseline.n), (25, 25, 25))
        self.assertEqual((left.edges, right.edges, baseline.edges), (35, 35, 36))

    def test_tree_constructor_validation(self):
        self.assertEqual(regular_tree(3, 2).n, 14)
        self.assertEqual(regular_tree(4, 1).max_degree, 4)
        with self.assertRaises(ValueError):
            regular_tree(1, 3)
        with self.assertRaises(ValueError):
            tree_cut(3, 0)

    def test_regular_face_and_root_edge_graph_on_small_fixtures(self):
        self.assertTrue(regular_face(cycle(4), 2))
        self.assertFalse(regular_face(path(3), 2))
        self.assertTrue(isomorphic(root_edge_graph(cycle(4)), disjoint_union(complete(1), complete(1))))
        self.assertTrue(isomorphic(root_edge_graph(tree_cut(3, 2).before), complete(3)))

    def test_full_matrix_laplacian_reference(self):
        self.assertEqual(laplacian_traces_full_matrix(complete(2), 4), (2, 2, 4, 8, 16))
        self.assertEqual(laplacian_traces_full_matrix(path(3), 4), (3, 4, 10, 28, 82))


if __name__ == "__main__":
    unittest.main()
