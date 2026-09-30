"""Exact independent checks of cut-set and planar moment calculations."""
import sys
import unittest
from fractions import Fraction as Q
from math import factorial
from pathlib import Path

from graphlocal import cycle, star

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from branch_planar_verification import (centered_grid_cuts, edges, interaction_components,
                                        mixed_laplacian_moments, planar_pair_records, quotient_leaf_cuts,
                                        spider, square_grid, subset_graphs, verify_trees)


def full_matrix_trace_moments(g, order):
    laplacian = [[g.rows[u].bit_count() if u == v else -int(bool(g.rows[u] & (1 << v)))
                  for v in range(g.n)] for u in range(g.n)]
    power = [[int(u == v) for v in range(g.n)] for u in range(g.n)]
    result = []
    for _ in range(order + 1):
        result.append(sum(power[u][u] for u in range(g.n)))
        power = [[sum(power[u][w] * laplacian[w][v] for w in range(g.n))
                  for v in range(g.n)] for u in range(g.n)]
    return result


class BranchPlanarTests(unittest.TestCase):
    def test_all_small_tree_cut_sets(self):
        result = verify_trees(5)
        self.assertEqual(result["cut_sets"], 63)
        self.assertGreater(result["proper_internal_cut_reductions"], 0)

    def test_mixed_sparse_recurrence_against_full_integer_matrices(self):
        g = cycle(4)
        cuts = edges(g)
        expected = [0] * 9
        for sign, modified in subset_graphs(g, cuts):
            for j, trace in enumerate(full_matrix_trace_moments(modified, 8)):
                expected[j] += sign * trace
        self.assertEqual(mixed_laplacian_moments(g, cuts), tuple(expected))
        self.assertEqual(expected[:5], [0, 0, 0, 0, 8])

    def test_star_and_triangle_leading_signs(self):
        for k in (2, 3, 4):
            g = star(k)
            moments = mixed_laplacian_moments(g, edges(g), k)
            self.assertEqual(moments[:k], (0,) * k)
            self.assertEqual(Q((-1) ** k * moments[k], factorial(k)), 1)
        g = cycle(3)
        self.assertEqual(mixed_laplacian_moments(g, edges(g), 3), (0, 0, 0, 6))
        with self.assertRaises(ValueError):
            quotient_leaf_cuts(g, edges(g))

    def test_centered_grid_first_five_orders_stabilize(self):
        for kind in ("Y3", "plaquette4"):
            values = [mixed_laplacian_moments(square_grid(side), centered_grid_cuts(side, kind), 5)
                      for side in (7, 9)]
            self.assertEqual(*values)

    def test_branched_spiders_have_delayed_nonzero_interactions(self):
        for lengths, order, value in (((2, 2, 2), 9, -18),
                                     ((3, 2, 1), 9, -18),
                                     ((2, 2, 2, 2), 12, 72)):
            g, cuts = spider(lengths)
            moments = mixed_laplacian_moments(g, cuts, order)
            self.assertEqual(moments[:order], (0,) * order)
            self.assertEqual(moments[order], value)

    def test_planar_pair_table_and_first_cross_moment_formula(self):
        rows = planar_pair_records(sides=(17, 21))
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["buffered_moments_stabilized"] for row in rows))
        leading = [row["finite_square_verifications"][0]["leading_heat_coefficient"]
                   for row in rows]
        self.assertEqual(leading, ["1", "1", "2/3", "1/30", "1/120"])
        perpendicular = rows[0]["finite_square_verifications"][0]
        collinear = rows[1]["finite_square_verifications"][0]
        self.assertEqual(perpendicular["mixed_laplacian_moments"][4]
                         - collinear["mixed_laplacian_moments"][4], 8)


if __name__ == "__main__":
    unittest.main()
