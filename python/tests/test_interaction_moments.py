"""Independent full-matrix checks of the covering cyclic-word algorithm."""
from fractions import Fraction as Q
from itertools import combinations
from math import comb
import unittest

from graphlocal import (BudgetExceeded, EdgeInteraction, apply_edge_edits,
                        complete, cycle, graph, path, star)
from graphlocal.interaction_moments import interaction_moments, tree_interaction_leading


def matrix_moments(g, order):
    n = g.n
    lap = [[(g.rows[i].bit_count() if i == j else
             -int(bool(g.rows[i] & (1 << j)))) for j in range(n)] for i in range(n)]
    power = [[int(i == j) for j in range(n)] for i in range(n)]
    result = [n]
    for _ in range(order):
        power = [[sum(power[i][a] * lap[a][j] for a in range(n))
                  for j in range(n)] for i in range(n)]
        result.append(sum(power[i][i] for i in range(n)))
    return tuple(result)


def subset_moments(before, edits, order):
    result = [0] * (order + 1)
    for size in range(len(edits) + 1):
        for subset in combinations(edits, size):
            after, _ = apply_edge_edits(before, subset)
            for j, moment in enumerate(matrix_moments(after, order)):
                result[j] += (-1) ** (len(edits) - size) * moment
    return tuple(result)


class InteractionMomentTests(unittest.TestCase):
    def test_exact_moments_and_uniformization(self):
        fixtures = [(star(4), [(0, i, -1) for i in range(1, 5)]),
                    (cycle(4), [(0, 1, -1), (1, 2, -1), (2, 3, -1)]),
                    (complete(4), [(0, 1, -1), (1, 2, -1), (2, 3, -1)]),
                    (cycle(5), [(0, 1, -1), (0, 2, 1), (2, 3, -1)])]
        for before, edits in fixtures:
            value = EdgeInteraction(before, edits, reduce_bridges=False)
            data = interaction_moments(value, 8)
            expected = subset_moments(before, edits, 8)
            self.assertEqual(data.laplacian, expected)
            d = value.degree_bound
            returns = tuple(sum(comb(j, a) * Q(-1, d) ** a * expected[a]
                                for a in range(j + 1)) for j in range(9))
            self.assertEqual(data.returns, returns)

    def test_repeated_defect_is_essential(self):
        value = EdgeInteraction(complete(4), [(0, 1, -1), (1, 2, -1), (2, 3, -1)])
        data = interaction_moments(value, 8)
        self.assertTrue(all(matrix[0][2] == 0 for matrix in data.cross_moments))
        self.assertEqual(data.first_nonzero_order, 4)
        self.assertEqual(data.laplacian[4], 4)
        self.assertEqual(data.to_data()["leading_heat_coefficient"], "1/6")

    def test_bridge_reduction_and_normalization(self):
        before, edits = path(6), [(i, i + 1, -1) for i in range(5)]
        raw = EdgeInteraction(before, edits, reduce_bridges=False)
        reduced = EdgeInteraction(before, edits)
        a, b = interaction_moments(raw, 10), interaction_moments(reduced, 10)
        self.assertEqual(a.laplacian, b.laplacian)
        normalized = interaction_moments(EdgeInteraction(before, edits, normalize=True), 10)
        self.assertEqual(normalized.laplacian, tuple(x / 6 for x in b.laplacian))

    def test_zero_prefix_and_budgets(self):
        value = EdgeInteraction(star(4), [(0, i, -1) for i in range(1, 5)])
        self.assertIsNone(interaction_moments(value, 3).first_nonzero_order)
        self.assertEqual(interaction_moments(value, 0).laplacian, (0,))
        self.assertEqual(interaction_moments(EdgeInteraction(graph(0), []), 4).laplacian, (0,) * 5)
        with self.assertRaises(BudgetExceeded):
            interaction_moments(value, 8, max_work=1)
        with self.assertRaises(TypeError):
            interaction_moments(star(4), 8)

    def test_tree_geometry_predicts_leading_moment(self):
        # Binary quartet with unselected exterior branches on both sides.
        before = graph(9, [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5),
                           (0, 6), (6, 7), (5, 8)])
        fixtures = [(before, [(0, 1, -1), (0, 2, -1), (3, 4, -1), (3, 5, -1)]),
                    (path(6), [(i, i + 1, -1) for i in range(5)]),
                    (star(4), [(0, i, -1) for i in range(1, 5)]),
                    (path(3), [(0, 1, -1)])]
        for g, edits in fixtures:
            value = EdgeInteraction(g, edits)
            leading = tree_interaction_leading(value)
            data = interaction_moments(value, leading.order)
            self.assertEqual(data.first_nonzero_order, leading.order)
            self.assertEqual(data.to_data()["leading_heat_coefficient"], str(leading.heat_coefficient))
        quartet = tree_interaction_leading(EdgeInteraction(*fixtures[0]))
        self.assertEqual((quartet.spanning_edges, quartet.terminal_edges, quartet.branching_factor), (5, 4, 4))
        self.assertEqual(quartet.heat_coefficient, Q(1, 30))
        with self.assertRaises(ValueError):
            tree_interaction_leading(EdgeInteraction(cycle(3), [(0, 1, -1)]))


if __name__ == "__main__":
    unittest.main()
