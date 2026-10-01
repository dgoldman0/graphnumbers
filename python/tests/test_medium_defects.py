"""Independent buffered witnesses for tree and lattice edge-cut limits."""
from fractions import Fraction as Q
from itertools import product
import unittest

from graphlocal import (BudgetExceeded, CutLineDefect, LocalHistogram,
                        SparseEdgeDifference, graph)
from graphlocal.medium_defects import InfiniteRegularTreeCut, SquareLatticeEdgeCut


def word_tree(degree, depth):
    """Independently label two regular half trees by words."""
    vertices = [(side, word) for length in range(depth + 1)
                for word in product(range(degree - 1), repeat=length) for side in (0, 1)]
    index = {vertex: i for i, vertex in enumerate(vertices)}
    edge = index[(0, ())], index[(1, ())]
    edges = [edge]
    edges.extend((index[(side, word)], index[(side, word[:-1])])
                 for side, word in vertices if word)
    return SparseEdgeDifference(graph(len(vertices), edges), [(*edge, -1)])


def torus_cut(period):
    """Periodic witnesses, independent of the production rectangular buffer."""
    vertices = list(reversed([(x, y) for x in range(period) for y in range(period)]))
    index = {vertex: i for i, vertex in enumerate(vertices)}
    edges = [(index[(x, y)], index[((x + dx) % period, (y + dy) % period)])
             for x, y in vertices for dx, dy in ((1, 0), (0, 1))]
    edge = index[(0, 0)], index[(1, 0)]
    return SparseEdgeDifference(graph(len(vertices), edges), [(*edge, -1)])


def laplacian_moments(histogram, order):
    """Direct integer diagonal powers; independent of uniformized moments."""
    moments = [Q(0)] * (order + 1)
    for key, coefficient in histogram.values.items():
        g = key.graph
        counts = [1] + [0] * (g.n - 1)
        moments[0] += coefficient
        for power in range(1, order + 1):
            counts = [g.rows[u].bit_count() * counts[u] - sum(counts[v] for v in g.neighbors(u))
                      for u in range(g.n)]
            moments[power] += coefficient * counts[0]
    return tuple(moments)


class MediumDefectTests(unittest.TestCase):
    def test_regular_tree_atoms_against_two_buffer_depths(self):
        cases = [(degree, radius) for degree in (2, 3, 4) for radius in (1, 2)] + [(3, 3)]
        for degree, radius in cases:
            source = InfiniteRegularTreeCut(degree)
            local = source.local(radius)
            standard = source.finite_at_radius(radius)
            larger = word_tree(degree, 2 * radius + 2)
            self.assertEqual(local, standard.local(radius))
            self.assertEqual(local, larger.local(radius))
            self.assertEqual(standard.degree_bound, degree)
            self.assertEqual(standard.edit_bound, 1)

    def test_tree_norms_and_degree_two_line_identity(self):
        for radius in range(6):
            self.assertEqual(InfiniteRegularTreeCut(2).local(radius), CutLineDefect().local(radius))
        for degree in (2, 3, 4):
            q = degree - 1
            for radius in range(1, 5):
                source = InfiniteRegularTreeCut(degree)
                local = source.local(radius)
                branch_sum = sum(q ** j for j in range(radius))
                regular_size = 1 + degree * branch_sum
                self.assertEqual(len(local.values), radius + 1)
                self.assertEqual(local.norm(0), 4 * branch_sum)
                self.assertEqual(local.mass, 0)
                for k in (1, 2):
                    expected = 2 * branch_sum * regular_size ** k
                    expected += 2 * sum(q ** j * (regular_size - sum(q ** i for i in range(radius - j))) ** k
                                        for j in range(radius))
                    self.assertEqual(local.norm(k), expected)

    def test_square_lattice_rectangular_and_periodic_witnesses(self):
        source = SquareLatticeEdgeCut()
        for radius in (1, 2, 3):
            local = source.local(radius)
            self.assertEqual(local, source.finite_at_radius(radius).local(radius))
            self.assertEqual(local, torus_cut(4 * radius + 5).local(radius))
            self.assertEqual(local, torus_cut(4 * radius + 7).local(radius))
            self.assertEqual(local.mass, 0)

    def test_radius_consistency_and_metadata(self):
        for source in (InfiniteRegularTreeCut(3), InfiniteRegularTreeCut(4), SquareLatticeEdgeCut()):
            highest = source.local(3)
            self.assertEqual(source.local(0), LocalHistogram(0))
            for radius in (0, 1, 2):
                self.assertEqual(highest.truncate(radius), source.local(radius))
            self.assertEqual(source.edit_bound, 1)
            self.assertEqual(source.mass, 0)
            self.assertFalse(source.positive)
            self.assertIsNone(source.variation_bound)
            self.assertEqual(source.norm_bound(1, 2), source.local(1).norm(2))
            self.assertEqual(source.approximate(1, 2, "1e-20").error, 0)

    def test_laplacian_moments_and_lattice_cycle_contribution(self):
        for degree in (2, 3, 4):
            moments = laplacian_moments(InfiniteRegularTreeCut(degree).local(2), 4)
            expected = (0, -2, -4 * degree, -6 * degree ** 2 - 6 * degree + 4,
                        -8 * degree ** 3 - 24 * degree ** 2 + 16 * degree)
            self.assertEqual(moments, expected)
        lattice = laplacian_moments(SquareLatticeEdgeCut().local(2), 4)
        self.assertEqual(lattice, (0, -2, -16, -116, -848))
        tree = laplacian_moments(InfiniteRegularTreeCut(4).local(2), 4)
        self.assertEqual(tree[:4], lattice[:4])
        self.assertEqual(tree[4] - lattice[4], 16)

    def test_vertex_budgets_and_argument_contracts(self):
        small = InfiniteRegularTreeCut(4, max_vertices=20)
        self.assertEqual(small.local(2).norm(0), 16)
        with self.assertRaises(BudgetExceeded):
            small.local(3)
        with self.assertRaises(BudgetExceeded):
            small.finite_at_radius(2)
        lattice = SquareLatticeEdgeCut(max_vertices=10)
        self.assertEqual(lattice.local(0), LocalHistogram(0))
        with self.assertRaises(BudgetExceeded):
            lattice.local(1)
        with self.assertRaises(BudgetExceeded):
            lattice.finite_at_radius(0)
        for invalid in (0, 1, True, Q(3, 2)):
            with self.assertRaises(ValueError):
                InfiniteRegularTreeCut(invalid)
        with self.assertRaises(ValueError):
            SquareLatticeEdgeCut(max_vertices=0)
        with self.assertRaises(ValueError):
            InfiniteRegularTreeCut(3).local(-1)


if __name__ == "__main__":
    unittest.main()
