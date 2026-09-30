"""Independent finite witnesses for higher interaction geometry."""
from fractions import Fraction as Q
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from higher_interaction_verification import (FIXTURES, cross_moments,
                                             cyclic_contributions, fixture, record,
                                             sign_reversal_certificates,
                                             tree_catalog_verification)


class HigherInteractionVerificationTests(unittest.TestCase):
    def test_incidence_runtime_against_integer_subset_traces(self):
        for item in FIXTURES:
            with self.subTest(case=item[0]):
                self.assertTrue(record(item[0])["rank_one_runtime_agrees"])

    def test_repeated_label_contributes_despite_zero_pair_coupling(self):
        g, cuts, _ = fixture("repeated_defect_triple")
        cross = cross_moments(g, cuts, g.n)
        self.assertTrue(all(matrix[1][2] == 0 for matrix in cross))
        self.assertEqual(cyclic_contributions(g, cuts, 4), {3: Q(0), 4: Q(4)})

    def test_shortest_covering_words_cancel(self):
        g, cuts, _ = fixture("cancelled_triple")
        terms = cyclic_contributions(g, cuts, 6)
        self.assertEqual(terms, {3: Q(-24), 4: Q(24), 5: Q(0), 6: Q(0)})
        g, cuts, _ = fixture("cancelled_quadruple")
        self.assertEqual(cyclic_contributions(g, cuts, 5), {4: Q(10), 5: Q(-10)})
        self.assertEqual(sum(cyclic_contributions(g, cuts, 6).values()), 0)

    def test_binary_quartet_tree_leading_term(self):
        data = record("binary_quartet_tree")
        self.assertEqual(data["first_nonzero_order_in_computed_range"], 6)
        self.assertEqual(data["mixed_laplacian_moments"][6], 24)
        self.assertEqual(data["leading_heat_coefficient"], "1/30")

    def test_time_dependent_sign_with_rational_certificates(self):
        positive, negative = sign_reversal_certificates()
        self.assertGreater(positive.interval.lower, 0)
        self.assertLess(negative.interval.upper, 0)
        self.assertLessEqual(positive.interval.radius, Q("1e-10"))
        self.assertLessEqual(negative.interval.radius, Q("1e-10"))

    def test_exhaustive_tree_leading_terms_through_six_vertices(self):
        data = tree_catalog_verification(6)
        self.assertEqual([row["unlabeled_trees"] for row in data["tree_counts"]],
                         [1, 1, 1, 2, 3, 6])
        self.assertEqual(data["nonempty_cut_sets_verified"], 249)
        self.assertEqual(sum(data["leading_order_counts"].values()), 249)


if __name__ == "__main__":
    unittest.main()
