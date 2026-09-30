"""Exact joint geometry, formal arithmetic, and weighted local certificates."""
from fractions import Fraction as Q
from itertools import combinations
from math import factorial
import unittest

from graphlocal import (BudgetExceeded, Element, Finite, LocalApproximation,
                        LocalHistogram, ball, cartesian, complete, cycle,
                        disjoint_union, graph, path, star)
from graphlocal.nonspectral import (ROOT_DEGREE, JointDistribution, MomentJet,
                                   RootStatistic, certified_jet,
                                   joint_distribution, link_components,
                                   multi_indices, rooted_cliques)


def paw():
    return graph(4, [(0, 1), (1, 2), (2, 0), (0, 3)])


def root_triangle_count(g, root):
    return sum(bool(g.rows[u] & (1 << v)) for u, v in combinations(g.neighbors(root), 2))


def direct_joint(g):
    result = {}
    for root in range(g.n):
        feature = (g.rows[root].bit_count(), root_triangle_count(g, root))
        result[feature] = result.get(feature, Q(0)) + 1
    return result


class InexactLocal(Element):
    def __init__(self, value):
        self.value = value
        self.request = None

    def approximate(self, radius, k=1, epsilon="1e-8"):
        self.request = radius, k, Q(epsilon)
        error = Q(epsilon) / 2
        noise = ball(complete(4), 0, radius)
        histogram = LocalHistogram(radius, [(noise, error / noise.n ** k)])
        return LocalApproximation(self.value.local(radius) + histogram, k, error)


class NonspectralTests(unittest.TestCase):
    def test_root_statistics_cartesian_additivity(self):
        statistics = (ROOT_DEGREE, rooted_cliques(3), rooted_cliques(4),
                      link_components(graph(1)), link_components(path(2)),
                      link_components(complete(3)))
        for g, h in ((paw(), path(3)), (complete(4), cycle(3))):
            product = cartesian(g, h)
            for u in range(g.n):
                for v in range(h.n):
                    for statistic in statistics:
                        expected = statistic(ball(g, u, 1)) + statistic(ball(h, v, 1))
                        self.assertEqual(statistic(ball(product, u * h.n + v, 1)), expected)
        self.assertEqual(rooted_cliques(3)(ball(complete(4), 0, 1)), 3)
        self.assertEqual(rooted_cliques(4)(ball(complete(4), 0, 1)), 1)

    def test_joint_pushforward_and_graph_arithmetic(self):
        axes = ROOT_DEGREE, rooted_cliques(3)
        for g in (path(4), paw(), complete(4), cartesian(paw(), path(3))):
            self.assertEqual(dict(joint_distribution(Finite.from_graph(g), axes).values), direct_joint(g))
        x = Finite.from_graph(paw()) - Finite.from_graph(cycle(3)) / 3
        y = Finite.from_graph(path(3)) + 2
        dx, dy = joint_distribution(x, axes), joint_distribution(y, axes)
        self.assertEqual(joint_distribution(x * y, axes), dx.convolve(dy))
        self.assertEqual(joint_distribution(2 * x - y, axes), 2 * dx - dy)
        self.assertEqual(dx.mass, x.mass)
        self.assertEqual(dx.moment((0, 0)), x.mass)

    def test_joint_correlations_survive_equal_marginals(self):
        axes = ROOT_DEGREE, rooted_cliques(3)
        g = disjoint_union(complete(3), star(3))
        h = disjoint_union(paw(), path(3))
        dg = joint_distribution(Finite.from_graph(g), axes)
        dh = joint_distribution(Finite.from_graph(h), axes)
        self.assertEqual(dict(dg.values), {(1, 0): 3, (2, 1): 3, (3, 0): 1})
        self.assertEqual(dict(dh.values), {(1, 0): 3, (2, 0): 1, (2, 1): 2, (3, 1): 1})
        for coordinate in range(2):
            marginals = []
            for distribution in (dg, dh):
                marginal = {}
                for feature, c in distribution.values.items():
                    key = feature[coordinate]
                    marginal[key] = marginal.get(key, Q(0)) + c
                marginals.append(marginal)
            self.assertEqual(*marginals)
        self.assertEqual(dg.moment((1, 1)), 6)
        self.assertEqual(dh.moment((1, 1)), 7)
        normalized_g, normalized_h = dg.scale(Q(1, 7)), dh.scale(Q(1, 7))
        self.assertEqual((normalized_g.jet(2).log() - normalized_h.jet(2).log()).moment((1, 1)), -Q(1, 7))

    def test_jet_coefficients_binomial_rule_and_cumulants(self):
        axes = ROOT_DEGREE, rooted_cliques(3)
        x = Finite.from_graph(paw(), normalize=True)
        y = Finite.from_graph(path(3), normalize=True)
        jx, jy = joint_distribution(x, axes).jet(4), joint_distribution(y, axes).jet(4)
        direct = joint_distribution((x * y).finite(), axes).jet(4)
        self.assertEqual(jx * jy, direct)
        for alpha in direct.indices:
            direct_moment = sum(c * a ** alpha[0] * b ** alpha[1]
                                for (a, b), c in direct_joint(cartesian(paw(), path(3))).items()) / 12
            self.assertEqual(direct.moment(alpha), direct_moment)
            self.assertEqual(direct.coefficient(alpha), direct_moment / (factorial(alpha[0]) * factorial(alpha[1])))
        self.assertEqual((jx * jy).log(), jx.log() + jy.log())
        self.assertEqual(jx.log().exp(), jx)
        self.assertEqual(jx.truncate(2), joint_distribution(x, axes).jet(2))
        self.assertEqual(jx.constant(2).mass, 2)

    def test_formal_reciprocal_and_exponential_arithmetic(self):
        axes = (rooted_cliques(4),)
        t = Q(3, 5)
        difference = JointDistribution(axes, {(2,): 1, (0,): -1}).jet(5)
        unit = 1 + t * difference
        inverse = unit.reciprocal()
        self.assertEqual(unit * inverse, unit.constant(1))
        self.assertEqual([inverse.moment((i,)) for i in (1, 2, 3)],
                         [-2 * t, 8 * t ** 2 - 4 * t, -48 * t ** 3 + 48 * t ** 2 - 8 * t])
        exponential = (t * difference).exp()
        self.assertEqual(exponential.log(), t * difference)
        for i in range(1, 6):
            self.assertEqual(exponential.log().moment((i,)), 2 ** i * t)
        a = MomentJet((ROOT_DEGREE, rooted_cliques(3)), 4,
                      {(1, 0): Q(2, 3), (0, 1): -1, (1, 1): Q(1, 7), (2, 0): 3})
        b = MomentJet(a.statistics, a.order, {(0, 2): Q(3, 2), (2, 1): 4})
        self.assertEqual((a + b).exp(), a.exp() * b.exp())
        self.assertEqual((2 + a).reciprocal() * (2 + a), a.constant(1))
        self.assertEqual((a + 1) ** 3, (a + 1) * (a + 1) * (a + 1))
        self.assertEqual((a ** 0), a.constant(1))
        self.assertEqual(a.constant(0).exp(), a.constant(1))

    def test_rook_shrikhande_nonspectral_axes(self):
        rook = cartesian(complete(4), complete(4))
        moves = ((1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3))
        shrikhande = graph(16, ((4 * a + b, 4 * ((a + da) % 4) + (b + db) % 4)
                               for a in range(4) for b in range(4) for da, db in moves))
        axes = (ROOT_DEGREE, rooted_cliques(4), link_components(complete(3)), link_components(cycle(6)))
        dr = joint_distribution(Finite.from_graph(rook, normalize=True), axes)
        ds = joint_distribution(Finite.from_graph(shrikhande, normalize=True), axes)
        self.assertEqual(dict(dr.values), {(6, 2, 2, 0): 1})
        self.assertEqual(dict(ds.values), {(6, 0, 0, 1): 1})
        for order in range(1, 5):
            self.assertEqual((dr - ds).moment((0, order, 0, 0)), 2 ** order)

    def test_certified_inexact_extraction(self):
        value = Finite.from_graph(paw()) - Finite.from_graph(cycle(3)) / 3
        axes = ROOT_DEGREE, rooted_cliques(4)
        source = InexactLocal(value)
        certificate = certified_jet(source, axes, 4, "1e-9")
        exact = joint_distribution(value, axes).jet(4)
        self.assertEqual(source.request, (1, 12, Q("1e-9")))
        self.assertEqual(certificate.local.error, Q("5e-10"))
        for alpha in exact.indices:
            interval = certificate.interval(alpha)
            self.assertTrue(interval.contains(exact.coefficient(alpha)))
            self.assertEqual(interval.radius, certificate.local.error / (factorial(alpha[0]) * factorial(alpha[1])))
        exact_certificate = certified_jet(value, axes, 4)
        self.assertEqual(exact_certificate.jet, exact)
        self.assertEqual(exact_certificate.interval((2, 2)).radius, 0)
        self.assertEqual(certified_jet(value, axes, 0).jet.mass, value.mass)
        self.assertEqual(certificate.to_data()["k"], 12)

    def test_contracts_axes_budgets_and_degenerate_orders(self):
        axes = (ROOT_DEGREE, rooted_cliques(3))
        source = InexactLocal(Finite.scalar(1))
        self.assertEqual(multi_indices(2, 2), ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)))
        with self.assertRaises(BudgetExceeded):
            certified_jet(source, axes, 4, max_coefficients=14)
        self.assertIsNone(source.request)
        with self.assertRaises(ValueError):
            joint_distribution(source, (ROOT_DEGREE, ROOT_DEGREE))
        with self.assertRaises(ValueError):
            link_components(disjoint_union(graph(1), graph(1)))
        with self.assertRaises(ValueError):
            rooted_cliques(2)
        with self.assertRaises(ValueError):
            RootStatistic("bad", 1, 0, lambda g: g.n)(ball(path(3), 1, 1))
        with self.assertRaises(ValueError):
            RootStatistic("bad", 1, 1, lambda g: -1)(graph(1))
        with self.assertRaises(ValueError):
            JointDistribution.from_histogram(Finite.scalar(1).local(0), axes)
        a = MomentJet(axes, 0, {(0, 0): 2})
        self.assertEqual(a.reciprocal().mass, Q(1, 2))
        self.assertEqual(a.constant(1).log().mass, 0)
        self.assertEqual(a.constant(0).exp().mass, 1)
        with self.assertRaises(ValueError):
            a.log()
        with self.assertRaises(ValueError):
            a.exp()
        with self.assertRaises(ValueError):
            a.constant(0).reciprocal()
        with self.assertRaises(ValueError):
            a.coefficient((1, 0))
        with self.assertRaises(ValueError):
            a + MomentJet(tuple(reversed(axes)), 0)
        class EqualCallable:
            def __init__(self, multiplier):
                self.multiplier = multiplier

            def __call__(self, g):
                return self.multiplier * g.rows[0].bit_count()

            def __eq__(self, other):
                return True

        first = RootStatistic("custom", 1, 2, EqualCallable(1))
        second = RootStatistic("custom", 1, 2, EqualCallable(2))
        with self.assertRaises(ValueError):
            JointDistribution((first,), {(1,): 1}) * JointDistribution((second,), {(2,): 1})
        with self.assertRaises(ValueError):
            MomentJet((first,), 1) + MomentJet((second,), 1)
        with self.assertRaises(TypeError):
            a * 0.5
        with self.assertRaises(ValueError):
            certified_jet(source, axes, 1, 0)


if __name__ == "__main__":
    unittest.main()
