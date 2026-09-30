from __future__ import annotations

import itertools
import random
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction as Q

from graphlocal import *
from graphlocal.elements import exp_bracket
from graphlocal.graphs import IsoGraph, induced, local_product
from graphlocal.heat import lazy_returns


def decimal(q):
    q = Q(q)
    return Decimal(q.numerator) / Decimal(q.denominator)


def dot(a, b):
    return sum((c * b.values.get(key, Q(0)) for key, c in a.values.items()), Q(0))


def matrix_returns(g, root, degree, steps):
    """Independent dense rational matrix multiplication, including all roots."""
    p = [[(Q(1) - Q(g.rows[i].bit_count(), degree)) if i == j
          else Q(bool(g.rows[i] & (1 << j)), degree)
          for j in range(g.n)] for i in range(g.n)]
    vector = [Q(i == root) for i in range(g.n)]
    result = [Q(1)]
    for _ in range(steps):
        vector = [sum(vector[j] * p[j][i] for j in range(g.n)) for i in range(g.n)]
        result.append(vector[root])
    return result


class Perturbed(Element):
    """Valid, deliberately inexact local oracle for exercising error propagation."""
    def __init__(self, value):
        self.value = value
        self.degree_bound = value.degree_bound
        self.variation_bound = value.variation_bound
        self.positive, self.mass = value.positive, value.mass

    def norm_bound(self, radius, k):
        return self.value.norm_bound(radius, k)

    def approximate(self, radius, k=1, epsilon="1e-6"):
        error = Q(epsilon) / 2
        shifted = self.value.local(radius) + Finite.scalar(error).local(radius)
        return LocalApproximation(shifted, k, error)


class GraphTests(unittest.TestCase):
    def test_validation(self):
        for rows in [(1,), (2, 0), (-1,), (4, 0)]:
            with self.assertRaises(ValueError):
                Graph(rows)
        with self.assertRaises(ValueError):
            graph(2, [(0, 0)])
        with self.assertRaises(ValueError):
            cycle(2)
        with self.assertRaises(ValueError):
            Finite.from_graph(graph(0), normalize=True)
        with self.assertRaises(TypeError):
            Finite.scalar(0.1)

    def test_isomorphism_against_permutations(self):
        rng = random.Random(314159)
        possibilities = list(itertools.combinations(range(4), 2))
        for _ in range(80):
            g = graph(4, [e for e in possibilities if rng.randrange(2)])
            h = graph(4, [e for e in possibilities if rng.randrange(2)])
            for rooted in (False, True):
                truth = any(induced(g, order) == h for order in itertools.permutations(range(4))
                            if not rooted or order[0] == 0)
                self.assertEqual(isomorphic(g, h, rooted), truth)
        self.assertFalse(isomorphic(cycle(6), disjoint_union(cycle(3), cycle(3))))

    def test_cartesian_against_kronecker_sum(self):
        for g, h in itertools.product([path(3), cycle(3), star(3)], repeat=2):
            c = cartesian(g, h)
            for u, v, x, y in itertools.product(range(g.n), range(h.n), range(g.n), range(h.n)):
                expected = (v == y and bool(g.rows[u] & (1 << x))) or (u == x and bool(h.rows[v] & (1 << y)))
                self.assertEqual(bool(c.rows[u * h.n + v] & (1 << (x * h.n + y))), expected)

    def test_disjoint_union_is_addition_and_labels_cancel(self):
        g = disjoint_union(path(3), path(3), graph(2))
        x = Finite.from_graph(g)
        y = (2 * Finite.from_graph(path(3)) + 2).finite()
        self.assertEqual(x, y)
        relabeled = induced(path(4), [2, 0, 3, 1])
        self.assertEqual(Finite([(1, path(4)), (-1, relabeled)]), Finite())

    def test_materialization_budget(self):
        with self.assertRaises(BudgetExceeded):
            (Finite.from_graph(path(3)) ** 3).finite(max_vertices=10)
        self.assertEqual((Finite.from_graph(path(3)) ** 0).finite(), Finite.scalar(1))


class LocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = catalog(4)

    def test_catalog_counts(self):
        self.assertEqual([sum(g.n == n for g in self.graphs) for n in range(1, 5)], [1, 1, 2, 6])

    def test_all_small_cartesian_histograms(self):
        for g, h in itertools.product(self.graphs, repeat=2):
            full = cartesian(g, h)
            for r in range(3):
                with self.subTest(g=g.rows, h=h.rows, radius=r):
                    a = LocalHistogram.from_graph(g, r)
                    b = LocalHistogram.from_graph(h, r)
                    self.assertEqual(a.multiply(b), LocalHistogram.from_graph(full, r))

    def test_contraction_and_signed_submultiplicativity(self):
        x = Finite([(1, path(4)), (-1, path(3))])
        y = Finite([(Q(1, 3), cycle(3)), (-2, path(2))])
        for r in range(4):
            for k in (1, 2, 3):
                a, b = x.local(r), y.local(r)
                self.assertLessEqual(a.multiply(b).norm(k), a.norm(k) * b.norm(k))
                self.assertLessEqual(a.truncate(0).norm(k), a.norm(k))

    def test_line_and_grid(self):
        for r in range(1, 5):
            line = Line().local(r)
            difference = Finite([(1, path(2 * r + 1)), (-1, path(2 * r))])
            self.assertEqual(line, difference.local(r))
            self.assertEqual(line, Line().finite_at_radius(r).local(r))
            grid = (Line() * Line()).local(r)
            self.assertEqual(grid.norm(1), 1 + 2 * r * (r + 1))

    def test_roundtrip_and_radius_validation(self):
        x = Finite([(Q(-2, 3), path(3)), (5, cycle(4))])
        self.assertEqual(Finite.from_data(x.to_data()), x)
        h = x.local(2)
        self.assertEqual(LocalHistogram.from_data(h.to_data()), h)
        with self.assertRaises(ValueError):
            LocalHistogram(1, [(path(3), 1)])
        with self.assertRaises(ValueError):
            h.truncate(3)
        with self.assertRaises(ValueError):
            LocalApproximation(h, 1).truncate(1, 2)

    def test_approximation_composition_with_inexact_inputs(self):
        x = Finite([(Q(1, 3), path(3)), (-1, path(2))])
        y = Finite.from_graph(cycle(3), True)
        for expression, exact in [(Perturbed(x) + Perturbed(y), x + y),
                                  (Perturbed(x) * Perturbed(y), x * y),
                                  (-3 * Perturbed(x), -3 * x)]:
            result = expression.approximate(2, 2, "1e-5")
            self.assertLessEqual(result.error, Q("1e-5"))
            self.assertLessEqual((result.histogram - exact.local(2)).norm(2), result.error)

    def test_observables_and_polynomial_derivative(self):
        x = Finite.from_graph(path(3))
        h = x.approximate(3, 4)
        self.assertTrue(VERTICES.evaluate(h).contains(3))
        self.assertTrue(EDGES.evaluate(h).contains(2))
        self.assertTrue(ISOLATED.evaluate(h).contains(0))
        self.assertTrue(walk_observable(4).evaluate(h).contains(8))
        derivative = polynomial_derivative(x, [1, 2, 3], direction=x)
        self.assertEqual(derivative.finite(), (2 * x + 6 * x * x).finite())

    def test_exponential_weighted_tail(self):
        h = Finite.from_graph(path(2), True)
        approx = exp(h / 10).approximate(1, 1, "1e-8")
        with localcontext() as ctx:
            ctx.prec = 70
            t = Decimal(1) / 10
            # On radius-one hypercube stars, the weight is n+1 and mass t^n/n!.
            total_norm = (1 + t) * t.exp()
            retained_norm = decimal(approx.histogram.norm(1))
            self.assertGreater(total_norm, retained_norm)
            self.assertLessEqual(total_norm - retained_norm, decimal(approx.error))
        for key, c in approx.histogram.values.items():
            n = key.graph.n - 1
            import math
            self.assertEqual(c, Q(1, 10) ** n / math.factorial(n))

    def test_exponential_inexact_oracle_and_derivative(self):
        with localcontext() as ctx:
            ctx.prec = 70
            expected = Decimal("0.2").exp()
            result = exp(Perturbed(Finite.scalar("0.2"))).approximate(1, 1, "1e-8")
            self.assertLessEqual(abs(decimal(result.histogram.mass) - expected), decimal(result.error))
            result = exp_derivative(Finite.scalar("0.2"), 3).approximate(0, 1, "1e-7")
            self.assertLessEqual(abs(decimal(result.histogram.mass) - 3 * expected), decimal(result.error))
        with self.assertRaises(BudgetExceeded):
            exp(Finite.scalar(10), max_terms=2).approximate(0, 1, "1e-10")


class HeatTests(unittest.TestCase):
    def assertDecimalEnclosed(self, interval, expected):
        self.assertLessEqual(decimal(interval.lower), expected)
        self.assertGreaterEqual(decimal(interval.upper), expected)

    def test_locality_against_full_matrix(self):
        for g in catalog(4):
            degree = max(1, g.max_degree + 1)
            for root in range(g.n):
                full = matrix_returns(g, root, degree, 6)
                for r in range(4):
                    self.assertEqual(list(lazy_returns(ball(g, root, r), degree, 2 * r)), full[:2 * r + 1])

    def test_boundary_failure_beyond_certified_order(self):
        g = cycle(7)
        full = matrix_returns(g, 0, 2, 3)
        local = lazy_returns(ball(g, 0, 1), 2, 3)
        self.assertEqual(local[:3], tuple(full[:3]))
        self.assertNotEqual(local[3], full[3])

    def test_heat_distinguishes_locally_identical_cycles(self):
        for radius in range(1, 5):
            n = 2 * radius + 2
            x, y = Finite.from_graph(cycle(n), True), Finite.from_graph(cycle(2 * n), True)
            self.assertEqual(x.local(radius), y.local(radius))
            hx, hy = heat_return(x, 1, "1e-15"), heat_return(y, 1, "1e-15")
            self.assertGreater(hx.interval.lower, hy.interval.upper)

    def test_complete_graph_closed_forms(self):
        with localcontext() as ctx:
            ctx.prec = 70
            for n, t in itertools.product(range(1, 6), [Q(0), Q(1, 10), Q(1, 2), Q(2)]):
                result = heat_return(Finite.from_graph(complete(n), True), t, "1e-9")
                expected = (1 + (n - 1) * (-n * decimal(t)).exp()) / n
                self.assertDecimalEnclosed(result.interval, expected)
                self.assertLessEqual(result.interval.radius, Q("1e-9"))

    def test_line_grid_and_signed_heat(self):
        with localcontext() as ctx:
            ctx.prec = 70
            for t in (Q(1, 10), Q(1, 2), Q(1)):
                d = decimal(t)
                term, series = Decimal(1), Decimal(1)
                for j in range(1, 150):
                    term *= d * d / (j * j)
                    series += term
                expected = (-2 * d).exp() * series
                self.assertDecimalEnclosed(heat_return(Line(), t, "1e-8").interval, expected)
                self.assertDecimalEnclosed(heat_return(Line() * Line(), t, "1e-8").interval, expected ** 2)
                signed = heat_return(3 * Line() - 2, t, "1e-8")
                self.assertDecimalEnclosed(signed.interval, 3 * expected - 2)

    def test_inexact_local_data(self):
        x = Finite.from_graph(path(2), True)
        with localcontext() as ctx:
            ctx.prec = 70
            expected = (1 + Decimal(-1).exp()) / 2
            for source in [Perturbed(x), Perturbed(3 * x - 1)]:
                result = heat_return(source, "1/2", "1e-7")
                truth = expected if source.mass == 1 else 3 * expected - 1
                self.assertDecimalEnclosed(result.interval, truth)
                self.assertLessEqual(result.interval.radius, Q("1e-7"))

    def test_zero_time_and_unnormalized_input(self):
        self.assertEqual(heat_return(Finite(), 1).interval, Interval(0, 0))
        self.assertEqual(heat_return(Finite.from_graph(path(4)), 0).interval, Interval(4, 4))
        self.assertEqual(heat_return(Finite.scalar(-3), 2).interval, Interval(-3, -3))

    def test_limits_and_rejection(self):
        with self.assertRaises(ValueError):
            heat_return(exp(Finite.from_graph(path(2))), 1)
        with self.assertRaises(ValueError):
            heat_return(Line(), -1)
        with self.assertRaises(TypeError):
            heat_return(Line(), 0.1)
        with self.assertRaises(BudgetExceeded):
            heat_return(Line(), 2, "1e-20", max_steps=1)
        with self.assertRaises(ValueError):
            lazy_returns(path(3), 1, 2)


class ReconstructionTests(unittest.TestCase):
    def test_line_cost_and_dual(self):
        target = Line().local(2)
        for graphs, cost, negative in [([path(4), path(5)], 9, 4),
                                        ([path(4), path(5), cycle(6)], 1, 0)]:
            result = reconstruct(target, graphs)
            self.assertEqual(result.element.local(2), target)
            self.assertEqual(result.cost, cost)
            self.assertEqual(result.negative_mass, negative)
            self.assertEqual(dot(result.dual, target), cost)
            for g in graphs:
                self.assertLessEqual(abs(dot(result.dual, LocalHistogram.from_graph(g, 2))), g.n)

    def test_infeasibility_witness_and_budgets(self):
        graphs = catalog(4)
        with self.assertRaises(OutOfSpan) as caught:
            reconstruct(Line().local(2), graphs)
        witness = caught.exception.witness
        self.assertNotEqual(dot(witness, Line().local(2)), 0)
        for g in graphs:
            self.assertEqual(dot(witness, LocalHistogram.from_graph(g, 2)), 0)
        with self.assertRaises(BudgetExceeded):
            reconstruct(Line().local(2), [path(4), path(5)], search_budget=1)
        with self.assertRaises(BudgetExceeded):
            catalog(5, label_budget=10)
        self.assertEqual(reconstruct(LocalHistogram(1), [path(2)]).cost, 0)


if __name__ == "__main__":
    unittest.main()
