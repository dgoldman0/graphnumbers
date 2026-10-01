"""Regressions for concrete failures identified by the September audit.

Valid-input checks use scalar closed forms and exact coefficient arithmetic.
The malformed-source cases check rejection, not the truth of a bad source.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from pathlib import Path
import os
import subprocess
import sys
import unittest

from graphlocal import (CutLineExponential, EdgeInteraction, Element, Finite,
                        GeometricInteraction, Line, LocalApproximation,
                        LocalHistogram, NeumannInverse, complete, controlled_heat,
                        exp, graph, interaction_heat_bound, interaction_moments,
                        joint_distribution, link_components, path, reconstruct,
                        rooted_cliques, star, tree_interaction_leading)
from graphlocal.graphs import disjoint_union


class WrongContract(Element):
    def __init__(self, violation):
        self.violation = violation

    def norm_bound(self, radius, k):
        return Q(1)

    def approximate(self, radius, k=1, epsilon="1e-6"):
        local = Finite.scalar(1).local(radius + (self.violation == "radius"))
        weight = 1 if self.violation == "weight" else k
        error = 2 * Q(epsilon) if self.violation == "error" else Q(0)
        return LocalApproximation(local, weight, error)


class MisdeclaredDegree(Element):
    degree_bound, variation_bound, mass, positive = 1, Q(1, 4), Q(1, 4), True

    def local(self, radius):
        return (Line() / 4).local(radius)


class StrongerScalar(Element):
    def norm_bound(self, radius, k):
        return Q(1)

    def approximate(self, radius, k=1, epsilon="1e-6"):
        return LocalApproximation(Finite.scalar(1).local(radius), k + 1)


class AuditRegressions(unittest.TestCase):
    def test_expression_operations_reject_bad_source_contracts(self):
        for violation in ("weight", "radius", "error"):
            for operation in (lambda x: exp(x), lambda x: x + 1,
                              lambda x: 2 * x, lambda x: x * Finite.scalar(2)):
                with self.subTest(violation=violation, operation=operation):
                    with self.assertRaisesRegex(ValueError, "approximation contract"):
                        operation(WrongContract(violation)).approximate(1, 3)

    def test_stronger_source_certificates_remain_usable(self):
        source = StrongerScalar()
        for expression, scalar in ((source + 1, 2), (2 * source, 2),
                                   (source * Finite.scalar(3), 3)):
            result = expression.approximate(2, 2)
            self.assertEqual(result.histogram, Finite.scalar(scalar).local(2))
            self.assertEqual(result.k, 2)
            self.assertEqual(result.error, 0)

    def test_degree_contradictions_rejected_by_both_certifiers(self):
        with self.assertRaisesRegex(ValueError, "degree bound"):
            NeumannInverse(MisdeclaredDegree()).approximate(1, 2)
        with self.assertRaisesRegex(ValueError, "degree bound"):
            controlled_heat(MisdeclaredDegree(), "1/2")

    def test_degree_projection_accepts_exactly_budgeted_noise(self):
        # True target is K1. The sole error is on a degree-two, 3-vertex ball.
        truth = Finite.scalar(1).local(1)
        noise = LocalHistogram(1, [(star(2), Q(1, 900))])
        source = LocalApproximation(truth + noise, 2, Q(1, 100))
        projected = source.project_degree(1)
        self.assertEqual(projected.histogram, truth)
        with self.assertRaisesRegex(ValueError, "degree bound"):
            LocalApproximation(truth + noise, 2, Q(1, 101)).project_degree(1)

    def test_reconstruction_rejects_disconnected_catalog(self):
        target = Finite.from_graph(path(2)).local(1)
        with self.assertRaisesRegex(ValueError, "connected"):
            reconstruct(target, [disjoint_union(path(2), graph(1)), graph(1)])

    def test_builtin_axes_normalize_call_syntax(self):
        x = Finite.from_graph(complete(4), normalize=True)
        a = joint_distribution(x, (rooted_cliques(3),))
        b = joint_distribution(x, (rooted_cliques(size=3),))
        self.assertEqual(dict(a.values), {(3,): Q(1)})
        self.assertEqual(a, b)
        self.assertEqual(dict(a.convolve(b).values), {(6,): Q(1)})
        p = path(2)
        self.assertIs(link_components(p), link_components(pattern=p, name=None))
        self.assertIs(link_components(p), link_components(p, link_components(p).name))

    def test_exact_scalar_cases_and_geometric_wrapper(self):
        for x in (NeumannInverse(0), CutLineExponential(0), NeumannInverse(Q(1, 4))):
            self.assertEqual(x.local(3), Finite.scalar(x.mass).local(3))
        self.assertEqual(controlled_heat(3, 1).interval.lower, 3)
        self.assertEqual(controlled_heat(3, 1).interval.upper, 3)
        raw = EdgeInteraction(path(4), [(0, 1, -1), (2, 3, -1)])
        wrapped = GeometricInteraction(raw)
        self.assertEqual(interaction_moments(wrapped, 4).laplacian,
                         interaction_moments(raw, 4).laplacian)
        self.assertEqual(tree_interaction_leading(wrapped), tree_interaction_leading(raw))
        self.assertEqual(interaction_heat_bound(wrapped, "1/4").interval,
                         interaction_heat_bound(raw, "1/4").interval)

    def test_explicit_exponential_budget_removes_hidden_majorant_ceiling(self):
        result = exp(600, max_terms=2000).approximate(0, 1, "1e-3")
        with localcontext() as context:
            context.prec = 380
            reference = Q(Decimal(600).exp())
        # Decimal rounding is much smaller than the spare error allowance.
        error = abs(result.histogram.mass - reference)
        self.assertLess(error + Q(1, 10**100), result.error)
        self.assertLessEqual(result.error, Q("1e-3"))

    def test_duplicate_edges_and_optimized_research_validation(self):
        for edges in ([(0, 1), (0, 1)], [(0, 1), (1, 0)]):
            with self.assertRaisesRegex(ValueError, "Repeated"):
                graph(2, edges)
        research = Path(__file__).resolve().parents[2] / "research" / "local-completion"
        program = """
from verify_local_algebra import Graph, RootTypes
for rows in ((2, 0), (3, 1)):
    try:
        Graph(rows)
    except ValueError:
        pass
    else:
        raise RuntimeError('Malformed graph accepted')
try:
    RootTypes().register(Graph((0, 0)))
except ValueError:
    pass
else:
    raise RuntimeError('Disconnected rooted type accepted')
"""
        env = dict(os.environ, PYTHONPATH=str(research), PYTHONDONTWRITEBYTECODE="1")
        checked = subprocess.run([sys.executable, "-O", "-c", program], env=env,
                                 capture_output=True, text=True, timeout=10)
        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == "__main__":
    unittest.main()
