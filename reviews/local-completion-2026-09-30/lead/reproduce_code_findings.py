"""Reproduce the four library findings confirmed by the lead reviewer.

Run from the repository root with PYTHONPATH=python/src.
(c) and (d) use custom Element subclasses that violate the documented
contract; they show missing defensive checks, not wrong answers on
built-in elements.
"""
from fractions import Fraction as Q

from graphlocal import (Finite, Line, NeumannInverse, complete, disjoint_union, exp,
                        graph, heat_return, joint_distribution, path, reconstruct,
                        rooted_cliques, star)
from graphlocal.elements import Element
from graphlocal.local import LocalApproximation, LocalHistogram

print("(a) reconstruct() with a disconnected catalog graph (reconstruction.py:105-164)")
result = reconstruct(Finite.from_graph(path(2)).local(1),
                     [disjoint_union(path(2), graph(1)), graph(1)])
print("    cost", result.cost, "| negative_mass", result.negative_mass,
      "| coefficients", tuple(str(c) for c in result.coefficients),
      "| returned element terms", [(str(c), g.n) for c, g in result.element.terms])
print("    -> cost/negative mass describe catalog coordinates; the merged element is just K2")

print("(b) lru_cache keys on call form (nonspectral.py:68,84)")
K4 = Finite.from_graph(complete(4), normalize=True)
a = joint_distribution(K4, (rooted_cliques(3),))
b = joint_distribution(K4, (rooted_cliques(size=3),))
print("    rooted_cliques(3) is rooted_cliques(size=3):", rooted_cliques(3) is rooted_cliques(size=3),
      "| identical laws compare equal:", a == b)


class WeightOne(Element):
    """Honest k=1 certificates that ignore the requested weight."""
    degree_bound, variation_bound, mass, positive = 2, Q(1), Q(1), True

    def norm_bound(self, r, k):
        return Line().norm_bound(r, k)

    def approximate(self, r, k=1, epsilon="1e-6"):
        eta = Q(epsilon)
        bump = LocalHistogram(r, [(star(99), eta / 100)])
        return LocalApproximation(Line().local(r) + bump, 1, eta)


print("(c) exp() trusts the inner approximation's weight (elements.py:332-347)")
got = exp(WeightOne() / 10).approximate(1, 3, "1e-3")
ref = exp(Line() / 10).approximate(1, 3, Q("1e-30"))
print("    claimed k", got.k, "| claimed error", float(got.error),
      "| actual weight-3 error", float((got.histogram - ref.histogram).norm(3)))


class WrongDegree(Element):
    """Line/4 (maximum degree 2) misreporting degree_bound=1."""
    degree_bound, variation_bound, mass, positive = 1, Q(1, 4), Q(1, 4), True

    def local(self, r):
        return (Line() / 4).local(r)


print("(d) degree-cap projection drops mass silently (inverse.py:109-113, controlled.py:83-88)")
certificate = NeumannInverse(WrongDegree()).approximation_certificate(1, 1, "1e-6")
histogram = certificate.approximation.histogram
mass = sum(c * key.graph.n for key, c in histogram.values.items())
print("    certified histogram mass", mass, "| certified error", certificate.approximation.error,
      "| NeumannInverse(...).mass", NeumannInverse(WrongDegree()).mass)
try:
    heat_return(WrongDegree(), time="1/2")
    print("    heat_return accepted the same input")
except ValueError as error:
    print("    heat_return rejects the same input:", error)
