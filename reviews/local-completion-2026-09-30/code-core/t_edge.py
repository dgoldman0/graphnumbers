from fractions import Fraction as Q
from graphlocal import *
from graphlocal.graphs import disjoint_union
from graphlocal.elements import Exponential

# 1. reconstruct with a disconnected catalog graph
K2K1 = disjoint_union(path(2), graph(1))
target = Finite.from_graph(path(2)).local(1)
rec = reconstruct(target, [K2K1, graph(1)])
el = rec.element
true_mass = sum(abs(c) * g.n for c, g in el.terms)
print("disconnected catalog: coefficients", rec.coefficients, "reported cost", rec.cost,
      "negative_mass", rec.negative_mass, "| returned element terms", [(str(c), g.rows) for c, g in el.terms],
      "its coefficient mass", true_mass)

# 2. Exponential max_terms does not govern exp_bracket's internal 512-term budget
X = Finite.scalar(600)
try:
    exp(X, max_terms=100000).approximate(0, 1, "1e-3")
except BudgetExceeded as e:
    print("exp with max_terms=100000:", e)

# 3. A source that silently returns a weaker weight than requested
class WeakK(Element):
    degree_bound, variation_bound, mass, positive = 2, Q(1), Q(1), True
    def norm_bound(self, radius, k): return Line().norm_bound(radius, k)
    def approximate(self, radius, k=1, epsilon="1e-6"):
        # valid k=1 certificate only
        return LocalApproximation(Line().local(radius), 1, Q(0))
p = (WeakK() * WeakK()).approximate(2, 3, "1e-6")
print("Product requested k=3, returned k =", p.k)
e = exp(WeakK() / 10).approximate(1, 3, "1e-6")
print("Exponential requested k=3, returned k =", e.k, "(input approximations had k=1)")
