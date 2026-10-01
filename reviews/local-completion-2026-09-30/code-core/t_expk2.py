from fractions import Fraction as Q
from graphlocal import *

class WeightOneOnly(Element):
    """Exact data = Line; approximations are honest k=1 certificates (labelled k=1)."""
    degree_bound, variation_bound, mass, positive = 2, Q(1), Q(1), True
    def norm_bound(self, radius, k): return Line().norm_bound(radius, k)
    def approximate(self, radius, k=1, epsilon="1e-6"):
        eta = Q(epsilon)
        big = LocalHistogram(radius, [(star(99), eta / 100)])   # k=1 weight exactly eta
        return LocalApproximation(Line().local(radius) + big, 1, eta)

r, k, eps = 1, 3, Q("1e-3")
src = WeightOneOnly()
a = src.approximate(r, k, Q("1e-6"))
print("source contract at k=1 holds:", (a.histogram - Line().local(r)).norm(1) <= a.error,
      "| source returned k =", a.k, "(requested", k, ")")
got = exp(src / 10).approximate(r, k, eps)
ref = exp(Line() / 10).approximate(r, k, Q("1e-30"))
lower = (got.histogram - ref.histogram).norm(k) - ref.error
print("exp() result labelled k =", got.k, " claimed error =", float(got.error), "<= eps:", got.error <= eps)
print("true k=3 error >=", float(lower), " => certificate false:", lower > got.error)
p = (src * Line()).approximate(r, k, eps)
print("Product with same source returns k =", p.k, "(honest)")
