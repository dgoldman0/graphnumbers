from fractions import Fraction as Q
from graphlocal import *

class WeightOneOnly(Element):
    """Line, but approximations certify only the k=1 seminorm (honestly labelled k=1)."""
    degree_bound, variation_bound, mass, positive = 2, Q(1), Q(1), True
    def norm_bound(self, radius, k): return Line().norm_bound(radius, k)
    def approximate(self, radius, k=1, epsilon="1e-6"):
        eta = Q(epsilon)
        ball = Line().local(radius)
        n = 2 * radius + 1                      # size of the line ball
        # perturb the line-ball coefficient by eta/n: k=1 error is exactly eta
        return LocalApproximation(ball.scale(1 + eta / n), 1, eta)

r, k, eps = 1, 3, Q("1e-3")
got = exp(WeightOneOnly() / 10).approximate(r, k, eps)
ref = exp(Line() / 10).approximate(r, k, Q("1e-30"))
true_err = (got.histogram - ref.histogram).norm(k)
print("claimed k =", got.k, " claimed error =", float(got.error), " <= eps:", got.error <= eps)
print("actual weighted (k=3) error >=", float(true_err - ref.error), " unsound:", true_err - ref.error > got.error)
