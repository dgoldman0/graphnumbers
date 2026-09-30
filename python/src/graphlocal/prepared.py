"""Reusable finite-resolution geometry and relative heat moment tables."""
from fractions import Fraction as Q

from .defects import (_relative_contract, _relative_from_moments,
                      _relative_moments, _relative_plan)
from .elements import Element
from .graphs import BudgetExceeded, integer, rational
from .local import LocalApproximation, LocalHistogram


class PreparedLocal(Element):
    """Cache exact local data from a supplied element through a fixed radius.

    This retains the source element and its analytic metadata. It does not
    treat an arbitrary histogram as a globally realizable completed element.
    Requests beyond the prepared radius raise BudgetExceeded.
    """
    def __init__(self, value, radius):
        self.value, self.radius = value, integer(radius, "radius")
        self.histogram = value.local(self.radius)
        if self.histogram.radius != self.radius:
            raise ValueError("Source returned the wrong local radius")
        self._locals = {self.radius: self.histogram}
        for field in ("degree_bound", "variation_bound", "edit_bound", "moment_profile", "mass", "positive"):
            setattr(self, field, getattr(value, field))

    def local(self, radius):
        integer(radius, "radius")
        if radius > self.radius:
            raise BudgetExceeded("Requested radius exceeds prepared geometry")
        if radius not in self._locals:
            self._locals[radius] = self.histogram.truncate(radius)
        return self._locals[radius]


class PreparedRelativeHeat:
    """Prepare geometry and return moments once for many times up to max_time.

    Every evaluation uses the same local approximation and prefixes of one
    aggregated moment vector. The error tolerance is fixed at preparation.
    Certificate.radius reports the retained geometry's actual radius, which
    may exceed the minimum radius needed by an individual time query.
    """
    def __init__(self, value, max_time, epsilon="1e-8", max_steps=256):
        self.max_time, self.epsilon, self.degree_bound, self.edit_bound = (
            _relative_contract(value, max_time, epsilon, max_steps))
        self.max_steps = integer(max_steps, "max_steps")
        plan = _relative_plan(self.max_time, self.degree_bound, self.edit_bound,
                              self.epsilon, max_steps)
        self.steps, self.radius = plan[0], (plan[0] + 1) // 2
        if not self.edit_bound or not self.max_time or not self.degree_bound:
            self.local = LocalApproximation(LocalHistogram(0), 1)
            self.returns = (Q(0),)
        else:
            self.local = value.approximate(self.radius, 1, self.epsilon / 2)
            if (self.local.histogram.radius != self.radius or self.local.k < 1
                    or self.local.error > self.epsilon / 2):
                raise ValueError("Source returned an invalid approximation contract")
            self.returns = _relative_moments(self.local, self.degree_bound, self.steps)

    def evaluate(self, time):
        time = rational(time)
        if not 0 <= time <= self.max_time:
            raise ValueError("Time must lie in the prepared interval")
        plan = _relative_plan(time, self.degree_bound, self.edit_bound,
                              self.epsilon, self.max_steps)
        if plan[0] > self.steps:
            raise BudgetExceeded("Query needs more moments than prepared")
        return _relative_from_moments(time, self.degree_bound, self.edit_bound,
                                      self.epsilon, plan, self.local, self.returns)

    at = evaluate
