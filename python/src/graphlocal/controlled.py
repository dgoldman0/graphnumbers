"""Heat certificates for finite polynomial bounds on signed return moments."""
from dataclasses import dataclass
from fractions import Fraction as Q

from .defects import _relative_moments
from .elements import as_element, moment_profile, _checked_approximation
from .graphs import BudgetExceeded, integer, rational
from .local import Interval, LocalApproximation, LocalHistogram


@dataclass(frozen=True)
class ControlledHeatCertificate:
    interval: Interval
    time: Q
    degree_bound: int
    moment_profile: tuple
    steps: int
    radius: int
    exp_partial_sum: Q
    exp_tail_bound: Q
    moment_tail_bound: Q
    local: LocalApproximation
    returns: tuple

    def to_data(self):
        return {"interval": self.interval.to_data(), "time": str(self.time),
                "degree_bound": self.degree_bound,
                "moment_profile": [str(a) for a in self.moment_profile],
                "steps": self.steps, "radius": self.radius,
                "exp_partial_sum": str(self.exp_partial_sum),
                "exp_tail_bound": str(self.exp_tail_bound),
                "moment_tail_bound": str(self.moment_tail_bound),
                "local_error_bound": str(self.local.error),
                "retained_returns": [str(a) for a in self.returns],
                "local_types": len(self.local.histogram.values),
                "local_variation": str(self.local.histogram.norm(0))}


def controlled_heat(value, time, epsilon="1e-8", max_steps=256):
    """Certified heat on a degree-bounded element with a finite moment profile.

    Profiles propagate under sums, scalars and arbitrary Cartesian products.
    This includes products of defects whose first-order edit certificate is
    unavailable. It does not define heat on every completed graph element.
    """
    value = as_element(value)
    time, epsilon = rational(time), rational(epsilon)
    integer(max_steps, "max_steps")
    if time < 0 or epsilon <= 0:
        raise ValueError("Time must be nonnegative and tolerance positive")
    profile = moment_profile(value)
    if value.degree_bound is None or profile is None:
        raise ValueError("Controlled heat requires degree and moment certificates")
    degree = integer(value.degree_bound, "degree_bound")
    if not degree:
        profile = (profile[0],)
    growth = tuple(a * time ** r for r, a in enumerate(profile))
    bound = sum(growth, Q(0))
    if not bound:
        return ControlledHeatCertificate(Interval(0, 0), time, degree, profile, 0, 0,
                                         Q(1), Q(0), Q(0),
                                         LocalApproximation(LocalHistogram(0), 1), (Q(0),))
    lam, term, partial = time * degree, Q(1), Q(1)
    partials = [partial]
    for steps in range(max_steps + 1):
        if steps + 2 > lam:
            tail = term * lam / (steps + 1) / (1 - lam / (steps + 2))
            prefixes = tuple(partials[steps - r] if r <= steps else Q(0)
                             for r in range(len(profile)))
            moment_tail = sum((g * (partial - s + tail) / (partial + tail)
                               for g, s in zip(growth, prefixes)), Q(0))
            budget = moment_tail + bound * tail / (2 * (partial + tail))
            if budget <= epsilon / 2:
                break
        term *= lam / (steps + 1)
        partial += term
        partials.append(partial)
    else:
        raise BudgetExceeded("Controlled heat truncation exceeds max_steps")
    radius = (steps + 1) // 2
    local = _checked_approximation(value, radius, 1, epsilon / 2)
    # The target has no mass on higher-degree types. Projecting an inexact
    # approximation onto its certified support cannot increase its error.
    local = local.project_degree(degree)
    moments = _relative_moments(local, degree, steps)
    weight, numerator = Q(1), Q(0)
    for j, m in enumerate(moments):
        if j:
            weight *= lam / j
        numerator += weight * m
    numerator_bound = sum((g * s for g, s in zip(growth, prefixes)), Q(0))
    lo = max(-numerator_bound, numerator - partial * local.error)
    hi = min(numerator_bound, numerator + partial * local.error)
    if lo > hi:
        raise ValueError("Local data contradict the supplied moment profile")
    lower = min(lo / partial, lo / (partial + tail)) - moment_tail
    upper = max(hi / partial, hi / (partial + tail)) + moment_tail
    interval = Interval(max(lower, -bound), min(upper, bound))
    if interval.radius > epsilon:
        raise ArithmeticError("Controlled heat error budget failure")
    return ControlledHeatCertificate(interval, time, degree, profile, steps, radius,
                                     partial, tail, moment_tail, local, moments)
