"""Certified average heat return for elements with degree/variation bounds.

For U(G)=G/|G| the result encloses tr(exp(-t Laplacian(G)))/|G|.
Uniformization and all certificate arithmetic use exact rational numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache

from .elements import Element
from .graphs import BudgetExceeded, integer, rational
from .local import Interval, LocalApproximation


@lru_cache(maxsize=4096)
def lazy_returns(g, degree_bound, steps):
    """Diagonal powers of I - Laplacian(g)/D at vertex zero."""
    integer(degree_bound, "degree_bound")
    integer(steps, "steps")
    if g.max_degree > degree_bound:
        raise ValueError("Histogram exceeds the certified degree bound")
    if degree_bound == 0:
        return (Q(1),) * (steps + 1)
    counts, denominator = [1] + [0] * (g.n - 1), 1
    result = [Q(1)]
    for _ in range(steps):
        counts = [(degree_bound - g.rows[u].bit_count()) * counts[u]
                  + sum(counts[v] for v in g.neighbors(u)) for u in range(g.n)]
        denominator *= degree_bound
        result.append(Q(counts[0], denominator))
    return tuple(result)


@dataclass(frozen=True)
class HeatCertificate:
    interval: Interval
    time: Q
    degree_bound: int
    steps: int
    radius: int
    variation_bound: Q
    positive_mass: Q | None
    exp_partial_sum: Q
    exp_tail_bound: Q
    retained_numerator: Q
    local: LocalApproximation
    returns: tuple[Q, ...]

    def to_data(self, include_histogram=False):
        values = {
            "interval": self.interval.to_data(), "time": str(self.time),
            "degree_bound": self.degree_bound, "steps": self.steps, "radius": self.radius,
            "variation_bound": str(self.variation_bound),
            "positive_mass": None if self.positive_mass is None else str(self.positive_mass),
            "exp_partial_sum": str(self.exp_partial_sum), "exp_tail_bound": str(self.exp_tail_bound),
            "retained_numerator": str(self.retained_numerator),
            "local_error_bound": str(self.local.error),
            "retained_returns": [str(c) for c in self.returns],
            "local_types": len(self.local.histogram.values),
            "largest_ball": max((g.graph.n for g in self.local.histogram.values), default=0),
        }
        if include_histogram:
            values["local_approximation"] = self.local.to_data()
        return values


def heat_return(value: Element, time, epsilon="0.000001", max_steps=128):
    """Enclose the heat functional with interval radius <= epsilon.

    The value must provide a uniform maximum-degree bound and a finite global
    total-variation bound. The full completion does not supply these for every
    element. Built-in finite combinations, Line and their sums/products do.
    """
    time, epsilon = rational(time), rational(epsilon)
    integer(max_steps, "max_steps")
    if time < 0 or epsilon <= 0:
        raise ValueError("time must be nonnegative and epsilon positive")
    if value.degree_bound is None or value.variation_bound is None:
        raise ValueError("Heat return requires degree and global variation certificates")
    degree = integer(value.degree_bound, "degree_bound")
    variation = rational(value.variation_bound)
    if variation < 0:
        raise ValueError("Variation bound must be nonnegative")
    mass = rational(value.mass) if value.positive and value.mass is not None else None
    if mass is not None and not 0 <= mass <= variation:
        raise ValueError("Inconsistent positivity, mass and variation certificates")
    lam = time * degree
    term, partial = Q(1), Q(1)
    for steps in range(max_steps + 1):
        if steps + 2 > lam:
            tail = term * lam / (steps + 1) / (1 - lam / (steps + 2))
            error = (mass / 2 if mass is not None else variation) * tail / (partial + tail)
            if error <= epsilon / 2:
                break
        term *= lam / (steps + 1)
        partial += term
    else:
        raise BudgetExceeded("Poisson truncation exceeds max_steps")
    radius = (steps + 1) // 2
    local = value.approximate(radius, 1, epsilon / 2)
    if local.histogram.radius != radius or local.k < 1 or local.error > epsilon / 2:
        raise ValueError("Element returned an invalid approximation contract")
    returns = [Q(0)] * (steps + 1)
    for key, coefficient in local.histogram.values.items():
        for j, prob in enumerate(lazy_returns(key.graph, degree, steps)):
            returns[j] += coefficient * prob
    term, numerator = Q(1), Q(0)
    for j, prob in enumerate(returns):
        if j:
            term *= lam / j
        numerator += term * prob
    error = partial * local.error
    if mass is not None:
        low_a, high_a = max(Q(0), numerator - error), min(mass * partial, numerator + error)
        lower, upper = low_a / (partial + tail), (high_a + mass * tail) / (partial + tail)
    else:
        low_a = max(-variation * partial, numerator - error)
        high_a = min(variation * partial, numerator + error)
        lower = (low_a - variation * tail) / (partial + tail)
        upper = (high_a + variation * tail) / (partial + tail)
    if low_a > high_a:
        raise ValueError("Local data contradict the supplied global bounds")
    interval = Interval(lower, upper)
    if interval.radius > epsilon:
        raise ArithmeticError("Internal heat error budget failure")
    return HeatCertificate(interval, time, degree, steps, radius, variation, mass,
                           partial, tail, numerator, local, tuple(returns))
