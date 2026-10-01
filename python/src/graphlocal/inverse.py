"""Constructive inverses using degree growth and strict variation control."""
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from math import comb, factorial

from .elements import Element, as_element, _checked_approximation
from .graphs import BudgetExceeded, graph, integer, rational
from .local import LocalApproximation, LocalHistogram


@lru_cache(maxsize=256)
def _stirling_row(order):
    if not order:
        return (1,)
    previous = _stirling_row(order - 1)
    return tuple((previous[j - 1] if j else 0) + (j * previous[j] if j < len(previous) else 0)
                 for j in range(order + 1))


def _geometric_moment(q, order):
    return sum((s * factorial(j) * q ** j / (1 - q) ** (j + 1)
                for j, s in enumerate(_stirling_row(order))), Q(0))


def _power_majorant(q, degree, radius, k, derivative=False):
    exponent, factor = radius * k, (radius + 1) ** k
    if derivative and not q:
        return Q(factor * (1 + degree) ** exponent)
    return factor * sum((comb(exponent, p) * degree ** p
                         * (_geometric_moment(q, p + 1) / q if derivative
                            else _geometric_moment(q, p))
                         for p in range(exponent + 1)), Q(0))


@dataclass(frozen=True)
class InverseCertificate:
    approximation: LocalApproximation
    truncation_degree: int
    source_variation_bound: Q
    approximant_variation: Q
    source_error: Q
    stability_bound: Q
    tail_bound: Q

    def to_data(self, include_histogram=False):
        result = {"radius": self.approximation.histogram.radius, "weight": self.approximation.k,
                  "error_bound": str(self.approximation.error),
                  "truncation_degree": self.truncation_degree,
                  "source_variation_bound": str(self.source_variation_bound),
                  "approximant_variation": str(self.approximant_variation),
                  "source_error": str(self.source_error),
                  "stability_bound": str(self.stability_bound), "tail_bound": str(self.tail_bound),
                  "local_types": len(self.approximation.histogram.values)}
        if include_histogram:
            result["histogram"] = self.approximation.histogram.to_data()
        return result


class NeumannInverse(Element):
    """The inverse (1-value)^(-1) in the local Cartesian graph completion.

    Requires a finite degree cap and a global signed variation bound q<1.
    Degree growth in Cartesian powers is linear, so fixed-radius weighted
    norms grow only polynomially with the power. This is a sufficient
    constructive domain, not a general invertibility decision procedure.
    """
    def __init__(self, value, max_terms=128, max_vertices=10000):
        self.value = as_element(value)
        self.max_terms = integer(max_terms, "max_terms")
        self.max_vertices = integer(max_vertices, "max_vertices", 1)
        if self.value.degree_bound is None or self.value.variation_bound is None:
            raise ValueError("Neumann inverse requires degree and global variation certificates")
        self.source_degree = integer(self.value.degree_bound, "degree_bound")
        self.q = rational(self.value.variation_bound)
        if not 0 <= self.q < 1:
            raise ValueError("Neumann inverse requires 0 <= variation_bound < 1")
        self.variation_bound = 1 / (1 - self.q)
        self.degree_bound = 0 if not self.source_degree or not self.q else None
        self.positive = not self.q or self.value.positive
        if self.value.mass is not None and abs(rational(self.value.mass)) > self.q:
            raise ValueError("Source mass contradicts its variation certificate")
        self.mass = (Q(1) if not self.q else None if self.value.mass is None
                     else 1 / (1 - rational(self.value.mass)))

    def norm_bound(self, radius, k):
        integer(radius, "radius")
        integer(k, "weight exponent", 1)
        if not self.q:
            return Q(1)
        return _power_majorant(self.q, self.source_degree, radius, k)

    def local(self, radius):
        integer(radius, "radius")
        if not self.q or (self.mass is not None and (not radius or not self.source_degree)):
            return LocalHistogram(radius, [(graph(1), Q(1) if not self.q else self.mass)])
        return super().local(radius)

    def approximation_certificate(self, radius, k=1, epsilon="1e-6"):
        integer(radius, "radius")
        integer(k, "weight exponent", 1)
        epsilon = rational(epsilon)
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not self.q or (self.mass is not None and (not radius or not self.source_degree)):
            mass = Q(1) if not self.q else self.mass
            approximation = LocalApproximation(LocalHistogram(radius, [(graph(1), mass)]), k)
            return InverseCertificate(approximation, 0, self.q, self.q, Q(0), Q(0), Q(0))
        reference = (1 + self.q) / 2
        lipschitz = _power_majorant(reference, self.source_degree, radius, k, derivative=True)
        delta = min((1 - self.q) / 2, epsilon / (2 * lipschitz))
        source = _checked_approximation(self.value, radius, k, delta)
        if (self.value.mass is not None
                and abs(source.histogram.mass - self.value.mass) > source.error):
            raise ValueError("Source approximation contradicts its mass certificate")
        histogram = source.project_degree(self.source_degree).histogram
        qhat = histogram.norm(0)
        if qhat > self.q + source.error or qhat >= 1:
            raise ValueError("Source approximation contradicts the variation certificate")
        stability = source.error * _power_majorant(max(self.q, qhat), self.source_degree,
                                                   radius, k, derivative=True)
        factor, exponent = (radius + 1) ** k, radius * k
        tail = _power_majorant(qhat, self.source_degree, radius, k)
        power = Q(1)
        for n in range(self.max_terms + 1):
            tail -= factor * (1 + n * self.source_degree) ** exponent * power
            if tail <= epsilon / 2:
                break
            power *= qhat
        else:
            raise BudgetExceeded("Neumann inverse truncation exceeds max_terms")
        if tail < 0:
            raise ArithmeticError("Negative geometric remainder")
        term = total = LocalHistogram(radius, [(graph(1), Q(1))])
        for _ in range(n):
            term = term.multiply(histogram, max_vertices=self.max_vertices)
            total = total + term
        approximation = LocalApproximation(total, k, stability + tail)
        if approximation.error > epsilon:
            raise ArithmeticError("Neumann inverse error budget failure")
        return InverseCertificate(approximation, n, self.q, qhat, source.error, stability, tail)

    def approximate(self, radius, k=1, epsilon="1e-6"):
        return self.approximation_certificate(radius, k, epsilon).approximation
