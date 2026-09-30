"""Certified algebra exponentials of the cut-line element with degree growth.

These are signed units beyond the finite signed-measure model.  Their local
variation is exp(4*r*abs(parameter)) for r >= 2.  This module uses the linear
degree growth of E**n to avoid exponentiating its much larger weighted norm.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from math import comb

from .defects import CutLineDefect
from .elements import Element, _request, exp_bracket
from .graphs import BudgetExceeded, graph, integer, rational
from .local import LocalApproximation, LocalHistogram


def _poisson_polynomial(parameter, exponent):
    """Return exp(-x)*sum_n (1+2*n)**exponent*x**n/n! exactly."""
    row = (1,)
    result = Q(0)
    for order in range(exponent + 1):
        if order:
            row = tuple((row[j - 1] if j else 0)
                        + (j * row[j] if j < len(row) else 0)
                        for j in range(order + 1))
        touchard = sum((s * parameter ** j for j, s in enumerate(row)), Q(0))
        result += comb(exponent, order) * 2 ** order * touchard
    return result


def _weighted_tail(parameter, degree, exponent, next_term):
    """Geometric bound for sum_(n>degree) (1+2*n)**exponent*x**n/n!.

    next_term is x**(degree+1)/(degree+1)!; the consecutive weighted
    summand ratios decrease.  None means the ratio is not yet below one.
    """
    ratio = (parameter / (degree + 2)
             * Q(2 * degree + 5, 2 * degree + 3) ** exponent)
    if ratio >= 1:
        return None, ratio
    return (2 * degree + 3) ** exponent * next_term / (1 - ratio), ratio


@dataclass(frozen=True)
class DefectExponentialCertificate:
    approximation: LocalApproximation
    parameter: Q
    poisson_parameter: Q
    truncation_degree: int
    tail_bound: Q
    tail_ratio: Q
    unweighted_tail_bound: Q

    @property
    def degree(self):
        return self.truncation_degree

    def to_data(self, include_histogram=False):
        result = {
            "radius": self.approximation.histogram.radius,
            "weight": self.approximation.k,
            "parameter": str(self.parameter),
            "poisson_parameter": str(self.poisson_parameter),
            "truncation_degree": self.truncation_degree,
            "tail_bound": str(self.tail_bound),
            "tail_ratio": str(self.tail_ratio),
            "unweighted_tail_bound": str(self.unweighted_tail_bound),
            "error_bound": str(self.approximation.error),
            "local_types": len(self.approximation.histogram.values),
        }
        if include_histogram:
            result["histogram"] = self.approximation.histogram.to_data()
        return result


class CutLineExponential(Element):
    """The signed unit exp(parameter*E), with inverse exp(-parameter*E).

    E is CutLineDefect().  The parameter is exact rational.  This is the
    algebra exponential, not the heat operator.  Nonzero parameters give
    unbounded degrees and unbounded marginal variation, so neither a global
    degree cap nor a finite signed-measure certificate is supplied.
    """
    mass = Q(1)

    def __init__(self, parameter, max_terms=128, max_vertices=10000):
        self.parameter = rational(parameter)
        self.max_terms = integer(max_terms, "max_terms")
        self.max_vertices = integer(max_vertices, "max_vertices", 1)
        self.degree_bound = None if self.parameter else 0
        self.variation_bound = None if self.parameter else Q(1)
        self.positive = not self.parameter

    def inverse(self):
        return CutLineExponential(-self.parameter, self.max_terms, self.max_vertices)

    def norm_bound(self, radius, k):
        integer(radius, "radius")
        integer(k, "weight exponent", 1)
        if not radius or not self.parameter:
            return Q(1)
        parameter = 4 * radius * abs(self.parameter)
        polynomial = _poisson_polynomial(parameter, radius * k)
        return (radius + 1) ** k * polynomial * exp_bracket(parameter)[1]

    def approximation_certificate(self, radius, k=1, epsilon="1e-6"):
        epsilon = _request(radius, k, epsilon)
        identity = LocalHistogram(radius, [(graph(1), Q(1))])
        if not radius or not self.parameter:
            approximation = LocalApproximation(identity, k)
            return DefectExponentialCertificate(
                approximation, self.parameter, Q(0), 0, Q(0), Q(0), Q(0))

        parameter = 4 * radius * abs(self.parameter)
        exponent, factor = radius * k, (radius + 1) ** k
        scalar_term = Q(1)
        for degree in range(self.max_terms + 1):
            next_term = scalar_term * parameter / (degree + 1)
            tail, ratio = _weighted_tail(parameter, degree, exponent, next_term)
            if tail is not None and factor * tail <= epsilon:
                tail *= factor
                unweighted_tail = next_term / (1 - parameter / (degree + 2))
                break
            scalar_term = next_term
        else:
            raise BudgetExceeded("Cut-line exponential truncation exceeds max_terms")

        term = total = identity
        if degree:
            if 2 * radius + 1 > self.max_vertices:
                raise BudgetExceeded("Cut-line source ball exceeds max_vertices")
            source = CutLineDefect().local(radius).scale(self.parameter)
            for n in range(1, degree + 1):
                term = term.multiply(source, max_vertices=self.max_vertices).scale(Q(1, n))
                total = total + term
        approximation = LocalApproximation(total, k, tail)
        return DefectExponentialCertificate(
            approximation, self.parameter, parameter, degree, tail, ratio, unweighted_tail)

    def approximate(self, radius, k=1, epsilon="1e-6"):
        return self.approximation_certificate(radius, k, epsilon).approximation
