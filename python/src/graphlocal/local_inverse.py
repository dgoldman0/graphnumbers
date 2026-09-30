"""Residual certificates for inversion in one weighted local algebra.

Success concerns the specified radius and weight only. It does not prove
that the source is a unit of the local Cartesian graph completion, or that the
local inverse belongs to a coherent balanced family. There is deliberately
no Element wrapper for these certificates.
"""
from dataclasses import dataclass
from fractions import Fraction as Q

from .elements import as_element
from .graphs import BudgetExceeded, graph, integer, rational
from .local import LocalApproximation, LocalHistogram


class UncertifiedLocalInverse(ValueError):
    """A sufficient residual test failed; this does not establish noninvertibility."""


@dataclass(frozen=True)
class LocalInverseCertificate:
    """Approximation to an inverse in one radius/weight local Banach algebra."""
    approximation: LocalApproximation
    inverse_norm_bound: Q
    source_error: Q
    residual_norm: Q
    residual_bound: Q
    candidate_norm: Q
    initial_residual_bound: Q
    truncation_degree: int
    stability_bound: Q
    tail_bound: Q

    def to_data(self, include_histogram=False):
        result = {"scope": "inverse of the specified weighted local marginal only",
                  "radius": self.approximation.histogram.radius,
                  "weight": self.approximation.k,
                  "error_bound": str(self.approximation.error),
                  "inverse_norm_bound": str(self.inverse_norm_bound),
                  "source_error": str(self.source_error),
                  "residual_norm": str(self.residual_norm),
                  "residual_bound": str(self.residual_bound),
                  "candidate_norm": str(self.candidate_norm),
                  "initial_residual_bound": str(self.initial_residual_bound),
                  "truncation_degree": self.truncation_degree,
                  "stability_bound": str(self.stability_bound),
                  "tail_bound": str(self.tail_bound)}
        if include_histogram:
            result["histogram"] = self.approximation.histogram.to_data()
        return result


def _positive_tolerance(value, name):
    value = rational(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _arguments(candidate, k, source_epsilon, max_vertices):
    if not isinstance(candidate, LocalHistogram):
        raise TypeError("The inverse candidate must be a LocalHistogram")
    integer(k, "weight exponent", 1)
    integer(max_vertices, "max_vertices", 1)
    return _positive_tolerance(source_epsilon, "source_epsilon")


def _source(value, radius, k, epsilon):
    source = as_element(value).approximate(radius, k, epsilon)
    if not isinstance(source, LocalApproximation) or not isinstance(source.histogram, LocalHistogram):
        raise TypeError("The source must return a LocalApproximation with a LocalHistogram")
    if source.histogram.radius != radius or source.k < k or source.error > epsilon:
        raise ValueError("The source returned an invalid local approximation contract")
    return source.truncate(radius, k)


def _residual(source, candidate, max_vertices):
    unit = LocalHistogram(candidate.radius, [(graph(1), Q(1))])
    return unit - source.histogram.multiply(candidate, max_vertices=max_vertices)


def local_inverse_certificate(value, candidate, k=1, source_epsilon="1e-8", max_vertices=10000):
    """Certify a candidate inverse of T_r(value), for r=candidate.radius.

    For B=||candidate||_k and a source a0 with error delta, compute
    q0=||1-a0*candidate||_k and q=q0+delta*B. If q<1, the local inverse
    has norm at most B/(1-q), and its distance from the candidate is at
    most B*q/(1-q). This supplies no global graph-algebra unit claim.
    """
    source_epsilon = _arguments(candidate, k, source_epsilon, max_vertices)
    source = _source(value, candidate.radius, k, source_epsilon)
    bound = candidate.norm(k)
    residual_norm = _residual(source, candidate, max_vertices).norm(k)
    residual_bound = residual_norm + source.error * bound
    if residual_bound >= 1:
        raise UncertifiedLocalInverse(
            "The sufficient local inverse test requires residual_bound < 1; "
            "failure does not prove local or global noninvertibility")
    inverse_bound = bound / (1 - residual_bound)
    error = inverse_bound * residual_bound
    approximation = LocalApproximation(candidate, k, error)
    return LocalInverseCertificate(approximation, inverse_bound, source.error,
                                   residual_norm, residual_bound, bound,
                                   residual_bound, 0, Q(0), error)


def refine_local_inverse(value, candidate, k=1, epsilon="1e-8", source_epsilon="1e-8",
                         max_terms=128, max_vertices=10000):
    """Refine a successful local candidate with certified finite residual powers.

    First certify the candidate, obtaining a true inverse-norm bound M.
    Refetch the source with delta <= min((1-q)/(4B), epsilon/(4M**2)).
    For e0=1-a0*candidate and q0=||e0||_k, the finite approximation is
    candidate*sum(e0**j, j=0..N). Its error is bounded by the sum of
    delta*M*B/(1-q0) and B*q0**(N+1)/(1-q0). The returned object still
    certifies only this specified weighted local marginal.
    """
    source_epsilon = _arguments(candidate, k, source_epsilon, max_vertices)
    epsilon = _positive_tolerance(epsilon, "epsilon")
    integer(max_terms, "max_terms")
    initial = local_inverse_certificate(value, candidate, k, source_epsilon, max_vertices)
    bound, inverse_bound, initial_q = (initial.candidate_norm, initial.inverse_norm_bound,
                                       initial.residual_bound)
    # A zero candidate has residual norm one and cannot pass the initial test.
    source_tolerance = min(source_epsilon, (1 - initial_q) / (4 * bound),
                           epsilon / (4 * inverse_bound ** 2))
    source = _source(value, candidate.radius, k, source_tolerance)
    residual = _residual(source, candidate, max_vertices)
    q0 = residual.norm(k)
    q = q0 + source.error * bound
    if q0 > initial_q + source.error * bound or q >= 1:
        raise ValueError("The refined source contradicts the initial local error certificate")
    approximant_inverse_bound = bound / (1 - q0)
    stability = source.error * inverse_bound * approximant_inverse_bound
    tail = bound * q0 / (1 - q0)
    for degree in range(max_terms + 1):
        if tail <= epsilon / 2:
            break
        tail *= q0
    else:
        raise BudgetExceeded("Local inverse refinement exceeds max_terms")
    term = total = candidate
    for _ in range(degree):
        term = term.multiply(residual, max_vertices=max_vertices)
        total = total + term
    error = stability + tail
    if error > epsilon:
        raise ArithmeticError("Local inverse refinement exceeded its error budget")
    approximation = LocalApproximation(total, k, error)
    return LocalInverseCertificate(approximation, inverse_bound, source.error, q0, q,
                                   bound, initial_q, degree, stability, tail)
