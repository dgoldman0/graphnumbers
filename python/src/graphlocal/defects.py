"""Sparse edge changes, a beyond-measure cut limit, and relative heat traces.

`edit_bound` certifies the return-moment and rank bounds supplied by a uniform
weighted edit budget, or their proved extension under sums and products with
finite-variation elements. This is additional analytic data, not a bound
supplied by the topology on every element. Standard finite-rank perturbation
estimates give the heat tail. All interval computations below are rational.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .elements import Element, Finite
from .graphs import (BudgetExceeded, Graph, ball, cycle, distances, induced,
                     integer, path, rational)
from .heat import lazy_returns
from .local import Interval, LocalApproximation, LocalHistogram


def apply_edge_edits(before, edits):
    """Apply distinct (u,v,+1/-1) edits; preserve all vertex labels."""
    rows, used, checked = list(before.rows), set(), []
    for u, v, sign in edits:
        integer(u, "edge vertex")
        integer(v, "edge vertex")
        if u >= before.n or v >= before.n or u == v:
            raise ValueError("Edited edges require distinct vertices in range")
        if isinstance(sign, bool) or not isinstance(sign, int) or sign not in (-1, 1):
            raise ValueError("An edge edit sign must be +1 or -1")
        u, v = min(u, v), max(u, v)
        if (u, v) in used:
            raise ValueError("Supply each edited edge once")
        used.add((u, v))
        present = bool(rows[u] & (1 << v))
        if present == (sign == 1):
            raise ValueError("Cannot insert an existing edge or remove an absent edge")
        rows[u] ^= 1 << v
        rows[v] ^= 1 << u
        checked.append((u, v, sign))
    return Graph(tuple(rows)), tuple(checked)


class SparseEdgeDifference(Element):
    """The exact signed element scale*(after-before), with local cancellation.

    Default output is unnormalized, so a fixed defect remains observable as
    graph size grows. normalize=True divides by the common vertex count.
    """
    def __init__(self, before, edits, normalize=False):
        if normalize and not before.n:
            raise ValueError("The empty graph cannot be normalized")
        self.before = before
        self.after, self.edits = apply_edge_edits(before, edits)
        self.scale = Q(1, before.n) if normalize else Q(1)
        self.degree_bound = max(before.max_degree, self.after.max_degree)
        self.edit_bound = len(self.edits) * self.scale
        self.variation_bound = 2 * before.n * self.scale if self.edits else Q(0)
        self.mass, self.positive = Q(0), not self.edits

    def affected_roots(self, radius):
        integer(radius, "radius")
        roots = set()
        for u in {v for edge in self.edits for v in edge[:2]}:
            roots.update(distances(self.before, u, radius))
            roots.update(distances(self.after, u, radius))
        return tuple(sorted(roots))

    def local(self, radius):
        return LocalHistogram(radius,
                              ((g, coefficient) for u in self.affected_roots(radius)
                               for g, coefficient in [(ball(self.after, u, radius), self.scale),
                                                       (ball(self.before, u, radius), -self.scale)]))

    def finite(self, max_vertices=10000):
        return Finite([(self.scale, self.after), (-self.scale, self.before)]).finite(max_vertices)


class CutLineDefect(Element):
    """Local limit of P_n-C_n; local unweighted variation is exactly 4r.

    There is no finite global signed measure for this element. A uniform
    one-edge-change certificate nevertheless defines its relative heat trace.
    """
    degree_bound, edit_bound, mass = 2, Q(1), Q(0)
    variation_bound, positive = None, False

    def local(self, radius):
        integer(radius, "radius")
        if not radius:
            return LocalHistogram(0)
        terms = []
        for j in range(radius):
            g = path(radius + j + 1)
            rooted = induced(g, [j] + [u for u in range(g.n) if u != j])
            terms.append((rooted, Q(2)))
        g = path(2 * radius + 1)
        rooted = induced(g, [radius] + [u for u in range(g.n) if u != radius])
        terms.append((rooted, Q(-2 * radius)))
        return LocalHistogram(radius, terms)

    def finite_at_radius(self, radius):
        integer(radius, "radius")
        n = max(3, 2 * radius + 2)
        return SparseEdgeDifference(cycle(n), [(0, n - 1, -1)])


@dataclass(frozen=True)
class RelativeHeatCertificate:
    interval: Interval
    time: Q
    degree_bound: int
    edit_bound: Q
    steps: int
    radius: int
    exp_partial_sum: Q
    exp_tail_bound: Q
    defect_tail_bound: Q
    retained_numerator: Q
    local: LocalApproximation
    returns: tuple[Q, ...]

    def to_data(self, include_histogram=False):
        result = {
            "interval": self.interval.to_data(), "time": str(self.time),
            "degree_bound": self.degree_bound, "edit_bound": str(self.edit_bound),
            "steps": self.steps, "radius": self.radius,
            "exp_partial_sum": str(self.exp_partial_sum),
            "exp_tail_bound": str(self.exp_tail_bound),
            "defect_tail_bound": str(self.defect_tail_bound),
            "retained_numerator": str(self.retained_numerator),
            "local_error_bound": str(self.local.error),
            "retained_returns": [str(x) for x in self.returns],
            "local_types": len(self.local.histogram.values),
            "local_variation": str(self.local.histogram.norm(0)),
            "largest_ball": max((key.graph.n for key in self.local.histogram.values), default=0),
        }
        if include_histogram:
            result["local_approximation"] = self.local.to_data()
        return result


def relative_heat(value, time, epsilon="1e-8", max_steps=256):
    """Certified heat-trace difference for a uniformly bounded edit class.

    Finite signed inputs and suitable beyond-measure limits are supported.
    The edit/degree bounds and mass zero are explicit mathematical contracts.
    This does not continuously extend heat return to the whole completion.
    """
    time, epsilon = rational(time), rational(epsilon)
    integer(max_steps, "max_steps")
    if time < 0 or epsilon <= 0:
        raise ValueError("time must be nonnegative and epsilon positive")
    if value.degree_bound is None or value.edit_bound is None or value.mass != 0:
        raise ValueError("Relative heat requires degree/edit certificates and mass zero")
    degree, q = integer(value.degree_bound, "degree_bound"), rational(value.edit_bound)
    if q < 0:
        raise ValueError("Weighted edit bound must be nonnegative")
    if not q or not time or not degree:
        return RelativeHeatCertificate(Interval(0, 0), time, degree, q, 0, 0,
                                       Q(1), Q(0), Q(0), Q(0),
                                       LocalApproximation(LocalHistogram(0), 1), (Q(0),))
    lam, beta = time * degree, 2 * q * time
    term, partial = Q(1), Q(1)
    for steps in range(max_steps + 1):
        if steps + 2 > lam:
            tail = term * lam / (steps + 1) / (1 - lam / (steps + 2))
            budget = beta * (term + Q(3, 2) * tail) / (partial + tail)
            if budget <= epsilon / 2:
                break
        term *= lam / (steps + 1)
        partial += term
    else:
        raise BudgetExceeded("Relative heat truncation exceeds max_steps")
    radius = (steps + 1) // 2
    local = value.approximate(radius, 1, epsilon / 2)
    if local.histogram.radius != radius or local.k < 1 or local.error > epsilon / 2:
        raise ValueError("Element returned an invalid approximation contract")
    moments = [Q(0)] * (steps + 1)
    for key, coefficient in local.histogram.values.items():
        for j, prob in enumerate(lazy_returns(key.graph, degree, steps)):
            moments[j] += coefficient * prob
    coefficient, numerator = Q(1), Q(0)
    for j, moment in enumerate(moments):
        if j:
            coefficient *= lam / j
        numerator += coefficient * moment
    # |A| <= (2q/D) sum j*lambda^j/j! = beta*(S-term_M).
    a_bound = beta * (partial - term)
    a_low = max(-a_bound, numerator - partial * local.error)
    a_high = min(a_bound, numerator + partial * local.error)
    if a_low > a_high:
        raise ValueError("Local data contradict the supplied edit bound")
    defect_tail = beta * (term + tail) / (partial + tail)
    lower = min(a_low / partial, a_low / (partial + tail)) - defect_tail
    upper = max(a_high / partial, a_high / (partial + tail)) + defect_tail
    global_bound = q * min(Q(1), 2 * time)
    lower, upper = max(lower, -global_bound), min(upper, global_bound)
    interval = Interval(lower, upper)
    if interval.radius > epsilon:
        raise ArithmeticError("Internal relative heat error budget failure")
    return RelativeHeatCertificate(interval, time, degree, q, steps, radius, partial,
                                   tail, defect_tail, numerator, local, tuple(moments))
