"""Exact geometry certificates for finite edge-edit interaction heat.

Support distances force cyclic walks to spend extra steps between edits.
All bounds use rational arithmetic; extracting a geometry needs no rooted
graph isomorphism or subset expansion.
"""
from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import permutations
from math import comb, factorial

from .defects import apply_edge_edits
from .edge_interactions import EdgeInteraction
from .elements import Element
from .graphs import BudgetExceeded, integer, rational
from .local import Interval


@dataclass(frozen=True)
class InteractionGeometry:
    edit_count: int
    degree_bound: int
    scale: Q
    support_distances: tuple
    tour_costs: tuple
    vanishing_order: object
    identically_zero: bool
    bridge_reduction_applies: bool

    @property
    def moment_profile(self):
        """A compositional profile carrying the geometric vanishing orders."""
        if self.identically_zero:
            return (Q(0),)
        k = self.edit_count
        result = [Q(0)] * (k + max(cost for cost, _ in self.tour_costs) + 1)
        for cost, count in self.tour_costs:
            result[k + cost] += (self.scale * Q(2 ** k * count, factorial(k + cost - 1))
                                 * self.degree_bound ** cost)
        return tuple(result)

    def moment_bound(self, step):
        """Bound the absolute mixed trace of (I-L/D)**step at this D."""
        integer(step, "step")
        if self.identically_zero:
            return Q(0)
        k = self.edit_count
        count = sum(multiplicity * comb(step - cost - 1, k - 1)
                    for cost, multiplicity in self.tour_costs if step >= k + cost)
        return self.scale * Q(2 ** k * step * count, self.degree_bound ** k)

    def to_data(self):
        return {"edit_count": self.edit_count, "degree_bound": self.degree_bound,
                "scale": str(self.scale),
                "support_distances": [list(row) for row in self.support_distances],
                "tour_costs": [{"cost": cost, "count": count}
                               for cost, count in self.tour_costs],
                "moment_profile": [str(value) for value in self.moment_profile],
                "vanishing_order": self.vanishing_order,
                "identically_zero": self.identically_zero,
                "bridge_reduction_applies": self.bridge_reduction_applies}


def interaction_geometry(value, max_cycles=1000000):
    """Certify cyclic support distances after any exact bridge reduction.

The distance graph contains every edge present in any intermediate graph.
An empty interaction or edits lying in different union components give zero.
Otherwise (k-1)! oriented cyclic orders are counted with the first edit fixed.
The work limit is checked before those orders are enumerated.
"""
    integer(max_cycles, "max_cycles")
    if not isinstance(value, EdgeInteraction):
        raise TypeError("Geometry requires an EdgeInteraction")
    edits = value.active_edits
    k = len(edits)
    union, _ = apply_edge_edits(value.before, [e for e in value.edits if e[2] == 1])
    rows = []
    for u, v, _ in edits:
        found, queue = {u: 0, v: 0}, deque((u, v))
        while queue:
            source = queue.popleft()
            for target in union.neighbors(source):
                if target not in found:
                    found[target] = found[source] + 1
                    queue.append(target)
        rows.append(tuple(min(found[a], found[b]) if a in found else None
                          for a, b, _ in edits))
    distances = tuple(rows)
    zero = not k or any(distance is None for row in distances for distance in row)
    if zero:
        return InteractionGeometry(k, value.degree_bound, value.scale, distances,
                                   (), None, True, value.bridge_reduction_applies)
    if factorial(k - 1) > max_cycles:
        raise BudgetExceeded("Cyclic interaction orders exceed max_cycles")
    costs = Counter()
    for rest in permutations(range(1, k)):
        order = (0,) + rest
        costs[sum(distances[order[i]][order[(i + 1) % k]] for i in range(k))] += 1
    return InteractionGeometry(k, value.degree_bound, value.scale, distances,
                               tuple(sorted(costs.items())), k + min(costs), False,
                               value.bridge_reduction_applies)


class GeometricInteraction(Element):
    """The same finite interaction equipped with a geometry-sensitive profile.

    The source is preserved, and its local, approximation and finite methods
    are forwarded. The alternative profile remains valid under algebraic
    operations; it can be sharper at small times and coarser at large times.
    """
    def __init__(self, source, max_cycles=1000000):
        self.source = source
        self.geometry = interaction_geometry(source, max_cycles)
        for field in ("degree_bound", "variation_bound", "edit_bound", "mass", "positive"):
            setattr(self, field, getattr(source, field))
        self.moment_profile = self.geometry.moment_profile

    def local(self, radius):
        return self.source.local(radius)

    def norm_bound(self, radius, k):
        return self.source.norm_bound(radius, k)

    def approximate(self, radius, k=1, epsilon="0.000001"):
        return self.source.approximate(radius, k, epsilon)

    def finite(self, max_vertices=10000):
        return self.source.finite(max_vertices)


@dataclass(frozen=True)
class InteractionHeatBound:
    interval: Interval
    time: Q
    geometry: InteractionGeometry
    steps: int
    exp_partial_sum: Q
    exp_tail_bound: Q
    poisson_bounds: tuple
    bound_enclosure_error: Q
    after_step: object = None

    @property
    def magnitude_bound(self):
        return self.interval.upper

    def to_data(self):
        return {"interval": self.interval.to_data(), "time": str(self.time),
                "geometry": self.geometry.to_data(), "steps": self.steps,
                "exp_partial_sum": str(self.exp_partial_sum),
                "exp_tail_bound": str(self.exp_tail_bound),
                "poisson_bounds": [{"threshold": threshold, "lower": str(lo), "upper": str(hi)}
                                   for threshold, lo, hi in self.poisson_bounds],
                "bound_enclosure_error": str(self.bound_enclosure_error),
                "after_step": self.after_step}


def interaction_heat_bound(value, time, epsilon="1e-8", max_steps=256,
                           max_cycles=1000000, after_step=None):
    """Bound interaction heat using geometry without computing local types.

If after_step is given, bound only the absolute uniformization tail beyond
that step. Epsilon bounds the numerical uncertainty in the analytic bound,
or certifies that the whole magnitude is at most epsilon. It does not promise
an interval of radius epsilon for a general nonzero heat interaction.
"""
    time, epsilon = rational(time), rational(epsilon)
    integer(max_steps, "max_steps")
    if time < 0 or epsilon <= 0:
        raise ValueError("Time must be nonnegative and tolerance positive")
    if after_step is not None:
        integer(after_step, "after_step")
    geometry = value if isinstance(value, InteractionGeometry) else interaction_geometry(value, max_cycles)
    if geometry.identically_zero or not time:
        return InteractionHeatBound(Interval(0, 0), time, geometry, 0, Q(1), Q(0), (), Q(0), after_step)
    k, degree = geometry.edit_count, geometry.degree_bound
    prefactor = geometry.scale * Q(2 ** k, factorial(k - 1)) * time ** k
    # Bound each cyclic contribution separately. This onset bound follows
    # by replacing (j-1)!/(j-k-cost)! by a falling factorial of order k+cost.
    leading = {cost: geometry.scale * Q(2 ** k, factorial(k + cost - 1))
               * degree ** cost * time ** (k + cost)
               for cost, _ in geometry.tour_costs}
    lam, term, partial = time * degree, Q(1), Q(1)
    partials = [partial]
    for steps in range(max_steps + 1):
        if steps + 2 > lam:
            tail = term * lam / (steps + 1) / (1 - lam / (steps + 2))
            bounds, total, uncertainty = {}, Q(0), Q(0)
            for cost, count in geometry.tour_costs:
                threshold = max(cost, after_step + 1 - k) if after_step is not None else cost
                if threshold <= 0:
                    lo = hi = Q(1)
                elif threshold <= steps + 1:
                    prefix = partials[threshold - 1]
                    lo, hi = 1 - prefix / partial, 1 - prefix / (partial + tail)
                else:
                    lo, hi = Q(0), tail / (partial + tail)
                bounds[threshold] = (lo, hi)
                total += count * min(prefactor * hi, leading[cost])
                uncertainty += count * prefactor * (hi - lo)
            if uncertainty <= epsilon or total <= epsilon:
                return InteractionHeatBound(Interval(-total, total), time, geometry,
                                            steps, partial, tail,
                                            tuple((q, *bounds[q]) for q in sorted(bounds)),
                                            uncertainty, after_step)
        term *= lam / (steps + 1)
        partial += term
        partials.append(partial)
    raise BudgetExceeded("Geometry heat bound exceeds max_steps")
