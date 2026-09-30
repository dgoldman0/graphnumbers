"""Exact joint local statistics and certified exponential-moment jets.

A RootStatistic certifies a nonnegative integer, rooted-isomorphism invariant
function f with f(B square C) = f(B) + f(C) and
f(B) <= |B|**growth_degree. Built-ins carry proofs in NONSPECTRAL_CALCULUS.md;
for custom functions these global properties are the caller's contract.
Runtime evaluation checks the value and growth bound on each encountered ball.

Jets are finite formal polynomials with coefficient M_alpha / alpha!.
Reciprocals in this finite target do not establish invertibility of a graph
element. Formal log and exp here do not assert convergence of an infinite
moment-generating series.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from itertools import combinations, combinations_with_replacement
from math import comb, factorial, prod
from types import MappingProxyType

from .elements import as_element
from .graphs import (BudgetExceeded, Graph, ball, components, induced, integer,
                     isomorphic, rational)
from .local import Interval, LocalApproximation, LocalHistogram


@dataclass(frozen=True, eq=False)
class RootStatistic:
    """Caller-certified additive local statistic; the root has index zero.

    Axes retain their own object identity when compatibility is checked.
    Reuse the same statistic object across distributions and jets.
    The callable receives a ball at exactly ``radius``.
    """
    name: str
    radius: int
    growth_degree: int
    function: object

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Statistic name must be a nonempty string")
        integer(self.radius, "statistic radius")
        integer(self.growth_degree, "statistic growth degree")
        if not callable(self.function):
            raise TypeError("Statistic function must be callable")

    def __call__(self, rooted_ball):
        if not isinstance(rooted_ball, Graph) or not rooted_ball.n:
            raise ValueError("A nonempty rooted graph is required")
        local = ball(rooted_ball, 0, self.radius)
        value = self.function(local)
        integer(value, "statistic value")
        if value > local.n ** self.growth_degree:
            raise ValueError("Statistic value exceeds its polynomial growth bound")
        return value


def _root_degree(g):
    return g.rows[0].bit_count()


ROOT_DEGREE = RootStatistic("degree", 1, 1, _root_degree)


@lru_cache(maxsize=None)
def rooted_cliques(size):
    """Number of size-cliques containing the root, for size >= 3.

    This is the rooted count, without the division by ``size`` that would
    count each clique once after summing over all roots.
    """
    integer(size, "clique size", 3)

    def count(g):
        return sum(all(g.rows[u] & (1 << v) for u, v in combinations(vertices, 2))
                   for vertices in combinations(g.neighbors(0), size - 1))

    return RootStatistic(f"rooted_cliques_{size}", 1, size - 1, count)


@lru_cache(maxsize=None)
def link_components(pattern, name=None):
    """Count neighbor-link components isomorphic to a connected pattern.

    Cartesian neighbor links are disjoint unions. Connectedness of the
    pattern is therefore required for this component count to be additive.
    The default axis name is deterministic for the supplied representation.
    """
    if not isinstance(pattern, Graph) or not pattern.n:
        raise ValueError("The link pattern must be a nonempty graph")
    if len(tuple(components(pattern))) != 1:
        raise ValueError("The link pattern must be connected")
    if name is None:
        name = "link_components_" + "_".join(map(str, pattern.rows))

    def count(g):
        link = induced(g, tuple(g.neighbors(0)))
        return sum(isomorphic(piece, pattern) for piece in components(link))

    return RootStatistic(name, 1, 1, count)


def _axes(statistics):
    axes = tuple(statistics)
    if not axes or any(not isinstance(axis, RootStatistic) for axis in axes):
        raise ValueError("At least one RootStatistic axis is required")
    if len({axis.name for axis in axes}) != len(axes):
        raise ValueError("Statistic axis names must be distinct")
    return axes


def _index(alpha, dimension, order=None):
    alpha = tuple(alpha)
    if len(alpha) != dimension:
        raise ValueError("Multi-index dimension must match the statistic axes")
    for entry in alpha:
        integer(entry, "multi-index entry")
    if order is not None and sum(alpha) > order:
        raise ValueError("Multi-index exceeds the jet order")
    return alpha


def _factorial(alpha):
    return prod(factorial(entry) for entry in alpha)


def multi_indices(dimension, order, max_coefficients=10000):
    """All indices through total order, in total-degree/lexicographic order.

    The binomial coefficient budget is checked before constructing indices.
    """
    integer(dimension, "dimension", 1)
    integer(order, "order")
    integer(max_coefficients, "max_coefficients", 1)
    if comb(order + dimension, dimension) > max_coefficients:
        raise BudgetExceeded("Moment jet exceeds max_coefficients")
    result = []
    for total in range(order + 1):
        layer = []
        for positions in combinations_with_replacement(range(dimension), total):
            alpha = [0] * dimension
            for position in positions:
                alpha[position] += 1
            layer.append(tuple(alpha))
        result.extend(sorted(layer))
    return tuple(result)


@dataclass(frozen=True, eq=False)
class JointDistribution:
    """Finite signed rational law of an ordered tuple of local statistics."""
    statistics: tuple
    values: object

    def __init__(self, statistics, terms=()):
        axes = _axes(statistics)
        values = {}
        items = terms.items() if hasattr(terms, "items") else terms
        for feature, coefficient in items:
            feature = _index(feature, len(axes))
            values[feature] = values.get(feature, Q(0)) + rational(coefficient)
        object.__setattr__(self, "statistics", axes)
        object.__setattr__(self, "values", MappingProxyType(
            {a: c for a, c in sorted(values.items()) if c}))

    @classmethod
    def from_histogram(cls, histogram, statistics):
        axes = _axes(statistics)
        if not isinstance(histogram, LocalHistogram):
            raise TypeError("A LocalHistogram is required")
        if histogram.radius < max(axis.radius for axis in axes):
            raise ValueError("Histogram radius is too small for the statistics")
        return cls(axes, ((tuple(axis(key.graph) for axis in axes), coefficient)
                          for key, coefficient in histogram.values.items()))

    def _compatible(self, other):
        if not isinstance(other, JointDistribution) or self.statistics != other.statistics:
            raise ValueError("Joint distributions must have the same ordered statistic axes")

    @property
    def mass(self):
        return sum(self.values.values(), Q(0))

    def moment(self, alpha):
        """Exact raw joint moment, including signed or unnormalized laws."""
        alpha = _index(alpha, len(self.statistics))
        return sum((coefficient * prod(x ** a for x, a in zip(feature, alpha))
                    for feature, coefficient in self.values.items()), Q(0))

    def __eq__(self, other):
        if not isinstance(other, JointDistribution):
            return NotImplemented
        return self.statistics == other.statistics and self.values == other.values

    def __add__(self, other):
        self._compatible(other)
        return JointDistribution(self.statistics, tuple(self.values.items()) + tuple(other.values.items()))

    def __neg__(self):
        return self.scale(-1)

    def __sub__(self, other):
        return self + -other

    def scale(self, coefficient):
        coefficient = rational(coefficient)
        return JointDistribution(self.statistics, ((a, coefficient * c) for a, c in self.values.items()))

    def convolve(self, other):
        self._compatible(other)
        return JointDistribution(self.statistics,
                                 ((tuple(x + y for x, y in zip(a, b)), c * d)
                                  for a, c in self.values.items() for b, d in other.values.items()))

    def __mul__(self, other):
        return self.convolve(other) if isinstance(other, JointDistribution) else self.scale(other)

    def __rmul__(self, coefficient):
        return self.scale(coefficient)

    def jet(self, order, max_coefficients=10000):
        indices = multi_indices(len(self.statistics), order, max_coefficients)
        return MomentJet(self.statistics, order,
                         ((alpha, self.moment(alpha) / _factorial(alpha)) for alpha in indices),
                         max_coefficients=max_coefficients)

    def to_data(self):
        return {"axes": [axis.name for axis in self.statistics],
                "terms": [{"features": list(a), "coefficient": str(c)} for a, c in self.values.items()]}


def joint_distribution(value, statistics):
    """Extract an exact finite joint law from an element's exact local array."""
    axes = _axes(statistics)
    return JointDistribution.from_histogram(as_element(value).local(max(axis.radius for axis in axes)), axes)


@dataclass(frozen=True, eq=False)
class MomentJet:
    """Exact polynomial modulo all monomials above a fixed total degree.

    Coefficients are exponential-moment coefficients M_alpha / alpha!;
    multiplication therefore implements the joint binomial moment rule.
    ``log`` gives cumulant coefficients after exact mass-one normalization.
    """
    statistics: tuple
    order: int
    coefficients: object
    indices: tuple

    def __init__(self, statistics, order, coefficients=(), max_coefficients=10000):
        axes = _axes(statistics)
        indices = multi_indices(len(axes), order, max_coefficients)
        values = {}
        items = coefficients.items() if hasattr(coefficients, "items") else coefficients
        for alpha, coefficient in items:
            alpha = _index(alpha, len(axes), order)
            values[alpha] = values.get(alpha, Q(0)) + rational(coefficient)
        object.__setattr__(self, "statistics", axes)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "coefficients", MappingProxyType(
            {a: values[a] for a in indices if values.get(a, Q(0))}))

    def _new(self, coefficients):
        return MomentJet(self.statistics, self.order, coefficients, max_coefficients=len(self.indices))

    def _compatible(self, other):
        if not isinstance(other, MomentJet) or (self.statistics, self.order) != (other.statistics, other.order):
            raise ValueError("Moment jets must have the same ordered axes and total order")

    def constant(self, value):
        return self._new({(0,) * len(self.statistics): rational(value)})

    @property
    def mass(self):
        return self.coefficients.get((0,) * len(self.statistics), Q(0))

    def coefficient(self, alpha):
        return self.coefficients.get(_index(alpha, len(self.statistics), self.order), Q(0))

    def moment(self, alpha):
        """Derivative at zero; a raw moment before taking formal log."""
        alpha = _index(alpha, len(self.statistics), self.order)
        return self.coefficient(alpha) * _factorial(alpha)

    def truncate(self, order):
        integer(order, "order")
        if order > self.order:
            raise ValueError("Truncation cannot increase the jet order")
        return MomentJet(self.statistics, order,
                         ((a, c) for a, c in self.coefficients.items() if sum(a) <= order),
                         max_coefficients=len(self.indices))

    def __eq__(self, other):
        if not isinstance(other, MomentJet):
            return NotImplemented
        return ((self.statistics, self.order, self.coefficients)
                == (other.statistics, other.order, other.coefficients))

    def __add__(self, other):
        if not isinstance(other, MomentJet):
            other = self.constant(other)
        self._compatible(other)
        return self._new(tuple(self.coefficients.items()) + tuple(other.coefficients.items()))

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if not isinstance(other, MomentJet):
            coefficient = rational(other)
            return self._new((a, coefficient * c) for a, c in self.coefficients.items())
        self._compatible(other)
        return self._new((tuple(x + y for x, y in zip(a, b)), c * d)
                         for a, c in self.coefficients.items() for b, d in other.coefficients.items()
                         if sum(a) + sum(b) <= self.order)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, coefficient):
        return self * (1 / rational(coefficient))

    def __pow__(self, exponent):
        integer(exponent, "exponent")
        result, base = self.constant(1), self
        while exponent:
            if exponent & 1:
                result = result * base
            exponent >>= 1
            if exponent:
                base = base * base
        return result

    def reciprocal(self):
        """Formal reciprocal; this supplies no graph-algebra unit certificate."""
        if not self.mass:
            raise ValueError("A formal reciprocal requires a nonzero constant coefficient")
        remainder = self / self.mass - 1
        result = term = self.constant(1)
        for _ in range(self.order):
            term = term * -remainder
            result += term
        return result / self.mass

    def log(self):
        """Formal logarithm at exactly constant coefficient one."""
        if self.mass != 1:
            raise ValueError("Formal log requires constant coefficient one; normalize first")
        remainder = self - 1
        result, power = self.constant(0), self.constant(1)
        for degree in range(1, self.order + 1):
            power = power * remainder
            result += power * Q((-1) ** (degree + 1), degree)
        return result

    def exp(self):
        """Formal exponential at exactly constant coefficient zero."""
        if self.mass:
            raise ValueError("Formal exp requires constant coefficient zero")
        result = term = self.constant(1)
        for degree in range(1, self.order + 1):
            term = term * self / degree
            result += term
        return result

    def to_data(self):
        return {"axes": [axis.name for axis in self.statistics], "order": self.order,
                "coefficients": [{"alpha": list(a), "value": str(self.coefficient(a))}
                                 for a in self.indices]}


@dataclass(frozen=True)
class JetCertificate:
    """Coefficient enclosures derived from one weighted local certificate.

    Only extraction is certified here. Applying nonlinear formal operations
    to ``jet`` alone does not propagate these intervals.
    """
    jet: MomentJet
    local: LocalApproximation

    def interval(self, alpha):
        alpha = _index(alpha, len(self.jet.statistics), self.jet.order)
        center = self.jet.coefficient(alpha)
        error = self.local.error / _factorial(alpha)
        return Interval(center - error, center + error)

    def to_data(self):
        return {"axes": [axis.name for axis in self.jet.statistics], "order": self.jet.order,
                "radius": self.local.histogram.radius, "k": self.local.k,
                "source_error": str(self.local.error),
                "coefficients": [{"alpha": list(a), **self.interval(a).to_data()}
                                 for a in self.jet.indices]}


def certified_jet(value, statistics, order, epsilon="1e-8", max_coefficients=10000):
    """Enclose every exponential-moment coefficient through total ``order``.

    At R=max(radius) and k=max(1, order*max(growth_degree)), a source error
    delta bounds coefficient alpha by delta/alpha!. Custom statistic
    additivity and polynomial growth are caller-supplied global certificates.
    """
    axes = _axes(statistics)
    multi_indices(len(axes), order, max_coefficients)
    epsilon = rational(epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    radius = max(axis.radius for axis in axes)
    k = max(1, order * max(axis.growth_degree for axis in axes))
    local = as_element(value).approximate(radius, k, epsilon)
    if not isinstance(local, LocalApproximation):
        raise TypeError("The source must return a LocalApproximation")
    local = local.truncate(radius, k)
    if local.error > epsilon:
        raise ValueError("The source did not meet the requested error bound")
    jet = JointDistribution.from_histogram(local.histogram, axes).jet(order, max_coefficients)
    return JetCertificate(jet, local)
