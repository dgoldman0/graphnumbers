"""Finite rooted histograms and exact bounds in the defining seminorms."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from types import MappingProxyType

from .graphs import Graph, IsoGraph, ball, distances, integer, local_product, rational


@dataclass(frozen=True, eq=False)
class LocalHistogram:
    """A finite local array, not by itself a certificate of global realizability."""
    radius: int
    values: object

    def __init__(self, radius, terms=()):
        object.__setattr__(self, "radius", integer(radius, "radius"))
        values = {}
        items = terms.items() if hasattr(terms, "items") else terms
        for key, coefficient in items:
            g = key.graph if isinstance(key, IsoGraph) else key
            if not isinstance(g, Graph) or not g.n:
                raise ValueError("Histogram keys must be nonempty rooted graphs")
            ds = distances(g, 0)
            if len(ds) != g.n or max(ds.values()) > radius:
                raise ValueError("Type must be connected and within the declared radius")
            key = IsoGraph(g, rooted=True)
            values[key] = values.get(key, Q(0)) + rational(coefficient)
        object.__setattr__(self, "values", MappingProxyType({g: c for g, c in values.items() if c}))

    @classmethod
    def from_graph(cls, g, radius, normalize=False):
        if normalize and not g.n:
            raise ValueError("The empty graph cannot be normalized")
        coefficient = Q(1, g.n) if normalize else Q(1)
        return cls(radius, ((ball(g, u, radius), coefficient) for u in range(g.n)))

    def __eq__(self, other):
        if not isinstance(other, LocalHistogram):
            return NotImplemented
        return self.radius == other.radius and self.values == other.values

    def __add__(self, other):
        if self.radius != other.radius:
            raise ValueError("Histogram radii must agree")
        return LocalHistogram(self.radius, list(self.values.items()) + list(other.values.items()))

    def __sub__(self, other):
        return self + other.scale(-1)

    def scale(self, coefficient):
        coefficient = rational(coefficient)
        return LocalHistogram(self.radius, ((g, coefficient * c) for g, c in self.values.items()))

    def multiply(self, other, max_vertices=10000):
        if self.radius != other.radius:
            raise ValueError("Histogram radii must agree")
        return LocalHistogram(self.radius,
                              ((local_product(g.graph, h.graph, self.radius, max_vertices), c * d)
                               for g, c in self.values.items() for h, d in other.values.items()))

    def truncate(self, radius):
        integer(radius, "radius")
        if radius > self.radius:
            raise ValueError("Truncation cannot increase radius")
        return LocalHistogram(radius, ((ball(g.graph, 0, radius), c)
                                       for g, c in self.values.items()))

    def norm(self, k=1):
        integer(k, "weight exponent")
        return sum((abs(c) * g.graph.n ** k for g, c in self.values.items()), Q(0))

    @property
    def mass(self):
        return sum(self.values.values(), Q(0))

    def evaluate(self, function):
        return sum((c * rational(function(g.graph)) for g, c in self.values.items()), Q(0))

    def to_data(self):
        return {"radius": self.radius,
                "terms": [{"rows": list(g.graph.rows), "coefficient": str(c)}
                          for g, c in sorted(self.values.items(), key=lambda x: x[0].graph.rows)]}

    @classmethod
    def from_data(cls, data):
        return cls(data["radius"], ((Graph(tuple(t["rows"])), rational(t["coefficient"]))
                                    for t in data["terms"]))


@dataclass(frozen=True)
class Interval:
    lower: Q
    upper: Q

    def __post_init__(self):
        object.__setattr__(self, "lower", rational(self.lower))
        object.__setattr__(self, "upper", rational(self.upper))
        if self.lower > self.upper:
            raise ValueError("Interval endpoints are reversed")

    @property
    def midpoint(self):
        return (self.lower + self.upper) / 2

    @property
    def radius(self):
        return (self.upper - self.lower) / 2

    def contains(self, value):
        return self.lower <= rational(value) <= self.upper

    def to_data(self):
        return {"lower": str(self.lower), "upper": str(self.upper),
                "midpoint": str(self.midpoint), "error_bound": str(self.radius)}


@dataclass(frozen=True)
class LocalApproximation:
    histogram: LocalHistogram
    k: int
    error: Q = Q(0)

    def __post_init__(self):
        integer(self.k, "weight exponent", 1)
        object.__setattr__(self, "error", rational(self.error))
        if self.error < 0:
            raise ValueError("Error bound must be nonnegative")

    def add(self, other):
        if self.k != other.k:
            raise ValueError("Weight exponents must agree")
        return LocalApproximation(self.histogram + other.histogram, self.k, self.error + other.error)

    def scale(self, coefficient):
        c = rational(coefficient)
        return LocalApproximation(self.histogram.scale(c), self.k, abs(c) * self.error)

    def multiply(self, other):
        if self.k != other.k:
            raise ValueError("Weight exponents must agree")
        error = (self.histogram.norm(self.k) * other.error
                 + other.histogram.norm(self.k) * self.error + self.error * other.error)
        return LocalApproximation(self.histogram.multiply(other.histogram), self.k, error)

    def truncate(self, radius, k=None):
        k = self.k if k is None else integer(k, "weight exponent", 1)
        if k > self.k:
            raise ValueError("A weaker certificate cannot imply a stronger weight")
        return LocalApproximation(self.histogram.truncate(radius), k, self.error)

    def project_degree(self, degree):
        """Project to a certified support, rejecting observable contradictions.

        The true local array is assumed supported on degree <= degree.
        Removed weighted mass must fit within this certificate's error.
        Under that assumption projection cannot increase the true error.
        """
        integer(degree, "degree_bound")
        removed = sum((abs(c) * key.graph.n ** self.k
                       for key, c in self.histogram.values.items()
                       if key.graph.max_degree > degree), Q(0))
        if removed > self.error:
            raise ValueError("Source approximation contradicts its degree bound")
        histogram = LocalHistogram(self.histogram.radius,
                                   ((key, c) for key, c in self.histogram.values.items()
                                    if key.graph.max_degree <= degree))
        return LocalApproximation(histogram, self.k, self.error)

    def to_data(self):
        return {"histogram": self.histogram.to_data(), "k": self.k, "error_bound": str(self.error)}


@dataclass(frozen=True)
class LocalObservable:
    """Caller certifies |function(B)| <= constant * |B|**k at the given radius."""
    radius: int
    k: int
    constant: Q
    function: object

    def __post_init__(self):
        integer(self.radius, "radius")
        integer(self.k, "weight exponent", 1)
        object.__setattr__(self, "constant", rational(self.constant))
        if self.constant < 0 or not callable(self.function):
            raise ValueError("A callable and nonnegative growth bound are required")

    def evaluate(self, approximation):
        local = approximation.truncate(self.radius, self.k)
        center = local.histogram.evaluate(self.function)
        error = self.constant * local.error
        return Interval(center - error, center + error)


def closed_walks(g, length):
    integer(length, "walk length")
    counts = [0] * g.n
    counts[0] = 1
    for _ in range(length):
        counts = [sum(counts[v] for v in g.neighbors(u)) for u in range(g.n)]
    return counts[0]


def walk_observable(length):
    integer(length, "walk length")
    return LocalObservable(length // 2, max(1, length), Q(1),
                           lambda g: closed_walks(g, length))


VERTICES = LocalObservable(0, 1, Q(1), lambda g: 1)
EDGES = LocalObservable(1, 1, Q(1, 2), lambda g: Q(g.rows[0].bit_count(), 2))
ISOLATED = LocalObservable(1, 1, Q(1), lambda g: int(g.n == 1))
