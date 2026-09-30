"""Exact finite combinations, lazy algebra expressions, and effective limits."""
from __future__ import annotations

from fractions import Fraction as Q

from .graphs import (BudgetExceeded, Graph, IsoGraph, cartesian, components, cycle,
                     graph, induced, integer, path, rational)
from .local import LocalApproximation, LocalHistogram


class ExactLocalUnavailable(ValueError):
    pass


def moment_profile(value):
    """Return certified falling-factorial bounds on relative return moments.

    Coefficient A[r] means |d_j| <= sum_r A[r]*(j)_r/D**r for every
    admissible positive degree bound D. At degree zero, |mass| <= A[0].
    These bounds concern spectral moments, not global graph-measure variation.
    """
    profile = value.moment_profile
    if profile is None:
        if value.edit_bound is not None:
            profile = (Q(0), 2 * rational(value.edit_bound))
        elif value.variation_bound is not None:
            profile = (rational(value.variation_bound),)
        else:
            return None
    profile = tuple(rational(a) for a in profile)
    if not profile or any(a < 0 for a in profile):
        raise ValueError("Moment profile must contain nonnegative rational bounds")
    while len(profile) > 1 and not profile[-1]:
        profile = profile[:-1]
    return profile


def _profile_add(left, right):
    a, b = moment_profile(left), moment_profile(right)
    if a is None or b is None:
        return None
    return tuple((a[i] if i < len(a) else Q(0)) + (b[i] if i < len(b) else Q(0))
                 for i in range(max(len(a), len(b))))


def _profile_product(left, right):
    a, b = moment_profile(left), moment_profile(right)
    if a is None or b is None:
        return None
    result = [Q(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i + j] += x * y
    return tuple(result)


def _request(radius, k, epsilon):
    integer(radius, "radius")
    integer(k, "weight exponent", 1)
    epsilon = rational(epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return epsilon


def exp_bracket(x, max_terms=512):
    """Rational lower and upper bounds for exp(x), x >= 0."""
    x = rational(x)
    integer(max_terms, "max_terms", 1)
    if x < 0:
        raise ValueError("exp_bracket requires a nonnegative argument")
    term, total = Q(1), Q(1)
    for n in range(max_terms + 1):
        if n + 2 > x:
            tail = term * x / (n + 1) / (1 - x / (n + 2))
            if tail <= total:
                return total, total + tail
        term *= x / (n + 1)
        total += term
    raise BudgetExceeded("Exponential majorant exceeds max_terms")


class Element:
    """Effective-element protocol.

    Subclasses supply rigorous norm/approximation bounds. Optional degree,
    variation, positivity and mass metadata are mathematical certificates.
    They are inherited conservatively by the built-in expression operations.
    An optional edit_bound additionally certifies the relative-heat moment
    and rank bounds documented in defects.py and the sparse-defect proof note.
    """
    degree_bound = None
    variation_bound = None
    edit_bound = None
    moment_profile = None
    positive = False
    mass = None

    def local(self, radius):
        raise ExactLocalUnavailable("Use approximate(radius, k, epsilon) for this element")

    def norm_bound(self, radius, k):
        integer(radius, "radius")
        integer(k, "weight exponent", 1)
        return self.local(radius).norm(k)

    def approximate(self, radius, k=1, epsilon="0.000001"):
        _request(radius, k, epsilon)
        return LocalApproximation(self.local(radius), k)

    def finite(self, max_vertices=10000):
        raise ExactLocalUnavailable("This expression has no supplied finite representation")

    def __add__(self, other):
        return Sum(self, as_element(other))

    def __radd__(self, other):
        return as_element(other) + self

    def __neg__(self):
        return Scale(self, Q(-1))

    def __sub__(self, other):
        return self + -as_element(other)

    def __rsub__(self, other):
        return as_element(other) + -self

    def __mul__(self, other):
        if isinstance(other, Element):
            return Product(self, other)
        return Scale(self, rational(other))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, scalar):
        return Scale(self, 1 / rational(scalar))

    def __pow__(self, exponent):
        integer(exponent, "exponent")
        result, base = Finite.scalar(1), self
        while exponent:
            if exponent & 1:
                result = result * base
            exponent >>= 1
            if exponent:
                base = base * base
        return result


def as_element(value):
    return value if isinstance(value, Element) else Finite.scalar(value)


class Finite(Element):
    def __init__(self, terms=()):
        values = {}
        for coefficient, g in terms:
            c = rational(coefficient)
            if not c:
                continue
            for component in components(g):
                key = IsoGraph(component)
                values[key] = values.get(key, Q(0)) + c
        self.terms = tuple((c, key.graph) for key, c in values.items() if c)
        self.degree_bound = max((g.max_degree for _, g in self.terms), default=0)
        self.variation_bound = sum((abs(c) * g.n for c, g in self.terms), Q(0))
        self.mass = sum((c * g.n for c, g in self.terms), Q(0))
        self.positive = all(c >= 0 for c, _ in self.terms)
        self.edit_bound = Q(0) if not self.terms else None

    @classmethod
    def from_graph(cls, value, normalize=False):
        if normalize and not value.n:
            raise ValueError("The empty graph cannot be normalized")
        return cls([(Q(1, value.n) if normalize else Q(1), value)])

    @classmethod
    def scalar(cls, value):
        return cls([(rational(value), graph(1))])

    def local(self, radius):
        integer(radius, "radius")
        result = LocalHistogram(radius)
        for c, g in self.terms:
            result = result + LocalHistogram.from_graph(g, radius).scale(c)
        return result

    def finite(self, max_vertices=10000):
        integer(max_vertices, "max_vertices", 1)
        if any(g.n > max_vertices for _, g in self.terms):
            raise BudgetExceeded("Finite representative exceeds max_vertices")
        return self

    def __eq__(self, other):
        if not isinstance(other, Finite):
            return NotImplemented
        return {IsoGraph(g): c for c, g in self.terms} == {IsoGraph(g): c for c, g in other.terms}

    def to_data(self):
        return {"terms": [{"coefficient": str(c), "rows": list(g.rows)}
                          for c, g in sorted(self.terms, key=lambda term: term[1].rows)]}

    @classmethod
    def from_data(cls, data):
        return cls((rational(t["coefficient"]), Graph(tuple(t["rows"]))) for t in data["terms"])


class Line(Element):
    degree_bound, variation_bound, mass, positive = 2, Q(1), Q(1), True

    def local(self, radius):
        integer(radius, "radius")
        g = path(2 * radius + 1)
        rooted = induced(g, [radius] + [u for u in range(g.n) if u != radius])
        return LocalHistogram(radius, [(rooted, Q(1))])

    def finite_at_radius(self, radius):
        integer(radius, "radius")
        return Finite.scalar(1) if radius == 0 else Finite.from_graph(cycle(2 * radius + 2), True)


class Sum(Element):
    def __init__(self, left, right):
        self.left, self.right = left, right
        self.degree_bound = (None if None in (left.degree_bound, right.degree_bound)
                             else max(left.degree_bound, right.degree_bound))
        self.variation_bound = (None if None in (left.variation_bound, right.variation_bound)
                                else left.variation_bound + right.variation_bound)
        self.mass = None if None in (left.mass, right.mass) else left.mass + right.mass
        self.positive = left.positive and right.positive
        self.edit_bound = (None if None in (left.edit_bound, right.edit_bound)
                           else left.edit_bound + right.edit_bound)
        self.moment_profile = _profile_add(left, right)

    def local(self, radius):
        return self.left.local(radius) + self.right.local(radius)

    def norm_bound(self, radius, k):
        return self.left.norm_bound(radius, k) + self.right.norm_bound(radius, k)

    def approximate(self, radius, k=1, epsilon="0.000001"):
        eps = _request(radius, k, epsilon)
        return self.left.approximate(radius, k, eps / 2).add(self.right.approximate(radius, k, eps / 2))

    def finite(self, max_vertices=10000):
        return Finite(self.left.finite(max_vertices).terms + self.right.finite(max_vertices).terms)


class Scale(Element):
    def __init__(self, value, coefficient):
        self.value, self.coefficient = value, rational(coefficient)
        c = self.coefficient
        self.degree_bound = value.degree_bound if c else 0
        self.variation_bound = abs(c) * value.variation_bound if value.variation_bound is not None else (Q(0) if not c else None)
        self.mass = c * value.mass if value.mass is not None else (Q(0) if not c else None)
        self.positive = not c or (c > 0 and value.positive)
        self.edit_bound = (abs(c) * value.edit_bound if value.edit_bound is not None
                           else (Q(0) if not c else None))
        profile = moment_profile(value)
        self.moment_profile = ((Q(0),) if not c else
                               None if profile is None else tuple(abs(c) * a for a in profile))

    def local(self, radius):
        return self.value.local(radius).scale(self.coefficient) if self.coefficient else LocalHistogram(radius)

    def norm_bound(self, radius, k):
        integer(radius, "radius")
        integer(k, "weight exponent", 1)
        return abs(self.coefficient) * self.value.norm_bound(radius, k) if self.coefficient else Q(0)

    def approximate(self, radius, k=1, epsilon="0.000001"):
        eps = _request(radius, k, epsilon)
        if not self.coefficient:
            return LocalApproximation(LocalHistogram(radius), k)
        return self.value.approximate(radius, k, eps / abs(self.coefficient)).scale(self.coefficient)

    def finite(self, max_vertices=10000):
        if not self.coefficient:
            return Finite()
        return Finite((self.coefficient * c, g) for c, g in self.value.finite(max_vertices).terms)


class Product(Element):
    def __init__(self, left, right):
        self.left, self.right = left, right
        self.degree_bound = (None if None in (left.degree_bound, right.degree_bound)
                             else left.degree_bound + right.degree_bound)
        self.variation_bound = (None if None in (left.variation_bound, right.variation_bound)
                                else left.variation_bound * right.variation_bound)
        self.mass = None if None in (left.mass, right.mass) else left.mass * right.mass
        self.positive = left.positive and right.positive
        edit_bounds = []
        if left.edit_bound is not None and right.variation_bound is not None:
            edit_bounds.append(left.edit_bound * right.variation_bound)
        if right.edit_bound is not None and left.variation_bound is not None:
            edit_bounds.append(right.edit_bound * left.variation_bound)
        self.edit_bound = min(edit_bounds) if edit_bounds else None
        self.moment_profile = _profile_product(left, right)

    def local(self, radius):
        return self.left.local(radius).multiply(self.right.local(radius))

    def norm_bound(self, radius, k):
        return self.left.norm_bound(radius, k) * self.right.norm_bound(radius, k)

    def approximate(self, radius, k=1, epsilon="0.000001"):
        eps = _request(radius, k, epsilon)
        a, b = self.left.norm_bound(radius, k), self.right.norm_bound(radius, k)
        delta = min(Q(1), eps / (a + b + 3))
        left = self.left.approximate(radius, k, delta)
        right = self.right.approximate(radius, k, delta)
        return left.multiply(right)

    def finite(self, max_vertices=10000):
        a, b = self.left.finite(max_vertices), self.right.finite(max_vertices)
        return Finite((c * d, cartesian(g, h, max_vertices))
                      for c, g in a.terms for d, h in b.terms)


class Exponential(Element):
    """Entire algebra exponential; this is distinct from a graph heat operator."""
    def __init__(self, value, max_terms=128):
        self.value = as_element(value)
        self.max_terms = integer(max_terms, "max_terms", 1)
        self.positive = self.value.positive

    def norm_bound(self, radius, k):
        return exp_bracket(self.value.norm_bound(radius, k))[1]

    def approximate(self, radius, k=1, epsilon="0.000001"):
        eps = _request(radius, k, epsilon)
        bound = self.value.norm_bound(radius, k)
        lipschitz = exp_bracket(bound + 1)[1]
        delta = min(Q(1), eps / (2 * lipschitz))
        approx = self.value.approximate(radius, k, delta)
        x = approx.histogram
        z = x.norm(k)
        term = total = LocalHistogram(radius, [(graph(1), Q(1))])
        scalar_term = Q(1)
        for n in range(self.max_terms + 1):
            if n + 2 > z:
                tail = scalar_term * z / (n + 1) / (1 - z / (n + 2))
                error = tail + lipschitz * approx.error
                if error <= eps:
                    return LocalApproximation(total, k, error)
            if n == self.max_terms:
                break
            term = term.multiply(x).scale(Q(1, n + 1))
            total = total + term
            scalar_term *= z / (n + 1)
        raise BudgetExceeded("Algebra exponential exceeds max_terms")


def exp(value, max_terms=128):
    return Exponential(value, max_terms)


def polynomial(value, coefficients):
    """Coefficients are in ascending order: c[0] + c[1] X + ... ."""
    value, result = as_element(value), Finite.scalar(0)
    for c in reversed(tuple(coefficients)):
        result = result * value + rational(c)
    return result


def polynomial_derivative(value, coefficients, direction=1):
    coeffs = tuple(rational(c) for c in coefficients)
    return polynomial(value, (i * c for i, c in enumerate(coeffs) if i)) * as_element(direction)


def exp_derivative(value, direction=1):
    return exp(value) * as_element(direction)
