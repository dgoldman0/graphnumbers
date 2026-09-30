"""Two cuts of the line and their connected signed interaction."""
from fractions import Fraction as Q

from .defects import CutLineDefect, SparseEdgeDifference
from .elements import Element, Finite, Line
from .graphs import cycle, integer, path
from .local import LocalHistogram


class TwoCutLineDefect(Element):
    """Limit of two cycle-edge deletions enclosing segment_vertices vertices.

    If E is the single cut and L the normalized line, this is
    E + P_ell - ell*L. It has weighted edit budget two.
    """
    degree_bound, edit_bound, mass = 2, Q(2), Q(0)
    variation_bound, positive = None, False

    def __init__(self, segment_vertices):
        self.segment_vertices = integer(segment_vertices, "segment_vertices", 1)

    def local(self, radius):
        integer(radius, "radius")
        cut = CutLineDefect().local(radius)
        if radius <= self.segment_vertices // 2:
            return cut.scale(2)
        return (cut + Finite.from_graph(path(self.segment_vertices)).local(radius)
                - Line().local(radius).scale(self.segment_vertices))

    def interaction(self):
        return CutInteraction(self.segment_vertices)

    def finite_at_radius(self, radius):
        integer(radius, "radius")
        ell = self.segment_vertices
        n = max(3, ell + 2 * radius + 2)
        return SparseEdgeDifference(cycle(n), [(0, n - 1, -1), (ell - 1, ell, -1)])


class CutInteraction(Element):
    """TwoCutLineDefect(ell) - 2*CutLineDefect(), with edit budget four.

    Geometry vanishes through radius floor(ell/2); relative heat begins
    at order t**(2*ell). The full element is beyond finite signed measures.
    """
    degree_bound, edit_bound, mass = 2, Q(4), Q(0)
    variation_bound, positive = None, False

    def __init__(self, segment_vertices):
        self.segment_vertices = integer(segment_vertices, "segment_vertices", 1)

    def local(self, radius):
        integer(radius, "radius")
        if radius <= self.segment_vertices // 2:
            return LocalHistogram(radius)
        return (Finite.from_graph(path(self.segment_vertices)).local(radius)
                - Line().local(radius).scale(self.segment_vertices)
                - CutLineDefect().local(radius))

    def finite_at_radius(self, radius):
        together = TwoCutLineDefect(self.segment_vertices).finite_at_radius(radius)
        first, second = together.edits
        return (together - SparseEdgeDifference(together.before, [first])
                - SparseEdgeDifference(together.before, [second]))


def _cut_positions(positions):
    positions = tuple(positions)
    if any(isinstance(p, bool) or not isinstance(p, int) for p in positions):
        raise TypeError("Cut positions must be integers")
    if len(set(positions)) != len(positions):
        raise ValueError("Cut positions must be distinct")
    positions = tuple(sorted(positions))
    return tuple(p - positions[0] for p in positions) if positions else ()


class LineCutDefect(Element):
    """Several line cuts, specified by distinct integer edge positions.

    Translation and input order do not matter. If k cuts have consecutive
    gaps ell_i, their full response equals k*E + sum_i CutInteraction(ell_i).
    This identity holds for the completed graph elements, hence for every
    defined linear observable, and gives a uniform edit budget k.
    """
    mass = Q(0)

    def __init__(self, positions):
        self.positions = _cut_positions(positions)
        self.gaps = tuple(b - a for a, b in zip(self.positions, self.positions[1:]))
        self.edit_bound = Q(len(self.positions))
        self.degree_bound = 2 if self.positions else 0
        self.variation_bound = None if self.positions else Q(0)
        self.positive = not self.positions

    def local(self, radius):
        integer(radius, "radius")
        if not self.positions:
            return LocalHistogram(radius)
        result = CutLineDefect().local(radius).scale(len(self.positions))
        for gap in self.gaps:
            result = result + CutInteraction(gap).local(radius)
        return result

    def finite_at_radius(self, radius):
        integer(radius, "radius")
        span = self.positions[-1] if self.positions else 0
        n = max(3, span + 2 * radius + 3)
        edits = [(0, n - 1, -1)] if self.positions else []
        edits += [(p - 1, p, -1) for p in self.positions[1:]]
        return SparseEdgeDifference(cycle(n), edits)


def connected_cut_interaction(positions):
    """Exact inclusion-exclusion reduction for a nonempty set of line cuts.

    The k>=2 interaction depends only on the two extreme positions:
    (-1)**k * CutInteraction(max(positions)-min(positions)).
    No subset enumeration is performed. This identity is specific to the
    line-cut family; general graph edits can have different interactions.
    """
    positions = _cut_positions(positions)
    if not positions:
        raise ValueError("Connected interaction needs at least one cut")
    if len(positions) == 1:
        return CutLineDefect()
    result = CutInteraction(positions[-1])
    return -result if len(positions) % 2 else result
