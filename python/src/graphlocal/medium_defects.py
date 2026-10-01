"""Single-edge cuts in infinite regular trees and the square lattice.

Every returned local histogram is exact. Finite witnesses have a buffer
of 2*r+1 around the central edge endpoints, so boundary effects cannot
enter the radius-r neighborhoods affected by the cut. Regular-tree local
histograms use symmetry directly and avoid constructing the full buffer.
"""
from fractions import Fraction as Q

from .defects import SparseEdgeDifference, apply_edge_edits
from .elements import Element
from .graphs import BudgetExceeded, ball, graph, integer
from .local import LocalHistogram


def _regular_ball(degree, radius, max_vertices):
    """Rooted regular-tree ball and one geodesic from its root."""
    children, edges, frontier = [[]], [], [0]
    for _ in range(radius):
        next_frontier = []
        for parent in frontier:
            for _ in range(degree if parent == 0 else degree - 1):
                if len(children) >= max_vertices:
                    raise BudgetExceeded("Regular-tree rooted ball exceeds max_vertices")
                child = len(children)
                children.append([])
                children[parent].append(child)
                edges.append((parent, child))
                next_frontier.append(child)
        frontier = next_frontier
    spine = [0]
    for _ in range(radius):
        spine.append(children[spine[-1]][0])
    return graph(len(children), edges), tuple(spine)


def _edge_centered_tree(degree, depth, max_vertices):
    """Two depth-deep (degree-1)-ary rooted trees joined at their roots."""
    if max_vertices < 2:
        raise BudgetExceeded("Regular-tree finite witness exceeds max_vertices")
    count, edges, frontier = 2, [(0, 1)], [0, 1]
    for _ in range(depth):
        next_frontier = []
        for parent in frontier:
            for _ in range(degree - 1):
                if count >= max_vertices:
                    raise BudgetExceeded("Regular-tree finite witness exceeds max_vertices")
                child = count
                count += 1
                edges.append((parent, child))
                next_frontier.append(child)
        frontier = next_frontier
    return graph(count, edges)


class InfiniteRegularTreeCut(Element):
    """Unnormalized correction after deleting one edge of a regular tree.

    At degree two this is the cut-line element. For degree d, put q=d-1.
    Radius r has r positive rooted types with coefficients 2*q**j,
    j=0..r-1, and one negative regular type with coefficient
    -2*sum(q**j). The total variation is 4*sum(q**j).

    ``max_vertices`` bounds each constructed rooted ball and each full
    finite witness separately. A local request can succeed when its much
    larger finite witness exceeds the budget.
    """
    edit_bound, mass, positive, variation_bound = Q(1), Q(0), False, None

    def __init__(self, degree, max_vertices=10000):
        self.degree_bound = integer(degree, "tree degree", 2)
        self.max_vertices = integer(max_vertices, "max_vertices", 1)

    def local(self, radius):
        integer(radius, "radius")
        if not radius:
            return LocalHistogram(0)
        regular, spine = _regular_ball(self.degree_bound, radius, self.max_vertices)
        terms, multiplicity, total = [], 2, 0
        for distance in range(radius):
            after, _ = apply_edge_edits(regular, [(spine[distance], spine[distance + 1], -1)])
            terms.append((ball(after, 0, radius), Q(multiplicity)))
            total += multiplicity
            multiplicity *= self.degree_bound - 1
        terms.append((regular, Q(-total)))
        return LocalHistogram(radius, terms)

    def finite_at_radius(self, radius):
        """Return a finite single-edge difference with this exact local array."""
        integer(radius, "radius")
        before = _edge_centered_tree(self.degree_bound, 2 * radius + 1, self.max_vertices)
        return SparseEdgeDifference(before, [(0, 1, -1)])


class SquareLatticeEdgeCut(Element):
    """Unnormalized correction for deleting {(0,0),(1,0)} in Z squared.

    Local data are computed inside an open rectangular buffer. The fixed
    finite-edge edit budget is one; there is no supplied global signed
    graph-measure variation bound.
    """
    degree_bound, edit_bound, mass, positive, variation_bound = 4, Q(1), Q(0), False, None

    def __init__(self, max_vertices=10000):
        self.max_vertices = integer(max_vertices, "max_vertices", 1)

    def finite_at_radius(self, radius):
        """Return a buffered finite lattice correction stable at radius r."""
        integer(radius, "radius")
        margin = 2 * radius + 1
        width, height = 2 * margin + 2, 2 * margin + 1
        count = width * height
        if count > self.max_vertices:
            raise BudgetExceeded("Square-lattice finite buffer exceeds max_vertices")
        edges = []
        for x in range(width):
            for y in range(height):
                vertex = x * height + y
                if x + 1 < width:
                    edges.append((vertex, vertex + height))
                if y + 1 < height:
                    edges.append((vertex, vertex + 1))
        before = graph(count, edges)
        first = margin * height + margin
        second = (margin + 1) * height + margin
        return SparseEdgeDifference(before, [(first, second, -1)])

    def local(self, radius):
        integer(radius, "radius")
        return LocalHistogram(0) if not radius else self.finite_at_radius(radius).local(radius)
