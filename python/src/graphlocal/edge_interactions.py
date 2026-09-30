"""Finite edit interactions and exact reduction for selected bridge edges."""
from fractions import Fraction as Q
from itertools import combinations

from .defects import SparseEdgeDifference, apply_edge_edits
from .elements import Element, Finite
from .graphs import BudgetExceeded, integer
from .local import LocalHistogram


def _component_labels(g):
    labels, count = {}, 0
    for root in range(g.n):
        if root in labels:
            continue
        labels[root] = count
        stack = [root]
        while stack:
            u = stack.pop()
            for v in g.neighbors(u):
                if v not in labels:
                    labels[v] = count
                    stack.append(v)
        count += 1
    return labels, count


def bridge_cut_reduction(before, edits):
    """Return (sign, retained_edits, applies) for a connected bridge-cut set.

    The components after deleting all k edges form a quotient tree exactly
    when every selected edge is a bridge. Keep its leaf-incident edges;
    the full interaction is (-1)**(k-l) times their interaction. Cycles
    inside uncut components are permitted. Other inputs are left unchanged.
    """
    after, edits = apply_edge_edits(before, edits)
    if not edits or any(sign != -1 for _, _, sign in edits):
        return 1, edits, False
    _, original_count = _component_labels(before)
    labels, count = _component_labels(after)
    if original_count != 1 or count != len(edits) + 1:
        return 1, edits, False
    degrees = [0] * count
    for u, v, _ in edits:
        degrees[labels[u]] += 1
        degrees[labels[v]] += 1
    retained = tuple(edit for edit in edits
                     if degrees[labels[edit[0]]] == 1 or degrees[labels[edit[1]]] == 1)
    return (-1) ** (len(edits) - len(retained)), retained, True


class EdgeInteraction(Element):
    """Connected inclusion-exclusion response of distinct finite edge edits.

    Equal to sum_S (-1)**(k-|S|) G_S for k>=1; the empty edit set is zero.
    Bridge reduction occurs before the subset work budget is applied.
    Default values are unnormalized. Mixed insertions and removals are valid,
    but only pure bridge deletions receive the structural reduction.
    """
    mass = Q(0)

    def __init__(self, before, edits, normalize=False, reduce_bridges=True, max_edits=10):
        integer(max_edits, "max_edits")
        _, self.edits = apply_edge_edits(before, edits)
        if normalize and not before.n:
            raise ValueError("The empty graph cannot be normalized")
        self.before, self.scale = before, Q(1, before.n) if normalize else Q(1)
        if reduce_bridges:
            sign, active, applies = bridge_cut_reduction(before, self.edits)
        else:
            sign, active, applies = 1, self.edits, False
        self.reduction_sign, self.active_edits, self.bridge_reduction_applies = sign, active, applies
        if len(active) > max_edits:
            raise BudgetExceeded("Active interaction edits exceed the subset budget")
        additions = [edit for edit in self.edits if edit[2] == 1]
        maximal, _ = apply_edge_edits(before, additions)
        self.degree_bound = maximal.max_degree
        count = len(active)
        self.edit_bound = count * 2 ** (count - 1) * self.scale if count else Q(0)
        self.moment_profile = ((Q(0),) * count + (2 ** count * self.scale,)
                               if count else (Q(0),))
        self.variation_bound = 2 ** count * before.n * self.scale if count else Q(0)
        self.positive = not count

    def _differences(self):
        count = len(self.active_edits)
        for size in range(1, count + 1):
            coefficient = self.scale * self.reduction_sign * (-1) ** (count - size)
            for edits in combinations(self.active_edits, size):
                yield coefficient, SparseEdgeDifference(self.before, edits)

    def local(self, radius):
        integer(radius, "radius")
        result = LocalHistogram(radius)
        for coefficient, difference in self._differences():
            result = result + difference.local(radius).scale(coefficient)
        return result

    def finite(self, max_vertices=10000):
        terms = []
        for coefficient, difference in self._differences():
            terms.extend((coefficient * c, g) for c, g in difference.finite(max_vertices).terms)
        return Finite(terms).finite(max_vertices)
