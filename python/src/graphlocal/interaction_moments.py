"""Exact mixed moments from incidence couplings and covering cyclic words.

The subset mask records which edits have appeared at least once; repeated
appearances are retained. No graph isomorphism or edited-subset matrices are
used. Work still grows exponentially with the number of active defects.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from math import comb, factorial

from .edge_interactions import EdgeInteraction
from .graphs import BudgetExceeded, integer
from .graphs import distances


@dataclass(frozen=True)
class TreeInteractionLeading:
    selected_edges: int
    spanning_edges: int
    terminal_edges: int
    branching_factor: int
    order: int
    heat_coefficient: Q

    def to_data(self):
        return {"selected_edges": self.selected_edges, "spanning_edges": self.spanning_edges,
                "terminal_edges": self.terminal_edges, "branching_factor": self.branching_factor,
                "first_nonzero_order": self.order, "leading_heat_coefficient": str(self.heat_coefficient)}


def tree_interaction_leading(value):
    """Exact leading heat term for a nonempty deletion interaction on a tree.

    The minimal subtree spanning all selected edges has s edges, ell terminal
    edges and p=product_(internal v)(deg(v)-1)!. For k>=2 the leading term is
    (-1)^(k-ell)*p*t^(2s-ell)/(2s-ell-1)!, with the supplied normalization.
    A single cut starts with 2*t. This helper performs no spectral computation.
    """
    if not isinstance(value, EdgeInteraction):
        raise TypeError("Tree leading terms require a finite EdgeInteraction")
    g, edits = value.before, value.edits
    if not edits or any(sign != -1 for _, _, sign in edits):
        raise ValueError("A nonempty set of tree-edge deletions is required")
    if g.edges != g.n - 1 or len(distances(g, 0)) != g.n:
        raise ValueError("The ambient graph must be a connected tree")
    k = len(edits)
    if k == 1:
        return TreeInteractionLeading(1, 1, 1, 1, 1, 2 * value.scale)
    protected = {x for u, v, _ in edits for x in (u, v)}
    adjacent = [set(g.neighbors(u)) for u in range(g.n)]
    leaves = [u for u in range(g.n) if len(adjacent[u]) == 1 and u not in protected]
    while leaves:
        u = leaves.pop()
        if len(adjacent[u]) != 1 or u in protected:
            continue
        v = adjacent[u].pop()
        adjacent[v].remove(u)
        if len(adjacent[v]) == 1 and v not in protected:
            leaves.append(v)
    degrees = [len(a) for a in adjacent]
    s, ell = sum(degrees) // 2, sum(d == 1 for d in degrees)
    p = 1
    for d in degrees:
        if d >= 2:
            p *= factorial(d - 1)
    order = 2 * s - ell
    coefficient = value.scale * Q((-1) ** (k - ell) * p, factorial(order - 1))
    return TreeInteractionLeading(k, s, ell, p, order, coefficient)


@dataclass(frozen=True)
class InteractionMoments:
    laplacian: tuple
    returns: tuple
    degree_bound: int
    active_edits: tuple
    cross_moments: tuple
    work: int

    @property
    def first_nonzero_order(self):
        return next((j for j, value in enumerate(self.laplacian) if value), None)

    def to_data(self):
        first = self.first_nonzero_order
        return {"laplacian_moments": [str(x) for x in self.laplacian],
                "uniformized_moments": [str(x) for x in self.returns],
                "degree_bound": self.degree_bound,
                "active_edits": [list(e) for e in self.active_edits],
                "cross_moments": [[list(row) for row in matrix] for matrix in self.cross_moments],
                "first_nonzero_order_in_computed_range": first,
                "leading_heat_coefficient": (str((-1) ** first * self.laplacian[first] / factorial(first))
                                             if first is not None else None),
                "work_units": self.work}


def interaction_moments(value, order, max_work=2000000):
    """Return exact moments through order for a finite EdgeInteraction.

    Cross moments are b_i^T L^a b_j on the original graph, with each edge
    oriented from its stored first endpoint to its second. The cyclic-word
    formula supplies all repeated edit occurrences and the factor n/m.
    A missing first nonzero order means only that the computed prefix is zero.
    """
    if not isinstance(value, EdgeInteraction):
        raise TypeError("Incidence moments require a finite EdgeInteraction")
    integer(order, "order")
    integer(max_work, "max_work", 1)
    edits, g = value.active_edits, value.before
    k, degree, work = len(edits), value.degree_bound, 0

    def charge():
        nonlocal work
        work += 1
        if work > max_work:
            raise BudgetExceeded("Incidence moment calculation exceeds max_work")

    if not k or not order:
        zeros = (Q(0),) * (order + 1)
        return InteractionMoments(zeros, zeros, degree, edits, (), work)
    cross = [[[0] * k for _ in range(k)] for _ in range(order)]
    for j, (u, v, _) in enumerate(edits):
        vector = {u: 1, v: -1}
        for a in range(order):
            for i, (x, y, _) in enumerate(edits):
                cross[a][i][j] = vector.get(x, 0) - vector.get(y, 0)
            if a + 1 == order:
                break
            nxt = {}
            for x, c in vector.items():
                charge()
                nxt[x] = nxt.get(x, 0) + g.rows[x].bit_count() * c
                for y in g.neighbors(x):
                    charge()
                    nxt[y] = nxt.get(y, 0) - c
            vector = {x: c for x, c in nxt.items() if c}
    result = [Q(0)] * (order + 1)
    full = (1 << k) - 1
    for start in range(k):
        # States contain the used mask, last edit and total degree before
        # the closing gap. A new edit contributes one degree and its sign.
        layer = {(1 << start, start, 1): edits[start][2]}
        for m in range(1, order + 1):
            for (mask, last, used), coefficient in layer.items():
                if mask == full:
                    for a in range(order - used + 1):
                        charge()
                        n = used + a
                        result[n] += Q(n, m) * coefficient * cross[a][last][start]
            if m == order:
                break
            nxt = {}
            for (mask, last, used), coefficient in layer.items():
                if not coefficient:
                    continue
                for following in range(k):
                    for a in range(order - used):
                        charge()
                        c = coefficient * cross[a][last][following] * edits[following][2]
                        if c:
                            key = (mask | (1 << following), following, used + a + 1)
                            nxt[key] = nxt.get(key, 0) + c
            layer = {key: c for key, c in nxt.items() if c}
            if not layer:
                break
    scale = value.scale * value.reduction_sign
    result = tuple(scale * x for x in result)
    if degree:
        returns = tuple(sum((comb(j, n) * Q(-1, degree) ** n * result[n]
                             for n in range(j + 1)), Q(0)) for j in range(order + 1))
    else:
        returns = (Q(0),) * (order + 1)
    return InteractionMoments(result, returns, degree, edits,
                              tuple(tuple(tuple(row) for row in matrix) for matrix in cross), work)
