# graph_integer.py
#
# GraphInteger: Grothendieck completion of (finite simple undirected graphs / iso, ⊔)
# with multiplication given by the Cartesian product (□).
#
# Representation is kept small by storing *connected components* as a signed multiset:
#   x =  ⊔_{k}  count_pos[k] * C_k   -   ⊔_{k}  count_neg[k] * C_k
# where each C_k is a connected *canonical* graph type (keyed by its canonical ID).
#
# Assumes canonical_graph.py provides (at least):
#   - CanonicalGraph  (constructed from adjacency matrix OR bitset-adj tuple)
#   - matrix_to_bitsets(M) -> tuple[int]
#   - bitsets_to_matrix(adj) -> list[list[int]]
#   - connected_components(adj) -> list[list[int]]
#   - induced_subgraph(adj, vertices) -> tuple[int]
#   - disjoint_union_from_components(list_of_component_adj_bitsets) -> tuple[int]
#

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple, Iterable, Optional, Any

import canonical_graph as cg 

# A connected component type key:
# (n, enc) where enc is the canonical upper-triangle encoding tuple
CompKey = Tuple[int, Tuple[int, ...]]

# Global registry for component representatives (canonical, connected)
# key -> canonical bitset adjacency (tuple[int]) for that component type
_COMPONENT_REP_ADJ: Dict[CompKey, Tuple[int, ...]] = {}


def _register_component_rep(key: CompKey, adj: Tuple[int, ...]) -> None:
    """Store a canonical representative adjacency for this connected component type."""
    if key not in _COMPONENT_REP_ADJ:
        _COMPONENT_REP_ADJ[key] = adj


def _key_from_canonical_component(comp: "cg.CanonicalGraph") -> CompKey:
    """Create a hashable key for a connected canonical component."""
    key: CompKey = (comp.n, comp.enc)
    _register_component_rep(key, comp.adj)
    return key


def _decompose_to_component_multiset(graph: Any, seed_leaf_budget: int = 1500) -> Dict[CompKey, int]:
    """
    Convert a (possibly disconnected) graph into a multiset of connected canonical component keys.
    Input can be: CanonicalGraph, adjacency matrix, or bitset-adj tuple.
    """
    if isinstance(graph, cg.CanonicalGraph):
        adj = graph.adj
    elif isinstance(graph, tuple) and all(isinstance(x, int) for x in graph):
        adj = graph
    else:
        adj = cg.matrix_to_bitsets(graph)

    counts: Dict[CompKey, int] = {}
    for vs in cg.connected_components(adj):
        sub_adj = cg.induced_subgraph(adj, vs)  # relabeled 0..k-1
        comp = cg.CanonicalGraph(sub_adj, seed_leaf_budget=seed_leaf_budget)  # canonicalize component
        key = _key_from_canonical_component(comp)
        counts[key] = counts.get(key, 0) + 1

    return counts


def _cancel_counts(pos: Dict[CompKey, int], neg: Dict[CompKey, int]) -> None:
    """In-place cancellation of common component types between pos and neg."""
    common = set(pos.keys()) & set(neg.keys())
    for k in common:
        m = min(pos[k], neg[k])
        pos[k] -= m
        neg[k] -= m
        if pos[k] == 0:
            del pos[k]
        if neg[k] == 0:
            del neg[k]


def _disjoint_union_from_counts(counts: Dict[CompKey, int]) -> Tuple[int, ...]:
    """Rebuild a disjoint union adjacency (bitsets) from component counts (deterministic order)."""
    comps = []
    for key in sorted(counts.keys()):
        rep = _COMPONENT_REP_ADJ[key]
        for _ in range(counts[key]):
            comps.append(rep)
    return cg.disjoint_union_from_components(comps)


def _cartesian_product_adj(adjG: Tuple[int, ...], adjH: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Cartesian product G □ H:
      V = V(G) x V(H), index(u,v) = u*nH + v
      Edges: (u,v)~(u',v) if uu' in E(G); (u,v)~(u,v') if vv' in E(H)
    Both graphs assumed simple undirected (no loops), but code tolerates diagonal bits being 0.
    """
    nG = len(adjG)
    nH = len(adjH)
    N = nG * nH
    out = [0] * N

    # Precompute neighbor lists (bitset -> list) for simplicity
    nbrsG = []
    for u in range(nG):
        b = adjG[u] & ~(1 << u)
        lst = []
        while b:
            lsb = b & -b
            x = lsb.bit_length() - 1
            lst.append(x)
            b -= lsb
        nbrsG.append(lst)

    nbrsH = []
    for v in range(nH):
        b = adjH[v] & ~(1 << v)
        lst = []
        while b:
            lsb = b & -b
            x = lsb.bit_length() - 1
            lst.append(x)
            b -= lsb
        nbrsH.append(lst)

    for u in range(nG):
        for v in range(nH):
            idx = u * nH + v
            bits = 0

            # Move in G (change u, fixed v)
            for up in nbrsG[u]:
                bits |= 1 << (up * nH + v)

            # Move in H (fixed u, change v)
            for vp in nbrsH[v]:
                bits |= 1 << (u * nH + vp)

            out[idx] = bits

    return tuple(out)


@lru_cache(maxsize=None)
def _product_component_key(k1: CompKey, k2: CompKey, seed_leaf_budget: int = 1500) -> CompKey:
    """
    Cached product of two connected component types (by key) -> connected component type key of (C1 □ C2).
    For connected simple graphs, Cartesian product is connected.
    """
    adj1 = _COMPONENT_REP_ADJ[k1]
    adj2 = _COMPONENT_REP_ADJ[k2]
    prod_adj = _cartesian_product_adj(adj1, adj2)

    prod_canon = cg.CanonicalGraph(prod_adj, seed_leaf_budget=seed_leaf_budget)
    keyP = _key_from_canonical_component(prod_canon)
    return keyP


def _add_counts(a: Dict[CompKey, int], b: Dict[CompKey, int]) -> Dict[CompKey, int]:
    """Return multiset-sum of two count dicts."""
    out = dict(a)
    for k, c in b.items():
        out[k] = out.get(k, 0) + c
    # drop zeros defensively
    out = {k: c for k, c in out.items() if c}
    return out


def _scale_counts(a: Dict[CompKey, int], factor: int) -> Dict[CompKey, int]:
    if factor == 1:
        return dict(a)
    if factor == 0:
        return {}
    return {k: c * factor for k, c in a.items()}


@dataclass(frozen=True)
class GraphInteger:
    """
    Element of the Grothendieck group completion of graph isomorphism classes under disjoint union.
    Stored as reduced signed multiset of connected component types.
    """
    pos: Dict[CompKey, int]
    neg: Dict[CompKey, int]
    seed_leaf_budget: int = 1500  # passed through to canonicalization/product

    # -----------------------------
    # Constructors
    # -----------------------------

    @staticmethod
    def from_graphs(pos_graph: Any = None, neg_graph: Any = None, *, seed_leaf_budget: int = 1500) -> "GraphInteger":
        """
        Build a GraphInteger from a pair (A, B) representing A - B.
        Inputs can be CanonicalGraph, adjacency matrix, or bitset adjacency.
        """
        if pos_graph is None:
            pos_counts = {}
        else:
            pos_counts = _decompose_to_component_multiset(pos_graph, seed_leaf_budget=seed_leaf_budget)

        if neg_graph is None:
            neg_counts = {}
        else:
            neg_counts = _decompose_to_component_multiset(neg_graph, seed_leaf_budget=seed_leaf_budget)

        # reduce
        _cancel_counts(pos_counts, neg_counts)
        return GraphInteger(pos_counts, neg_counts, seed_leaf_budget=seed_leaf_budget)

    @staticmethod
    def zero(*, seed_leaf_budget: int = 1500) -> "GraphInteger":
        return GraphInteger({}, {}, seed_leaf_budget=seed_leaf_budget)

    @staticmethod
    def one(*, seed_leaf_budget: int = 1500) -> "GraphInteger":
        """
        Multiplicative identity for Cartesian product is K1.
        """
        K1 = cg.CanonicalGraph([[0]], seed_leaf_budget=seed_leaf_budget)
        key = _key_from_canonical_component(K1)
        return GraphInteger({key: 1}, {}, seed_leaf_budget=seed_leaf_budget)

    # -----------------------------
    # Internal normalization
    # -----------------------------

    def _normalized(self, pos: Dict[CompKey, int], neg: Dict[CompKey, int]) -> "GraphInteger":
        pos = {k: c for k, c in pos.items() if c}
        neg = {k: c for k, c in neg.items() if c}
        _cancel_counts(pos, neg)
        return GraphInteger(pos, neg, seed_leaf_budget=self.seed_leaf_budget)

    # -----------------------------
    # Basic queries
    # -----------------------------

    def is_zero(self) -> bool:
        return not self.pos and not self.neg

    def support_size(self) -> int:
        """Number of distinct component types appearing with nonzero coefficient."""
        return len(set(self.pos.keys()) | set(self.neg.keys()))

    def total_vertices_pos(self) -> int:
        return sum(k[0] * c for k, c in self.pos.items())

    def total_vertices_neg(self) -> int:
        return sum(k[0] * c for k, c in self.neg.items())

    # -----------------------------
    # Conversion back to graphs (on demand)
    # -----------------------------

    def as_graph_pair(self, *, canonical_output: bool = True) -> Tuple["cg.CanonicalGraph", "cg.CanonicalGraph"]:
        """
        Rebuild (A, B) where self = A - B (as disjoint unions of connected components).
        This may be expensive if counts are huge; only do when needed.
        """
        A_adj = _disjoint_union_from_counts(self.pos)
        B_adj = _disjoint_union_from_counts(self.neg)
        if canonical_output:
            return cg.CanonicalGraph(A_adj, seed_leaf_budget=self.seed_leaf_budget), cg.CanonicalGraph(B_adj, seed_leaf_budget=self.seed_leaf_budget)
        return cg.CanonicalGraph(A_adj, seed_leaf_budget=self.seed_leaf_budget), cg.CanonicalGraph(B_adj, seed_leaf_budget=self.seed_leaf_budget)

    # -----------------------------
    # Addition / subtraction
    # -----------------------------

    def __neg__(self) -> "GraphInteger":
        return GraphInteger(dict(self.neg), dict(self.pos), seed_leaf_budget=self.seed_leaf_budget)

    def __add__(self, other: "GraphInteger") -> "GraphInteger":
        if self.seed_leaf_budget != other.seed_leaf_budget:
            # keep left's setting as default
            pass
        pos = _add_counts(self.pos, other.pos)
        neg = _add_counts(self.neg, other.neg)
        return self._normalized(pos, neg)

    def __sub__(self, other: "GraphInteger") -> "GraphInteger":
        # (A - B) - (C - D) = (A + D) - (B + C)
        pos = _add_counts(self.pos, other.neg)
        neg = _add_counts(self.neg, other.pos)
        return self._normalized(pos, neg)

    # -----------------------------
    # Multiplication: Cartesian product
    # -----------------------------

    def __mul__(self, other: "GraphInteger") -> "GraphInteger":
        """
        Distributive product:
          (A - B) * (C - D) = (A□C + B□D) - (A□D + B□C)
        where □ distributes over disjoint union, and we store components.
        """
        seed = self.seed_leaf_budget

        pos_out: Dict[CompKey, int] = {}
        neg_out: Dict[CompKey, int] = {}

        def add_product_terms(left: Dict[CompKey, int], right: Dict[CompKey, int], out: Dict[CompKey, int], sign: int):
            """
            Add contributions from (⊔ left_i) □ (⊔ right_j) into out with multiplicities.
            sign is +1 to add, -1 to subtract into the chosen out dict.
            """
            if not left or not right:
                return
            for k1, c1 in left.items():
                for k2, c2 in right.items():
                    kp = _product_component_key(k1, k2, seed_leaf_budget=seed)
                    out[kp] = out.get(kp, 0) + sign * (c1 * c2)

        # Positive terms: A□C and B□D
        add_product_terms(self.pos, other.pos, pos_out, +1)
        add_product_terms(self.neg, other.neg, pos_out, +1)

        # Negative terms: A□D and B□C
        add_product_terms(self.pos, other.neg, neg_out, +1)
        add_product_terms(self.neg, other.pos, neg_out, +1)

        # Clean up any negative counts due to potential later cancellations (shouldn't happen, but safe)
        pos_out = {k: c for k, c in pos_out.items() if c > 0}
        neg_out = {k: c for k, c in neg_out.items() if c > 0}

        return self._normalized(pos_out, neg_out)

    # -----------------------------
    # Optional: scalar multiply
    # -----------------------------

    def scale(self, m: int) -> "GraphInteger":
        if m == 0:
            return GraphInteger.zero(seed_leaf_budget=self.seed_leaf_budget)
        if m > 0:
            return self._normalized(_scale_counts(self.pos, m), _scale_counts(self.neg, m))
        # negative scalar flips sign
        return (-self).scale(-m)

    # -----------------------------
    # Pretty printing
    # -----------------------------

    def __repr__(self) -> str:
        def fmt(counts: Dict[CompKey, int]) -> str:
            items = []
            for (n, enc), c in sorted(counts.items()):
                items.append(f"{c}*C(n={n})")
            return " ⊔ ".join(items) if items else "0"

        return f"GraphInteger(+[{fmt(self.pos)}] - [{fmt(self.neg)}])"
