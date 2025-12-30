# graph_integer.py
"""
GraphInteger ring using Canon*(.) from canonstar.py

- Elements are reduced signed multisets of Canon* CONNECTED components.
- Addition = disjoint union (multiset add) with reduction (cancellation).
- Multiplication = Cartesian product, distributed over components, with Canon*
  applied to each connected product component.

Norm (UNCHANGED FORMULA):
  ||(A,B)|| = |v(A)-v(B)| + sum_{n>=1} 2^{-n} |a_n - b_n|
where a_n,b_n are the upper-triangle encodings of Canon*(A), Canon*(B),
with infinite zero-padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import canonstar as cs


CompKey = Tuple[int, Tuple[int, ...]]  # (n, enc) of a connected Canon* component

_COMPONENT_ADJ: Dict[CompKey, Tuple[int, ...]] = {}
_PRODUCT_CACHE: Dict[Tuple[CompKey, CompKey], CompKey] = {}


def _register_connected_component(adj_bitrows: Sequence[int]) -> CompKey:
    """
    Canon* a CONNECTED graph and register its representative adjacency.
    """
    g = cs.canonstar(adj_bitrows)
    # For connected input, canonstar returns single component key:
    key = (g.n, g.enc)
    if key not in _COMPONENT_ADJ:
        _COMPONENT_ADJ[key] = g.adj
    return key


def _cancel(pos: Dict[CompKey, int], neg: Dict[CompKey, int]) -> Tuple[Dict[CompKey, int], Dict[CompKey, int]]:
    for k in list(set(pos).intersection(neg)):
        c = min(pos[k], neg[k])
        if c:
            pos[k] -= c
            neg[k] -= c
            if pos[k] == 0:
                del pos[k]
            if neg[k] == 0:
                del neg[k]
    return pos, neg


def _add_counts(a: Dict[CompKey, int], b: Dict[CompKey, int]) -> Dict[CompKey, int]:
    out = dict(a)
    for k, v in b.items():
        if v:
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


def _mul_component_keys(a: CompKey, b: CompKey) -> CompKey:
    """
    Cartesian product of two connected components -> connected component key.
    Cached (commutative).
    """
    if a <= b:
        pair = (a, b)
    else:
        pair = (b, a)

    if pair in _PRODUCT_CACHE:
        return _PRODUCT_CACHE[pair]

    adjA = _COMPONENT_ADJ[a]
    adjB = _COMPONENT_ADJ[b]
    prod = cs.cartesian_product_bitrows([adjA, adjB])  # tuple-lex order

    # Canon* the product (connected) and register
    key = _register_connected_component(prod)
    _PRODUCT_CACHE[pair] = key
    return key


def _mul_nonneg(A: Dict[CompKey, int], B: Dict[CompKey, int]) -> Dict[CompKey, int]:
    if not A or not B:
        return {}
    out: Dict[CompKey, int] = {}
    for ka, ca in A.items():
        for kb, cb in B.items():
            kp = _mul_component_keys(ka, kb)
            out[kp] = out.get(kp, 0) + ca * cb
    return {k: v for k, v in out.items() if v}


def _build_union_from_counts(counts: Dict[CompKey, int]) -> Tuple[int, Tuple[int, ...]]:
    """
    Build the Canon* encoding of the disjoint union of connected components in `counts`.
    Because Canon* for disconnected graphs is defined as sorted block-diagonal union,
    we can build the union directly in that canonical order.
    """
    if not counts:
        return 0, ()
    blocks: List[List[int]] = []
    for key in sorted(counts.keys()):
        adj = _COMPONENT_ADJ[key]
        for _ in range(counts[key]):
            blocks.append(list(adj))
    union = cs.disjoint_union_bitrows(blocks)
    return len(union), cs.encode_upper_triangle(union)


def _weighted_bit_l1(encA: Tuple[int, ...], encB: Tuple[int, ...]) -> float:
    la, lb = len(encA), len(encB)
    m = la if la < lb else lb
    w = 0.5
    s = 0.0

    for i in range(m):
        if encA[i] ^ encB[i]:
            s += w
        w *= 0.5

    if la > lb:
        for i in range(lb, la):
            if encA[i]:
                s += w
            w *= 0.5
    elif lb > la:
        for i in range(la, lb):
            if encB[i]:
                s += w
            w *= 0.5

    return s


@dataclass
class GraphInteger:
    pos: Dict[CompKey, int]
    neg: Dict[CompKey, int]

    @staticmethod
    def zero() -> "GraphInteger":
        return GraphInteger({}, {})

    @staticmethod
    def from_graph(adj: Sequence[Sequence[int]] | Sequence[int]) -> "GraphInteger":
        """
        Build (G,0) where G is a finite simple graph.
        Stored as a multiset of Canon* connected components.
        """
        bitrows = cs.to_bitrows(adj)
        comps = cs.connected_components(bitrows)

        counts: Dict[CompKey, int] = {}
        for comp in comps:
            sub = cs.induced_subgraph(bitrows, comp)     # connected
            key = _register_connected_component(sub)      # Canon* + register
            counts[key] = counts.get(key, 0) + 1
        return GraphInteger(counts, {})

    def reduced(self) -> "GraphInteger":
        p, n = _cancel(dict(self.pos), dict(self.neg))
        return GraphInteger(p, n)

    def __add__(self, other: "GraphInteger") -> "GraphInteger":
        p = _add_counts(self.pos, other.pos)
        n = _add_counts(self.neg, other.neg)
        return GraphInteger(p, n).reduced()

    def __sub__(self, other: "GraphInteger") -> "GraphInteger":
        p = _add_counts(self.pos, other.neg)
        n = _add_counts(self.neg, other.pos)
        return GraphInteger(p, n).reduced()

    def __neg__(self) -> "GraphInteger":
        return GraphInteger(dict(self.neg), dict(self.pos))

    def __mul__(self, other: "GraphInteger") -> "GraphInteger":
        # (A-B)(C-D) = (AC + BD) - (AD + BC)
        AC = _mul_nonneg(self.pos, other.pos)
        BD = _mul_nonneg(self.neg, other.neg)
        AD = _mul_nonneg(self.pos, other.neg)
        BC = _mul_nonneg(self.neg, other.pos)

        p = _add_counts(AC, BD)
        n = _add_counts(AD, BC)
        return GraphInteger(p, n).reduced()

    def norm(self) -> float:
        """
        UNCHANGED FORMULA.
        """
        r = self.reduced()
        nP, encP = _build_union_from_counts(r.pos)
        nN, encN = _build_union_from_counts(r.neg)
        vertex_term = abs(nP - nN)
        bit_term = _weighted_bit_l1(encP, encN)
        return float(vertex_term) + bit_term

    def distance(self, other: "GraphInteger") -> float:
        return (self - other).norm()

    def is_zero(self) -> bool:
        r = self.reduced()
        return (not r.pos) and (not r.neg)

