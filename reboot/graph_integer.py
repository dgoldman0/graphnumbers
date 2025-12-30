# graph_integer.py
"""
GraphInteger: Grothendieck completion of graph isomorphism classes under disjoint union,
with multiplication given by Cartesian product.

Representation:
  X = (A, B) reduced, stored as signed multisets of Canon* CONNECTED components.
  Keys are exact: (n, enc) where enc is the Canon* upper-triangle bit encoding.

Reduction:
  Cancels identical connected components between positive and negative sides.

Norm (UNCHANGED FORMULA):
  ||(A,B)|| = |v(A)-v(B)| + Σ_{k≥1} 2^{-k} |a_k - b_k|
where (a_k), (b_k) are Canon* encodings of A and B (zero-padded).

Assumes canonstar.py provides:
  - canonstar()
  - to_bitrows, connected_components, induced_subgraph
  - disjoint_union_bitrows, encode_upper_triangle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import canonstar as cs

CompKey = Tuple[int, Tuple[int, ...]]  # (n, enc)

# Store canonical connected component adjacency by key (so we can multiply components)
_COMPONENT_ADJ: Dict[CompKey, Tuple[int, ...]] = {}
_PRODUCT_CACHE: Dict[Tuple[CompKey, CompKey], CompKey] = {}


def _register_connected_component(adj_bitrows: Sequence[int]) -> CompKey:
    """Canon* a connected graph component and register its canonical adjacency."""
    g = cs.canonstar(adj_bitrows)
    key: CompKey = (g.n, g.enc)
    if key not in _COMPONENT_ADJ:
        _COMPONENT_ADJ[key] = g.adj
    return key


def _ensure_K1_registered() -> CompKey:
    """Register the multiplicative identity K1."""
    k1_adj = [0]
    return _register_connected_component(k1_adj)


K1_KEY = _ensure_K1_registered()


def _add_counts(a: Dict[CompKey, int], b: Dict[CompKey, int]) -> Dict[CompKey, int]:
    out = dict(a)
    for k, v in b.items():
        if v:
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


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


def _neighbors(adj: Sequence[int], i: int) -> List[int]:
    x = adj[i]
    out = []
    while x:
        lsb = x & -x
        j = lsb.bit_length() - 1
        x -= lsb
        out.append(j)
    return out


def _cartesian_product_bitrows(adjA: Sequence[int], adjB: Sequence[int]) -> List[int]:
    """
    Cartesian product adjacency, vertex index = i*nB + j.
    """
    nA = len(adjA)
    nB = len(adjB)
    N = nA * nB
    out = [0] * N

    nbrA = [_neighbors(adjA, i) for i in range(nA)]
    nbrB = [_neighbors(adjB, j) for j in range(nB)]

    for i in range(nA):
        for j in range(nB):
            idx = i * nB + j
            row = 0
            for ip in nbrA[i]:
                row |= (1 << (ip * nB + j))
            for jp in nbrB[j]:
                row |= (1 << (i * nB + jp))
            row &= ~(1 << idx)
            out[idx] = row
    return out


def _component_product_key(a: CompKey, b: CompKey) -> CompKey:
    """Multiply two connected components, return connected component key (cached)."""
    if a == K1_KEY:
        return b
    if b == K1_KEY:
        return a

    pair = (a, b) if a <= b else (b, a)
    if pair in _PRODUCT_CACHE:
        return _PRODUCT_CACHE[pair]

    adjA = _COMPONENT_ADJ[a]
    adjB = _COMPONENT_ADJ[b]
    prod_adj = _cartesian_product_bitrows(adjA, adjB)

    # product of connected graphs is connected
    key = _register_connected_component(prod_adj)
    _PRODUCT_CACHE[pair] = key
    return key


def _mul_nonneg_counts(A: Dict[CompKey, int], B: Dict[CompKey, int]) -> Dict[CompKey, int]:
    """Distributive product of nonnegative graphs represented by connected-component multisets."""
    if not A or not B:
        return {}  # empty graph as multiplicative zero (V=∅)
    out: Dict[CompKey, int] = {}
    for ka, ca in A.items():
        for kb, cb in B.items():
            kp = _component_product_key(ka, kb)
            out[kp] = out.get(kp, 0) + ca * cb
    return {k: v for k, v in out.items() if v}


def _union_key_from_counts(counts: Dict[CompKey, int]) -> Tuple[int, Tuple[int, ...]]:
    """Build the Canon* encoding for a disjoint union of canonical components (already sorted by key)."""
    if not counts:
        return 0, ()
    blocks: List[Tuple[int, ...]] = []
    for key in sorted(counts.keys()):
        adj = _COMPONENT_ADJ[key]
        for _ in range(counts[key]):
            blocks.append(adj)
    union_adj = cs.disjoint_union_bitrows(blocks)
    return len(union_adj), cs.encode_upper_triangle(union_adj)


def _weighted_bit_l1(encA: Tuple[int, ...], encB: Tuple[int, ...]) -> float:
    """
    Σ_{k≥1} 2^{-k} |a_k - b_k| with infinite zero padding.
    """
    la, lb = len(encA), len(encB)
    m = min(la, lb)
    s = 0.0
    w = 0.5  # 2^{-1}

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


@dataclass(frozen=True)
class GraphInteger:
    """
    X = pos - neg, reduced; pos/neg are multisets of connected Canon* components.
    """
    pos: Dict[CompKey, int]
    neg: Dict[CompKey, int]

    @staticmethod
    def zero() -> "GraphInteger":
        return GraphInteger({}, {})

    @staticmethod
    def one() -> "GraphInteger":
        return GraphInteger({K1_KEY: 1}, {})

    @staticmethod
    def from_graph(adj: Sequence[Sequence[int]] | Sequence[int]) -> "GraphInteger":
        """Construct (G, 0) from an adjacency matrix or bitrows."""
        bitrows = cs.to_bitrows(adj)
        if not bitrows:
            return GraphInteger.zero()

        counts: Dict[CompKey, int] = {}
        for comp in cs.connected_components(bitrows):
            sub = cs.induced_subgraph(bitrows, comp)
            key = _register_connected_component(sub)
            counts[key] = counts.get(key, 0) + 1
        return GraphInteger(counts, {}).reduced()

    def reduced(self) -> "GraphInteger":
        p, n = _cancel(dict(self.pos), dict(self.neg))
        return GraphInteger(p, n)

    # ---- group ops ----

    def __add__(self, other: "GraphInteger") -> "GraphInteger":
        p = _add_counts(self.pos, other.pos)
        n = _add_counts(self.neg, other.neg)
        return GraphInteger(p, n).reduced()

    def __neg__(self) -> "GraphInteger":
        return GraphInteger(dict(self.neg), dict(self.pos))

    def __sub__(self, other: "GraphInteger") -> "GraphInteger":
        return (self + (-other)).reduced()

    # ---- ring ops ----

    def __mul__(self, other: "GraphInteger") -> "GraphInteger":
        # (A-B)(C-D) = (AC + BD) - (AD + BC)
        AC = _mul_nonneg_counts(self.pos, other.pos)
        BD = _mul_nonneg_counts(self.neg, other.neg)
        AD = _mul_nonneg_counts(self.pos, other.neg)
        BC = _mul_nonneg_counts(self.neg, other.pos)

        p = _add_counts(AC, BD)
        n = _add_counts(AD, BC)
        return GraphInteger(p, n).reduced()

    # ---- norm / metric (UNCHANGED) ----

    def norm(self) -> float:
        r = self.reduced()
        nP, encP = _union_key_from_counts(r.pos)
        nN, encN = _union_key_from_counts(r.neg)
        vertex_term = abs(nP - nN)
        bit_term = _weighted_bit_l1(encP, encN)
        return float(vertex_term) + float(bit_term)

    def distance(self, other: "GraphInteger") -> float:
        return (self - other).norm()

    # ---- convenience ----

    def is_zero(self) -> bool:
        r = self.reduced()
        return (not r.pos) and (not r.neg)

