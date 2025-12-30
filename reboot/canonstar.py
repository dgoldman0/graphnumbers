# graph_integer.py
"""
GraphInteger ring for finite simple undirected graphs under:
  - addition: disjoint union
  - multiplication: Cartesian product

Representation:
  X = (A, B) in Grothendieck completion, stored as signed multisets of
  Canon* CONNECTED components, keyed by exact bit representation (n, enc).

Reduction:
  cancels identical connected components between positive and negative parts.

Norm (UNCHANGED FORMULA):
  ||(A,B)|| = |v(A)-v(B)| + sum_{n>=1} 2^{-n} |a_n - b_n|
where (a_n) and (b_n) are the Canon* encodings (upper-triangle bits, zero-padded).
We do NOT change the formula; we only changed the canonical representative Canon*.

This file assumes:
  - canonstar.py is available and provides CanonStarGraph and canonstar()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from functools import lru_cache

import canonstar as cs


CompKey = Tuple[int, Tuple[int, ...]]  # (n, enc)


# ---------- global registries / caches ----------

_COMPONENT_ADJ: Dict[CompKey, Tuple[int, ...]] = {}  # canonical connected component adjacency
_PRODUCT_CACHE: Dict[Tuple[CompKey, CompKey], CompKey] = {}


def _register_component(g: cs.CanonStarGraph) -> CompKey:
    """
    Register a connected CanonStarGraph in the global registry and return its key.
    Assumes g is connected (single component).
    """
    key: CompKey = (g.n, g.enc)
    if key not in _COMPONENT_ADJ:
        _COMPONENT_ADJ[key] = g.adj
    return key


def _is_K1(key: CompKey) -> bool:
    n, enc = key
    if n != 1:
        return False
    # upper triangle including diagonal for n=1 is a single 0-bit
    return len(enc) == 1 and enc[0] == 0


def _add_counts(a: Dict[CompKey, int], b: Dict[CompKey, int]) -> Dict[CompKey, int]:
    out = dict(a)
    for k, v in b.items():
        if v:
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


def _sub_counts(a: Dict[CompKey, int], b: Dict[CompKey, int]) -> Dict[CompKey, int]:
    out = dict(a)
    for k, v in b.items():
        if v:
            out[k] = out.get(k, 0) - v
            if out[k] == 0:
                del out[k]
    return out


def _cancel(pos: Dict[CompKey, int], neg: Dict[CompKey, int]) -> Tuple[Dict[CompKey, int], Dict[CompKey, int]]:
    common = set(pos).intersection(neg)
    for k in list(common):
        c = min(pos[k], neg[k])
        if c:
            pos[k] -= c
            neg[k] -= c
            if pos[k] == 0:
                del pos[k]
            if neg[k] == 0:
                del neg[k]
    return pos, neg


def _neighbors_from_bitrows(adj: Sequence[int], i: int) -> List[i]()_
