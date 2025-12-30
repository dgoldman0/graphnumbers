# canonstar.py
"""
Canon*(G): an injective canonical form that *honors Cartesian product structure*.

Design goals:
  1) If G is (nontrivially) a Cartesian product of connected graphs, Canon* labels
     vertices in *tuple-lex order* of the canonical factors, so fibers/coordinates
     are visible and "reverse-by-fiber" works.
  2) If G does NOT admit a certified Cartesian factorization (or factor extraction
     is ambiguous/too expensive), Canon* falls back to your original global lex-min
     canonicalization from canonical_graph.py. This preserves injectivity.
  3) Disconnected graphs: Canon* is block-diagonal disjoint union of Canon* of each
     connected component, with components sorted by (n, enc).

This file assumes canonical_graph.py provides:
  - CanonicalGraph(bitrows) -> object with .enc (upper-triangle encoding) AND some
    vertex order attribute. We accept .order or .canonical_order or .perm/.perm_inv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import canonical_graph as cg

# -------------------- bitset adjacency basics --------------------

def to_bitrows(adj: Sequence[Sequence[int]] | Sequence[int]) -> List[int]:
    """Accept matrix (0/1) or bitrows; return bitrows list[int]."""
    if not adj:
        return []
    if isinstance(adj[0], int):
        return list(adj)
    n = len(adj)
    rows = [0] * n
    for i in range(n):
        r = 0
        for j in range(n):
            if adj[i][j]:
                r |= (1 << j)
        r &= ~(1 << i)  # no loops
        rows[i] = r
    return rows


def n_vertices(bitrows: Sequence[int]) -> int:
    return len(bitrows)


def has_edge(bitrows: Sequence[int], u: int, v: int) -> bool:
    return ((bitrows[u] >> v) & 1) == 1


def encode_upper_triangle(bitrows: Sequence[int]) -> Tuple[int, ...]:
    """
    Upper-triangle row-major including diagonal:
      (0,0),(0,1),...,(0,n-1),(1,1),(1,2),...,(n-1,n-1)
    """
    n = len(bitrows)
    out: List[int] = []
    for i in range(n):
        out.append(0)  # diagonal
        row = bitrows[i]
        for j in range(i + 1, n):
            out.append(1 if ((row >> j) & 1) else 0)
    return tuple(out)


def apply_order(bitrows: Sequence[int], order: Sequence[int]) -> List[int]:
    """Reorder vertices so new index i corresponds to old vertex order[i]."""
    n = len(bitrows)
    inv = [0] * n
    for new_i, old_i in enumerate(order):
        inv[old_i] = new_i

    out = [0] * n
    for old_u in range(n):
        new_u = inv[old_u]
        row = bitrows[old_u]
        new_row = 0
        x = row
        while x:
            lsb = x & -x
            old_v = lsb.bit_length() - 1
            x -= lsb
            new_v = inv[old_v]
            new_row |= (1 << new_v)
        new_row &= ~(1 << new_u)
        out[new_u] = new_row
    return out


def induced_subgraph(bitrows: Sequence[int], verts: Sequence[int]) -> List[int]:
    """Induced subgraph on verts, reindexed to 0..k-1, returned as bitrows."""
    idx = {v: i for i, v in enumerate(verts)}
    k = len(verts)
    out = [0] * k
    for i, v in enumerate(verts):
        row = bitrows[v]
        nr = 0
        for w in verts:
            if ((row >> w) & 1) == 1:
                nr |= (1 << idx[w])
        nr &= ~(1 << i)
        out[i] = nr
    return out


def disjoint_union_bitrows(components: Sequence[Sequence[int]]) -> List[int]:
    """Block-diagonal disjoint union of bitrow components."""
    total = sum(len(c) for c in components)
    if total == 0:
        return []
    out = [0] * total
    off = 0
    for comp in components:
        k = len(comp)
        for i in range(k):
            row = comp[i]
            shifted = 0
            x = row
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                x -= lsb
                shifted |= (1 << (off + j))
            out[off + i] = shifted
        off += k
    for i in range(total):
        out[i] &= ~(1 << i)
    return out


# -------------------- graph traversal --------------------

def connected_components(bitrows: Sequence[int]) -> List[List[int]]:
    n = len(bitrows)
    seen = [False] * n
    comps: List[List[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp: List[int] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            x = bitrows[v]
            while x:
                lsb = x & -x
                w = lsb.bit_length() - 1
                x -= lsb
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


def edge_list(bitrows: Sequence[int]) -> List[Tuple[int, int]]:
    n = len(bitrows)
    edges: List[Tuple[int, int]] = []
    for u in range(n):
        x = bitrows[u] >> (u + 1)
        base = u + 1
        while x:
            lsb = x & -x
            j = lsb.bit_length() - 1
            x -= lsb
            v = base + j
            edges.append((u, v))
    return edges


# -------------------- canonical_graph.py interop --------------------

def _cg_order(bitrows: Sequence[int]) -> List[int]:
    """Try to extract canonical vertex order from cg.CanonicalGraph."""
    G = cg.CanonicalGraph(list(bitrows))

    if hasattr(G, "order"):
        return list(getattr(G, "order"))
    if hasattr(G, "canonical_order"):
        return list(getattr(G, "canonical_order"))

    # Some implementations expose perm / perm_inv. We accept either.
    if hasattr(G, "perm"):
        perm = list(getattr(G, "perm"))
        n = len(bitrows)
        if sorted(perm) == list(range(n)):
            return perm
    if hasattr(G, "perm_inv"):
        perm_inv = list(getattr(G, "perm_inv"))
        n = len(bitrows)
        if sorted(perm_inv) == list(range(n)):
            return perm_inv

    raise AttributeError(
        "canonical_graph.CanonicalGraph must expose a canonical vertex order "
        "via .order or .canonical_order or .perm/.perm_inv"
    )


def _global_lexmin(bitrows: Sequence[int]) -> Tuple[List[int], Tuple[int, ...]]:
    """Return (bitrows in global-lex-min order, encoding)."""
    order = _cg_order(bitrows)
    re = apply_order(bitrows, order)
    return re, encode_upper_triangle(re)


# -------------------- DSU --------------------

class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


# -------------------- Cartesian factor direction extraction (certified) --------------------

def _edge_index_map(edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    m: Dict[Tuple[int, int], int] = {}
    for i, (u, v) in enumerate(edges):
        if u > v:
            u, v = v, u
        m[(u, v)] = i
    return m


def _neighbors(bitrows: Sequence[int], u: int) -> List[int]:
    x = bitrows[u]
    out: List[int] = []
    while x:
        lsb = x & -x
        v = lsb.bit_length() - 1
        x -= lsb
        out.append(v)
    return out


def theta_star_classes_by_chordless_squares(bitrows: Sequence[int]) -> List[List[Tuple[int, int]]]:
    """
    Build Θ* classes (candidate factor directions) by unioning opposite edges
    in chordless 4-cycles.

    For a cartesian product graph, the transitive closure of "opposite in a
    chordless square" groups edges by factor direction.

    For non-products, this may produce classes that fail certification later.
    """
    edges = edge_list(bitrows)
    m = len(edges)
    if m == 0:
        return []
    idx = _edge_index_map(edges)
    dsu = _DSU(m)

    n = len(bitrows)
    nbr = [_neighbors(bitrows, u) for u in range(n)]

    # For each edge (u,v), find chordless squares u-x-y-v-u:
    # edges: (u,x), (x,y), (y,v), (v,u)
    # opposite pairs: (v,u) opposite (x,y), and (u,x) opposite (v,y)
    for (u, v) in edges:
        # iterate x in N(u)\{v}, y in N(v)\{u}
        for x in nbr[u]:
            if x == v:
                continue
            # early prune: if u-x is not in idx due to ordering, normalize later
            for y in nbr[v]:
                if y == u or y == x:
                    continue
                # need x-y edge
                if not has_edge(bitrows, x, y):
                    continue
                # chordless: no u-y and no v-x
                if has_edge(bitrows, u, y) or has_edge(bitrows, v, x):
                    continue

                # union opposite edges
                a = idx[(u, v) if u < v else (v, u)]
                b = idx[(x, y) if x < y else (y, x)]
                dsu.union(a, b)

                # other opposite pair: (u,x) with (v,y)
                ex = idx[(u, x) if u < x else (x, u)]
                ey = idx[(v, y) if v < y else (y, v)]
                dsu.union(ex, ey)

    buckets: Dict[int, List[Tuple[int, int]]] = {}
    for i, e in enumerate(edges):
        r = dsu.find(i)
        buckets.setdefault(r, []).append(e)
    return list(buckets.values())


def quotient_graph_for_edgeclass(bitrows: Sequence[int], E: Sequence[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    """
    Quotient graph Q corresponding to an edge class E:
      - Remove edges in E, compute connected components.
      - Quotient vertices are components; quotient edges connect components
        that were connected by an edge in E.
    Returns (comp_of_vertex, quotient_bitrows).
    """
    n = len(bitrows)
    removed = set((u, v) if u < v else (v, u) for (u, v) in E)
    g_minus = list(bitrows)
    for (u, v) in removed:
        g_minus[u] &= ~(1 << v)
        g_minus[v] &= ~(1 << u)

    comps = connected_components(g_minus)
    comp_of = [-1] * n
    for ci, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = ci

    k = len(comps)
    q = [0] * k
    for (u, v) in removed:
        cu, cv = comp_of[u], comp_of[v]
        if cu != cv:
            q[cu] |= (1 << cv)
            q[cv] |= (1 << cu)
    for i in range(k):
        q[i] &= ~(1 << i)
    return comp_of, q


def cartesian_product_bitrows(factors: Sequence[Sequence[int]]) -> List[int]:
    """
    Cartesian product of factor graphs (bitrows). Vertex order is tuple-lex
    with factor 0 most significant.
    """
    if not factors:
        return []
    # start with K1 as identity
    cur = [0]  # K1
    for B in factors:
        A = cur
        nA, nB = len(A), len(B)
        N = nA * nB
        out = [0] * N

        nbrA = [_neighbors(A, i) for i in range(nA)]
        nbrB = [_neighbors(B, j) for j in range(nB)]

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

        cur = out
    return cur


# -------------------- Canon* object --------------------

@dataclass(frozen=True)
class CanonStarGraph:
    n: int
    adj: Tuple[int, ...]   # bitrows in Canon* labeling
    enc: Tuple[int, ...]   # upper-triangle encoding of adj
    # For disconnected graphs: list of connected component keys in order
    connected_component_keys: Tuple[Tuple[int, Tuple[int, ...]], ...]


# -------------------- Canon*(connected) with certification --------------------

def _canonstar_connected(bitrows: Sequence[int]) -> CanonStarGraph:
    n = len(bitrows)
    if n == 0:
        return CanonStarGraph(0, (), (), ())
    if n == 1:
        enc = encode_upper_triangle(bitrows)
        return CanonStarGraph(1, (0,), enc, ((1, enc),))

    # Candidate directions from chordless squares
    classes = theta_star_classes_by_chordless_squares(bitrows)

    # If no squares/relations, treat as prime: global lex-min fallback.
    if len(classes) <= 1:
        adj_g, enc_g = _global_lexmin(bitrows)
        return CanonStarGraph(n, tuple(adj_g), enc_g, ((n, enc_g),))

    # Build candidate factor quotient graphs for each class
    factor_bitrows: List[List[int]] = []
    factor_keys_for_sort: List[Tuple[int, Tuple[int, ...]]] = []
    for E in classes:
        _, q = quotient_graph_for_edgeclass(bitrows, E)
        # canonicalize factor by global lex-min (tie-breaker inside factor)
        q_can, q_enc = _global_lexmin(q)
        factor_bitrows.append(q_can)
        factor_keys_for_sort.append((len(q_can), q_enc))

    # Sort factors deterministically by (n, enc)
    perm = sorted(range(len(factor_bitrows)), key=lambda i: factor_keys_for_sort[i])
    factor_bitrows = [factor_bitrows[i] for i in perm]
    factor_keys_for_sort = [factor_keys_for_sort[i] for i in perm]

    # Handle repeated identical factors: choose best coordinate ordering by searching
    # permutations within each equal-key block (small factorial).
    # If a block is large, fall back to global lex-min.
    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < len(factor_keys_for_sort):
        j = i + 1
        while j < len(factor_keys_for_sort) and factor_keys_for_sort[j] == factor_keys_for_sort[i]:
            j += 1
        blocks.append((i, j))
        i = j

    # Build a "best" factor order by permuting within identical blocks (if small).
    best_factors = factor_bitrows

    # Only do expensive tie-break if there is any nontrivial identical block.
    need_tie = any((b1 - b0) > 1 for (b0, b1) in blocks)
    if need_tie:
        # If any block is too large, fallback
        if any((b1 - b0) > 7 for (b0, b1) in blocks):
            adj_g, enc_g = _global_lexmin(bitrows)
            return CanonStarGraph(n, tuple(adj_g), enc_g, ((n, enc_g),))

        # Enumerate permutations within blocks via recursion
        import itertools

        choices: List[List[Tuple[int, ...]]] = []
        block_factor_lists: List[List[List[int]]] = []
        for (b0, b1) in blocks:
            r = b1 - b0
            if r == 1:
                choices.append([(0,)])
                block_factor_lists.append([factor_bitrows[b0]])
            else:
                perms = list(itertools.permutations(range(r)))
                choices.append(perms)  # list of tuples
                block_factor_lists.append([factor_bitrows[b0 + t] for t in range(r)])

        best_enc: Optional[Tuple[int, ...]] = None
        best_list: Optional[List[List[int]]] = None

        def rec(block_idx: int, acc: List[List[int]]) -> None:
            nonlocal best_enc, best_list
            if block_idx == len(blocks):
                prod = cartesian_product_bitrows(acc)
                enc = encode_upper_triangle(prod)
                if best_enc is None or enc < best_enc:
                    best_enc = enc
                    best_list = [list(x) for x in acc]
                return
            perms = choices[block_idx]
            flist = block_factor_lists[block_idx]
            for p in perms:
                new_acc = acc + [flist[t] for t in p]
                rec(block_idx + 1, new_acc)

        rec(0, [])
        assert best_list is not None
        best_factors = best_list

    # Build tuple-lex product adjacency from (tie-broken) factor order
    prod = cartesian_product_bitrows(best_factors)

    # CERTIFY: product we built must be isomorphic to original.
    # Use your original global canonicalizer as the oracle for isomorphism here.
    _, enc_orig = _global_lexmin(bitrows)
    _, enc_prod = _global_lexmin(prod)
    if enc_orig != enc_prod:
        # Not a valid product decomposition; fall back to global lex-min
        adj_g, enc_g = _global_lexmin(bitrows)
        return CanonStarGraph(n, tuple(adj_g), enc_g, ((n, enc_g),))

    # Accept: Canon* is tuple-lex product labeling
    enc_star = encode_upper_triangle(prod)
    return CanonStarGraph(n, tuple(prod), enc_star, ((n, enc_star),))


# -------------------- Canon*(general) --------------------

def canonstar(adj: Sequence[Sequence[int]] | Sequence[int]) -> CanonStarGraph:
    """
    Canon*(G):
      - Decompose G into connected components.
      - Canon* each component (connected).
      - Sort components by (n, enc).
      - Assemble block-diagonal union in that order.
    """
    bitrows = to_bitrows(adj)
    n = len(bitrows)
    if n == 0:
        return CanonStarGraph(0, (), (), ())

    comps = connected_components(bitrows)
    comp_canon: List[CanonStarGraph] = []
    for comp in comps:
        sub = induced_subgraph(bitrows, comp)
        comp_canon.append(_canonstar_connected(sub))

    comp_canon.sort(key=lambda g: (g.n, g.enc))
    blocks = [list(g.adj) for g in comp_canon]
    union = disjoint_union_bitrows(blocks)
    enc = encode_upper_triangle(union)
    keys = tuple((g.n, g.enc) for g in comp_canon)
    return CanonStarGraph(len(union), tuple(union), enc, keys)

