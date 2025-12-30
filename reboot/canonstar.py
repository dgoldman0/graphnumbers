# canonstar.py
"""
Canon*(G): a canonical representative for finite simple undirected graphs that is
coordinate-driven by the *Cartesian prime factorization*.

Key idea (connected case):
  1) Compute the finest product relation (edge partition) via Feder's theorem:
       R = (Θ ∪ Φ)^*
     where:
       - Θ: for edges e=xy, f=uv, if d(x,u)+d(y,v) != d(x,v)+d(y,u)
       - Φ: for incident edges yx and yv, if y is the only common neighbor of x and v
  2) Each equivalence class corresponds to one prime factor direction.
  3) For each class i, remove E_i and take components => coordinate values for axis i.
     Build the quotient graph Q_i whose vertices are these components and edges are E_i.
     Q_i is isomorphic to the corresponding prime factor.
  4) Canonicalize each Q_i using canonical_graph.py (lex-min tie-breaker).
  5) Order axes deterministically by (|V(Q_i)|, enc(Q_i)), with a deterministic tie-break
     for identical keys using the global canonical order of G (rare, but important).
  6) Order vertices by lexicographic coordinate tuples and relabel adjacency accordingly.

Disconnected case:
  - Canon* each connected component
  - sort components by (n, enc) and assemble block-diagonal.

This file assumes your existing canonical_graph.py provides:
  - class CanonicalGraph(bitrows) with a canonical vertex order attribute:
      .order or .canonical_order or .perm/.perm_inv

If your attribute name differs, adjust _cg_order_from_obj().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import itertools

import canonical_graph as cg


# ---------------------------
# Bitrow graph utilities
# ---------------------------

def to_bitrows(adj: Sequence[Sequence[int]] | Sequence[int]) -> List[int]:
    """Accepts either 0/1 adjacency matrix or bitrows; returns bitrows."""
    if not adj:
        return []
    if isinstance(adj[0], int):
        return list(adj)
    n = len(adj)
    rows = [0] * n
    for i in range(n):
        r = 0
        row = adj[i]
        for j in range(n):
            if row[j]:
                r |= (1 << j)
        r &= ~(1 << i)
        rows[i] = r
    return rows


def encode_upper_triangle(adj_bitrows: Sequence[int]) -> Tuple[int, ...]:
    """
    Upper triangle row-major INCLUDING diagonal:
    (0,0),(0,1),...,(0,n-1),(1,1),(1,2),...,(n-1,n-1)
    """
    n = len(adj_bitrows)
    bits: List[int] = []
    for i in range(n):
        bits.append(0)  # diagonal
        row = adj_bitrows[i]
        for j in range(i + 1, n):
            bits.append(1 if ((row >> j) & 1) else 0)
    return tuple(bits)


def apply_order(adj_bitrows: Sequence[int], order_new_to_old: Sequence[int]) -> List[int]:
    """Relabel so new vertex i corresponds to old vertex order_new_to_old[i]."""
    n = len(adj_bitrows)
    inv = [0] * n
    for new_i, old_i in enumerate(order_new_to_old):
        inv[old_i] = new_i
    out = [0] * n
    for old_u in range(n):
        new_u = inv[old_u]
        row = adj_bitrows[old_u]
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


def induced_subgraph(adj_bitrows: Sequence[int], verts: Sequence[int]) -> List[int]:
    """Induced subgraph on verts, reindexed 0..k-1."""
    idx = {v: i for i, v in enumerate(verts)}
    k = len(verts)
    out = [0] * k
    for i, v in enumerate(verts):
        row = adj_bitrows[v]
        rr = 0
        for w in verts:
            if (row >> w) & 1:
                rr |= (1 << idx[w])
        rr &= ~(1 << i)
        out[i] = rr
    return out


def disjoint_union_bitrows(components: Sequence[Sequence[int]]) -> List[int]:
    """Block-diagonal disjoint union of bitrow graphs."""
    if not components:
        return []
    offsets = []
    total = 0
    for comp in components:
        offsets.append(total)
        total += len(comp)
    out = [0] * total
    for comp, off in zip(components, offsets):
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
    for i in range(total):
        out[i] &= ~(1 << i)
    return out


def connected_components(adj_bitrows: Sequence[int]) -> List[List[int]]:
    n = len(adj_bitrows)
    seen = [False] * n
    comps: List[List[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            x = adj_bitrows[v]
            while x:
                lsb = x & -x
                w = lsb.bit_length() - 1
                x -= lsb
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


def edge_list(adj_bitrows: Sequence[int]) -> List[Tuple[int, int]]:
    n = len(adj_bitrows)
    edges: List[Tuple[int, int]] = []
    for u in range(n):
        x = adj_bitrows[u] >> (u + 1)
        base = u + 1
        while x:
            lsb = x & -x
            j = lsb.bit_length() - 1
            x -= lsb
            v = base + j
            edges.append((u, v))
    return edges


# ---------------------------
# Canonical_graph interop
# ---------------------------

def _cg_order_from_obj(obj: object, n: int) -> List[int]:
    if hasattr(obj, "order"):
        order = list(getattr(obj, "order"))
        if len(order) == n:
            return order
    if hasattr(obj, "canonical_order"):
        order = list(getattr(obj, "canonical_order"))
        if len(order) == n:
            return order
    if hasattr(obj, "perm_inv"):
        order = list(getattr(obj, "perm_inv"))
        if len(order) == n:
            return order
    if hasattr(obj, "perm"):
        perm = list(getattr(obj, "perm"))
        if len(perm) == n and sorted(perm) == list(range(n)):
            # treat as new->old (most common in this style)
            return perm
    raise AttributeError(
        "canonical_graph.CanonicalGraph must expose a canonical vertex order "
        "(.order or .canonical_order or .perm_inv or .perm)."
    )


def cg_canon_order(adj_bitrows: Sequence[int]) -> List[int]:
    """Global canonical order (lex-min tie-breaker) from canonical_graph.py."""
    n = len(adj_bitrows)
    obj = cg.CanonicalGraph(list(adj_bitrows))
    return _cg_order_from_obj(obj, n)


def cg_canon_bitrows(adj_bitrows: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Return (canon_bitrows, order_new_to_old)."""
    order = cg_canon_order(adj_bitrows)
    return apply_order(adj_bitrows, order), order


# ---------------------------
# Distances (BFS) for Θ
# ---------------------------

def all_pairs_shortest_paths(adj_bitrows: Sequence[int]) -> List[List[int]]:
    n = len(adj_bitrows)
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        q = [s]
        dist[s][s] = 0
        qi = 0
        while qi < len(q):
            v = q[qi]
            qi += 1
            dv = dist[s][v]
            x = adj_bitrows[v]
            while x:
                lsb = x & -x
                w = lsb.bit_length() - 1
                x -= lsb
                if dist[s][w] == -1:
                    dist[s][w] = dv + 1
                    q.append(w)
    return dist


# ---------------------------
# DSU for edge relations
# ---------------------------

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


def finest_product_relation_edge_classes(
    adj_bitrows: Sequence[int],
    *,
    max_edges_for_theta: int = 6000,
) -> Tuple[List[Tuple[int, int]], List[int], int]:
    """
    Computes the edge partition given by R = (Θ ∪ Φ)^* (Feder),
    where Θ is the distance-inequality relation and Φ is the "only common neighbor" relation.

    Returns:
      edges: list of undirected edges (u,v) with u<v
      cls_of_edge: list[int], class id per edge index
      num_classes: number of classes

    Note: Θ computation is O(m^2). If m is too large, we raise with instructions.
    """
    edges = edge_list(adj_bitrows)
    m = len(edges)
    if m == 0:
        return edges, [], 0

    if m > max_edges_for_theta:
        raise RuntimeError(
            f"Too many edges (m={m}) for the O(m^2) Θ computation.\n"
            "Increase max_edges_for_theta or switch to the Imrich–Peterin linear algorithm if needed."
        )

    dist = all_pairs_shortest_paths(adj_bitrows)

    dsu = _DSU(m)

    # Θ: for all pairs of edges
    for i in range(m):
        x, y = edges[i]
        for j in range(i + 1, m):
            u, v = edges[j]
            s1 = dist[x][u] + dist[y][v]
            s2 = dist[x][v] + dist[y][u]
            if s1 != s2:
                dsu.union(i, j)

    # Φ: for incident edges yx and yv where y is the only common neighbor of x and v
    edge_to_idx: Dict[Tuple[int, int], int] = {e: i for i, e in enumerate(edges)}
    n = len(adj_bitrows)
    for y in range(n):
        nbrs = []
        x = adj_bitrows[y]
        while x:
            lsb = x & -x
            u = lsb.bit_length() - 1
            x -= lsb
            nbrs.append(u)
        # check all pairs of neighbors (x,v)
        for a in range(len(nbrs)):
            x1 = nbrs[a]
            for b in range(a + 1, len(nbrs)):
                x2 = nbrs[b]
                common = adj_bitrows[x1] & adj_bitrows[x2]
                if common == (1 << y):
                    e1 = (x1, y) if x1 < y else (y, x1)
                    e2 = (x2, y) if x2 < y else (y, x2)
                    i1 = edge_to_idx.get(e1)
                    i2 = edge_to_idx.get(e2)
                    if i1 is not None and i2 is not None:
                        dsu.union(i1, i2)

    # Compress roots to class ids
    root_to_class: Dict[int, int] = {}
    cls_of_edge = [0] * m
    for i in range(m):
        r = dsu.find(i)
        if r not in root_to_class:
            root_to_class[r] = len(root_to_class)
        cls_of_edge[i] = root_to_class[r]

    return edges, cls_of_edge, len(root_to_class)


def components_without_class(
    adj_bitrows: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    cls_of_edge: Sequence[int],
    class_id: int,
) -> List[int]:
    """
    Returns comp_of_vertex for graph with edges of class_id removed.
    """
    n = len(adj_bitrows)
    # build adjacency with removed edges
    g_minus = list(adj_bitrows)
    for (u, v), c in zip(edges, cls_of_edge):
        if c == class_id:
            g_minus[u] &= ~(1 << v)
            g_minus[v] &= ~(1 << u)

    comp_of = [-1] * n
    cid = 0
    for s in range(n):
        if comp_of[s] != -1:
            continue
        stack = [s]
        comp_of[s] = cid
        while stack:
            v = stack.pop()
            x = g_minus[v]
            while x:
                lsb = x & -x
                w = lsb.bit_length() - 1
                x -= lsb
                if comp_of[w] == -1:
                    comp_of[w] = cid
                    stack.append(w)
        cid += 1
    return comp_of


def quotient_graph_for_class(
    adj_bitrows: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    cls_of_edge: Sequence[int],
    class_id: int,
) -> Tuple[List[int], List[int]]:
    """
    Quotient graph Q_i:
      - vertices are components of G - E_i
      - edges connect components joined by an edge from E_i

    Returns:
      comp_of_vertex (length n)
      quotient_adj_bitrows (length k)
    """
    comp_of = components_without_class(adj_bitrows, edges, cls_of_edge, class_id)
    k = max(comp_of) + 1
    q = [0] * k
    for (u, v), c in zip(edges, cls_of_edge):
        if c != class_id:
            continue
        cu, cv = comp_of[u], comp_of[v]
        if cu != cv:
            q[cu] |= (1 << cv)
            q[cv] |= (1 << cu)
    for i in range(k):
        q[i] &= ~(1 << i)
    return comp_of, q


# ---------------------------
# Canon* data class
# ---------------------------

@dataclass(frozen=True)
class CanonStarGraph:
    n: int
    adj: Tuple[int, ...]                          # bitrows in Canon* labeling
    enc: Tuple[int, ...]                          # upper-triangle encoding
    component_keys: Tuple[Tuple[int, Tuple[int, ...]], ...]  # (n, enc) per connected comp, sorted


# ---------------------------
# Canon* core
# ---------------------------

def _canonstar_connected(
    adj_bitrows: Sequence[int],
    *,
    max_edges_for_theta: int = 6000,
) -> CanonStarGraph:
    n = len(adj_bitrows)
    if n == 0:
        return CanonStarGraph(0, (), (), ())
    if n == 1:
        enc = encode_upper_triangle(adj_bitrows)
        return CanonStarGraph(1, (0,), enc, ((1, enc),))

    # Compute edge partition into prime-factor directions
    edges, cls_of_edge, num_classes = finest_product_relation_edge_classes(
        adj_bitrows, max_edges_for_theta=max_edges_for_theta
    )

    # If graph has no edges (connected implies n=1, already handled), but keep safe:
    if num_classes == 0:
        canon, _ = cg_canon_bitrows(adj_bitrows)
        enc = encode_upper_triangle(canon)
        return CanonStarGraph(n, tuple(canon), enc, ((n, enc),))

    # Prime case (only one class): Canon* agrees with your global canonicalization
    if num_classes == 1:
        canon, _ = cg_canon_bitrows(adj_bitrows)
        enc = encode_upper_triangle(canon)
        return CanonStarGraph(n, tuple(canon), enc, ((n, enc),))

    # Global canonical order used only to break *axis* ties when factor keys match
    global_order = cg_canon_order(adj_bitrows)
    pos_in_global = [0] * n
    for new_i, old_i in enumerate(global_order):
        pos_in_global[old_i] = new_i
    root = global_order[0]

    # Build axis info
    axes: List[Tuple[Tuple[int, Tuple[int, ...]], int, List[int], int]] = []
    # tuple: (factor_key, class_id, coords_per_vertex, tie_sig)

    for c in range(num_classes):
        comp_of, q_adj = quotient_graph_for_class(adj_bitrows, edges, cls_of_edge, c)

        q_canon, q_order = cg_canon_bitrows(q_adj)
        q_enc = encode_upper_triangle(q_canon)
        factor_key = (len(q_canon), q_enc)

        inv_q = [0] * len(q_order)
        for new_i, old_i in enumerate(q_order):
            inv_q[old_i] = new_i

        coords = [inv_q[comp_of[v]] for v in range(n)]

        # Tie signature: among edges of class c incident to canonical root, pick the smallest
        # global-canonical position of the neighbor (stable, deterministic).
        tie_candidates = []
        for (u, v), cc in zip(edges, cls_of_edge):
            if cc != c:
                continue
            if u == root:
                tie_candidates.append(pos_in_global[v])
            elif v == root:
                tie_candidates.append(pos_in_global[u])
        tie_sig = min(tie_candidates) if tie_candidates else (10**9)

        axes.append((factor_key, c, coords, tie_sig))

    # Sort axes by factor_key, then tie_sig, then class_id
    axes.sort(key=lambda t: (t[0][0], t[0][1], t[3], t[1]))

    # Build coordinate tuples
    coord_tuples = [tuple(ax[2][v] for ax in axes) for v in range(n)]

    # Order vertices lex by coordinate tuple
    v_order = sorted(range(n), key=lambda v: coord_tuples[v])

    canon = apply_order(adj_bitrows, v_order)
    enc = encode_upper_triangle(canon)
    return CanonStarGraph(n, tuple(canon), enc, ((n, enc),))


def canonstar(
    adj: Sequence[Sequence[int]] | Sequence[int],
    *,
    max_edges_for_theta: int = 6000,
) -> CanonStarGraph:
    """
    Canon*(G) for arbitrary graphs (possibly disconnected).
    """
    bitrows = to_bitrows(adj)
    n = len(bitrows)
    if n == 0:
        return CanonStarGraph(0, (), (), ())

    comps = connected_components(bitrows)
    if len(comps) == 1:
        return _canonstar_connected(bitrows, max_edges_for_theta=max_edges_for_theta)

    comp_graphs: List[CanonStarGraph] = []
    for comp in comps:
        sub = induced_subgraph(bitrows, comp)
        comp_graphs.append(_canonstar_connected(sub, max_edges_for_theta=max_edges_for_theta))

    # sort components by their canonical key (n, enc)
    comp_graphs.sort(key=lambda g: (g.n, g.enc))

    blocks = [g.adj for g in comp_graphs]
    union_adj = disjoint_union_bitrows(blocks)
    union_enc = encode_upper_triangle(union_adj)
    comp_keys = tuple((g.n, g.enc) for g in comp_graphs)

    return CanonStarGraph(len(union_adj), tuple(union_adj), union_enc, comp_keys)

