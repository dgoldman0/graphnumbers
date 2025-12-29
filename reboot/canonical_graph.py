"""
Grothendieck reduction for finite simple undirected graphs under disjoint union.

Key idea:
- Disjoint union monoid splits uniquely into connected components (up to isomorphism).
- In the Grothendieck completion, you "cancel" matching component isomorphism-types
  between (A, B) by multiset subtraction.

This module provides:
- Bitset graph representation (fast, hashable).
- Connected-component extraction.
- Canonical IDs for components (lex-min upper-triangle encoding under relabeling).
- Pair reduction: (A, B) -> (A', B') with no cancellable component types remaining.

Drop-in entrypoint:
    reduce_grothendieck_pair(A, B)
"""

from functools import lru_cache

# ============================================================
# Bitset adjacency utilities
# ============================================================

def matrix_to_bitsets(M):
    """Convert 0/1 adjacency matrix (list-of-lists or numpy-like) to bitsets."""
    n = len(M)
    adj = [0] * n
    for i in range(n):
        bits = 0
        row = M[i]
        for j in range(n):
            if int(row[j]) != 0:
                bits |= 1 << j
        adj[i] = bits
    return tuple(adj)

def bitsets_to_matrix(adj):
    """Convert bitsets to 0/1 adjacency matrix (list-of-lists)."""
    n = len(adj)
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        b = adj[i]
        for j in range(n):
            M[i][j] = 1 if ((b >> j) & 1) else 0
    return M

def inverse_perm(perm_new_to_old):
    """perm_new_to_old maps new label -> old vertex. Return old -> new."""
    n = len(perm_new_to_old)
    inv = [0] * n
    for new_i, old_v in enumerate(perm_new_to_old):
        inv[old_v] = new_i
    return tuple(inv)

def permute_adj(adj, perm_new_to_old):
    """
    Return adjacency in the new labeling induced by perm_new_to_old.
    New vertex i corresponds to old vertex perm[i].
    """
    n = len(adj)
    inv = inverse_perm(perm_new_to_old)  # old -> new
    new_adj = [0] * n
    for i_new in range(n):
        u_old = perm_new_to_old[i_new]
        b = adj[u_old]
        mapped = 0
        while b:
            lsb = b & -b
            v_old = lsb.bit_length() - 1
            mapped |= 1 << inv[v_old]
            b -= lsb
        new_adj[i_new] = mapped
    return tuple(new_adj)

def encoding_for_perm(adj, perm_new_to_old):
    """
    Enc(P^T A P) in row-major upper-triangle order (including diagonal),
    where perm_new_to_old maps new label -> old vertex.
    """
    n = len(adj)
    out = []
    for i_new in range(n):
        u = perm_new_to_old[i_new]
        out.append(1 if ((adj[u] >> u) & 1) else 0)  # diagonal
        for j_new in range(i_new + 1, n):
            v = perm_new_to_old[j_new]
            out.append(1 if ((adj[u] >> v) & 1) else 0)
    return tuple(out)

# ============================================================
# Connected components + induced subgraphs
# ============================================================

def connected_components(adj):
    """
    Return list of components, each as a list of vertex indices (in the original graph).
    Graph is treated as undirected.
    """
    n = len(adj)
    unvisited = (1 << n) - 1
    comps = []

    while unvisited:
        lsb = unvisited & -unvisited
        start = lsb.bit_length() - 1

        comp_mask = 0
        frontier = 1 << start
        unvisited ^= 1 << start
        comp_mask |= 1 << start

        while frontier:
            lsb = frontier & -frontier
            u = lsb.bit_length() - 1
            frontier ^= lsb

            nbrs = adj[u] & unvisited
            if nbrs:
                frontier |= nbrs
                unvisited ^= nbrs
                comp_mask |= nbrs

        # decode mask to vertex list
        vs = []
        b = comp_mask
        while b:
            lsb = b & -b
            v = lsb.bit_length() - 1
            vs.append(v)
            b -= lsb
        comps.append(vs)

    return comps

def induced_subgraph(adj, vertices):
    """
    Induced subgraph on 'vertices' (list of old indices), relabeled to 0..k-1.
    Returns bitset adjacency tuple of length k.
    """
    n = len(adj)
    vs = list(vertices)
    k = len(vs)
    if k == 0:
        return tuple()

    old_to_new = [-1] * n
    for i, v in enumerate(vs):
        old_to_new[v] = i

    set_mask = 0
    for v in vs:
        set_mask |= 1 << v

    sub = [0] * k
    for i_new, v_old in enumerate(vs):
        b = adj[v_old] & set_mask
        mapped = 0
        while b:
            lsb = b & -b
            w_old = lsb.bit_length() - 1
            j_new = old_to_new[w_old]
            mapped |= 1 << j_new
            b -= lsb
        sub[i_new] = mapped

    return tuple(sub)

def disjoint_union_from_components(component_adjs):
    """
    Build disjoint union from a list of component adjacencies (each relabeled 0..k-1).
    Returns bitset adjacency of the union (not canonicalized).
    """
    total_n = sum(len(c) for c in component_adjs)
    if total_n == 0:
        return tuple()

    adj = [0] * total_n
    offset = 0
    for comp in component_adjs:
        k = len(comp)
        for i in range(k):
            adj[offset + i] = comp[i] << offset
        offset += k
    return tuple(adj)

# ============================================================
# Automorphism-aided lex-min canonicalization (self-contained)
# ============================================================

def automorphism_from_equal_perms(p, q):
    """
    p and q are new->old permutations with identical permuted adjacency.
    Returns sigma (old->old) automorphism: sigma = p ∘ q^{-1}.
    """
    n = len(p)
    inv_q = inverse_perm(q)  # old -> new
    sigma = [0] * n
    for u_old in range(n):
        sigma[u_old] = p[inv_q[u_old]]
    return tuple(sigma)

def is_automorphism(adj, sigma_old_to_old):
    """
    Check A[u,v] == A[sigma(u), sigma(v)] via neighbor-set transport:
    sigma(N(u)) == N(sigma(u)).
    """
    n = len(adj)
    for u in range(n):
        image = 0
        b = adj[u]
        while b:
            lsb = b & -b
            v = lsb.bit_length() - 1
            image |= 1 << sigma_old_to_old[v]
            b -= lsb
        if image != adj[sigma_old_to_old[u]]:
            return False
    return True

def _equitable_refine_ordered(adj, partition):
    """
    Ordered equitable refinement (simple color refinement).
    Used to quickly discover some automorphisms (seeding).
    """
    part = [tuple(sorted(cell)) for cell in partition]
    changed = True
    while changed:
        changed = False
        masks = []
        for cell in part:
            m = 0
            for v in cell:
                m |= 1 << v
            masks.append(m)

        new_part = []
        for cell in part:
            if len(cell) <= 1:
                new_part.append(cell)
                continue

            buckets = {}
            for v in cell:
                sig = tuple(((adj[v] & mask).bit_count()) for mask in masks)
                buckets.setdefault(sig, []).append(v)

            if len(buckets) == 1:
                new_part.append(cell)
            else:
                changed = True
                for sig in sorted(buckets.keys()):
                    new_part.append(tuple(sorted(buckets[sig])))

        part = new_part
    return part

def _is_discrete(partition):
    return all(len(c) == 1 for c in partition)

def _perm_from_partition(partition):
    return tuple(c[0] for c in partition)

def ir_find_automorphisms(adj, max_leaves=1500, verify=False):
    """
    Limited individualization-refinement sweep to harvest automorphisms.
    """
    n = len(adj)
    identity = tuple(range(n))
    autos = {identity}
    rep_by_enc = {}
    leaves = 0

    def dfs(partition):
        nonlocal leaves, autos
        if leaves >= max_leaves:
            return

        part = _equitable_refine_ordered(adj, partition)

        if _is_discrete(part):
            perm = _perm_from_partition(part)
            enc = encoding_for_perm(adj, perm)
            leaves += 1
            if enc in rep_by_enc:
                sigma = automorphism_from_equal_perms(perm, rep_by_enc[enc])
                if sigma not in autos:
                    if (not verify) or is_automorphism(adj, sigma):
                        autos.add(sigma)
            else:
                rep_by_enc[enc] = perm
            return

        idx = next(i for i, c in enumerate(part) if len(c) > 1)
        cell = part[idx]
        for v in cell:
            rest = tuple(x for x in cell if x != v)
            child = list(part[:idx]) + [(v,), rest] + list(part[idx + 1 :])
            dfs(child)
            if leaves >= max_leaves:
                return

    dfs([tuple(range(n))])
    return autos

class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self):
        g = {}
        for x in self.parent:
            r = self.find(x)
            g.setdefault(r, []).append(x)
        return list(g.values())

def _stabilizer_for_prefix(autos, fixed_vertices):
    """Automorphisms fixing all vertices in fixed_vertices pointwise."""
    if not fixed_vertices:
        return list(autos)
    out = []
    for sigma in autos:
        ok = True
        for v in fixed_vertices:
            if sigma[v] != v:
                ok = False
                break
        if ok:
            out.append(sigma)
    return out

def _orbit_reps_under(vertices, generators):
    """
    Orbit reps of 'vertices' under generators (approx: union v~sigma(v) for each generator).
    """
    verts = tuple(vertices)
    if len(verts) <= 1 or not generators:
        return list(verts)
    uf = _UnionFind(verts)
    vset = set(verts)
    for sigma in generators:
        for v in verts:
            w = sigma[v]
            if w in vset:
                uf.union(v, w)
    reps = [min(g) for g in uf.groups()]
    reps.sort()
    return reps

def canonicalize_bitsets_lexmin(adj, seed_leaf_budget=1500, verify_autos=False):
    """
    Correct lex-min canonicalization:
      min_{perm} Enc(P^T A P)
    Uses orbit pruning under stabilizers, seeded by IR-found automorphisms.
    """
    n = len(adj)
    identity = tuple(range(n))
    degrees = [(adj[i] & ~(1 << i)).bit_count() for i in range(n)]

    autos = {identity} | ir_find_automorphisms(adj, max_leaves=seed_leaf_budget, verify=verify_autos)

    best_enc = None
    best_perm = None
    best_adj = None
    rep_by_enc = {}  # to discover additional automorphisms during search

    def candidate_key(prefix_perm, x):
        vec = tuple(1 if ((adj[u] >> x) & 1) else 0 for u in prefix_perm)
        return (vec, degrees[x], x)

    def dfs(prefix_perm, remaining):
        nonlocal best_enc, best_perm, best_adj, autos

        if not remaining:
            perm = tuple(prefix_perm)
            enc = encoding_for_perm(adj, perm)

            if enc in rep_by_enc:
                sigma = automorphism_from_equal_perms(perm, rep_by_enc[enc])
                if sigma not in autos:
                    if (not verify_autos) or is_automorphism(adj, sigma):
                        autos.add(sigma)
            else:
                rep_by_enc[enc] = perm

            if best_enc is None or enc < best_enc:
                best_enc = enc
                best_perm = perm
                best_adj = permute_adj(adj, perm)
            return

        fixed = set(prefix_perm)
        stabs = _stabilizer_for_prefix(autos, fixed)
        choices = _orbit_reps_under(remaining, stabs)
        choices.sort(key=lambda x: candidate_key(prefix_perm, x))

        for x in choices:
            new_remaining = tuple(v for v in remaining if v != x)
            dfs(prefix_perm + [x], new_remaining)

    dfs([], tuple(range(n)))
    return best_adj, best_enc, best_perm

@lru_cache(maxsize=None)
def _canonicalize_cached(adj_tuple, seed_leaf_budget=1500):
    best_adj, best_enc, _ = canonicalize_bitsets_lexmin(
        adj_tuple,
        seed_leaf_budget=seed_leaf_budget,
        verify_autos=False,
    )
    return best_adj, best_enc

def canonicalize_matrix(M, seed_leaf_budget=1500):
    """Return (canonical_matrix, canonical_encoding)."""
    adj = matrix_to_bitsets(M)
    canon_adj, canon_enc = _canonicalize_cached(adj, seed_leaf_budget=seed_leaf_budget)
    return bitsets_to_matrix(canon_adj), canon_enc

# ============================================================
# CanonicalGraph wrapper (optional, but handy)
# ============================================================

class CanonicalGraph:
    """
    Stores a graph in canonical form (lex-min under relabeling).
    Use .adj (bitsets), .enc (canonical ID), .n.
    """
    __slots__ = ("n", "adj", "enc")

    def __init__(self, M_or_adj, seed_leaf_budget=1500):
        if isinstance(M_or_adj, tuple) and all(isinstance(x, int) for x in M_or_adj):
            adj = M_or_adj
        else:
            adj = matrix_to_bitsets(M_or_adj)
        canon_adj, canon_enc = _canonicalize_cached(adj, seed_leaf_budget=seed_leaf_budget)
        self.n = len(canon_adj)
        self.adj = canon_adj
        self.enc = canon_enc

    def adjacency_matrix(self):
        return bitsets_to_matrix(self.adj)

    def __hash__(self):
        return hash((self.n, self.enc))

    def __eq__(self, other):
        return isinstance(other, CanonicalGraph) and self.n == other.n and self.enc == other.enc

    def __repr__(self):
        return f"CanonicalGraph(n={self.n}, enc={self.enc})"

# ============================================================
# Grothendieck reduction machinery
# ============================================================

def component_multiset(adj, seed_leaf_budget=1500):
    """
    Decompose graph into connected components, canonicalize each component, and count types.

    Returns:
        counts: dict[key] = count
        rep_adj: dict[key] = canonical component adjacency (bitsets, relabeled 0..k-1)
    where key = (k, enc) ensures no size-collisions.
    """
    counts = {}
    rep_adj = {}

    for vs in connected_components(adj):
        sub = induced_subgraph(adj, vs)  # relabeled 0..k-1
        canon_sub_adj, canon_sub_enc, _ = canonicalize_bitsets_lexmin(
            sub,
            seed_leaf_budget=seed_leaf_budget,
            verify_autos=False,
        )
        k = len(canon_sub_adj)
        key = (k, canon_sub_enc)
        counts[key] = counts.get(key, 0) + 1
        # keep a representative adjacency for reconstruction
        rep_adj[key] = canon_sub_adj

    return counts, rep_adj

def cancel_component_multisets(countsA, countsB):
    """
    In-place cancellation of common keys between two multisets of components.
    """
    common = set(countsA.keys()) & set(countsB.keys())
    for key in common:
        k = min(countsA[key], countsB[key])
        countsA[key] -= k
        countsB[key] -= k
        if countsA[key] == 0:
            del countsA[key]
        if countsB[key] == 0:
            del countsB[key]

def rebuild_from_multiset(counts, reps):
    """
    Build a (non-canonical) disjoint union graph from a component multiset.
    Components are inserted in sorted key order for determinism.
    Returns bitset adjacency of the union.
    """
    component_adjs = []
    for key in sorted(counts.keys()):
        comp_adj = reps[key]
        c = counts[key]
        for _ in range(c):
            component_adjs.append(comp_adj)
    return disjoint_union_from_components(component_adjs)

def reduce_grothendieck_pair(A, B, seed_leaf_budget=1500, canonical_output=True):
    """
    Reduce a Grothendieck pair (A, B) under disjoint union:
      (A, B) ~ (A', B') where common connected component types are cancelled.

    Inputs:
      A, B can be adjacency matrices (list-of-lists / numpy-like) OR bitset adj tuples
      OR CanonicalGraph objects.

    Output:
      If canonical_output=True: returns (CanonicalGraph(A'), CanonicalGraph(B'))
      else: returns (adj_bitsets_Aprime, adj_bitsets_Bprime)
    """
    def to_adj(x):
        if isinstance(x, CanonicalGraph):
            return x.adj
        if isinstance(x, tuple) and all(isinstance(t, int) for t in x):
            return x
        return matrix_to_bitsets(x)

    adjA = to_adj(A)
    adjB = to_adj(B)

    countsA, repsA = component_multiset(adjA, seed_leaf_budget=seed_leaf_budget)
    countsB, repsB = component_multiset(adjB, seed_leaf_budget=seed_leaf_budget)

    # Reps are keyed identically, so merge reps maps (they should agree on canonical adj)
    reps = dict(repsA)
    reps.update(repsB)

    cancel_component_multisets(countsA, countsB)

    Aprime_adj = rebuild_from_multiset(countsA, reps)
    Bprime_adj = rebuild_from_multiset(countsB, reps)

    if canonical_output:
        return CanonicalGraph(Aprime_adj, seed_leaf_budget=seed_leaf_budget), CanonicalGraph(Bprime_adj, seed_leaf_budget=seed_leaf_budget)
    return Aprime_adj, Bprime_adj

# ============================================================
# Quick sanity check (optional)
# ============================================================

if __name__ == "__main__":
    # A = (triangle) ⊔ (path on 3)
    tri = [
        [0,1,1],
        [1,0,1],
        [1,1,0],
    ]
    p3 = [
        [0,1,0],
        [1,0,1],
        [0,1,0],
    ]
    A = disjoint_union_from_components([matrix_to_bitsets(tri), matrix_to_bitsets(p3)])

    # B = (triangle) ⊔ (edge)
    edge = [
        [0,1],
        [1,0],
    ]
    B = disjoint_union_from_components([matrix_to_bitsets(tri), matrix_to_bitsets(edge)])

    Ar, Br = reduce_grothendieck_pair(A, B, canonical_output=False)
    # Expect: Ar ~ p3, Br ~ edge (up to canonicalization if enabled)
    print("Reduced sizes:", len(Ar), len(Br))
