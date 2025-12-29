import functools
from functools import lru_cache

# ============================================================
# Bitset adjacency (fast + hashable)
# ============================================================

def matrix_to_bitsets(M):
    """Convert 0/1 adjacency matrix (list-of-lists or numpy-like) to bitsets."""
    n = len(M)
    adj = [0] * n
    for i in range(n):
        row = M[i]
        bits = 0
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
        bits = adj[i]
        for j in range(n):
            M[i][j] = 1 if ((bits >> j) & 1) else 0
    return M

def inverse_perm(perm_new_to_old):
    """perm_new_to_old: tuple/list where new i -> old perm[i]. Returns old -> new."""
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
    Enc(P^T A P) in your row-major upper-triangle order (including diagonal),
    where perm_new_to_old maps new label -> old vertex.
    """
    n = len(adj)
    out = []
    for i_new in range(n):
        u = perm_new_to_old[i_new]
        # diagonal
        out.append(1 if ((adj[u] >> u) & 1) else 0)
        # upper triangle
        for j_new in range(i_new + 1, n):
            v = perm_new_to_old[j_new]
            out.append(1 if ((adj[u] >> v) & 1) else 0)
    return tuple(out)

# ============================================================
# Automorphisms: utilities
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

# ============================================================
# A tiny IR sweep to discover some automorphisms (seed)
# ============================================================

def _equitable_refine_ordered(adj, partition):
    """
    Ordered equitable refinement used ONLY to find automorphisms quickly.
    It splits cells by signature; cell order is deterministic.
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

def ir_find_automorphisms(adj, max_leaves=2000, verify=False):
    """
    Explore a limited number of refined leaves to harvest automorphisms.
    These automorphisms are valid for the full lex-min search later.
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

# ============================================================
# Orbit pruning under stabilizers (Option 1)
# ============================================================

class UnionFind:
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

def stabilizer_for_prefix(autos, fixed_vertices):
    """Return automorphisms that fix all vertices in fixed_vertices pointwise."""
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

def orbit_reps_under(vertices, generators):
    """
    Orbit reps of 'vertices' under the group generated by 'generators'.
    Uses undirected unions v ~ sigma(v) for each generator sigma.
    """
    verts = tuple(vertices)
    if len(verts) <= 1 or not generators:
        return list(verts)

    uf = UnionFind(verts)
    vset = set(verts)

    for sigma in generators:
        for v in verts:
            w = sigma[v]
            if w in vset:
                uf.union(v, w)

    reps = [min(g) for g in uf.groups()]
    reps.sort()
    return reps

def canonicalize_bitsets_lexmin(adj, seed_autos=True, seed_leaf_budget=2000, verify_autos=False):
    """
    Correct lex-min canonicalization:
      min_{perm} Enc(P^T A P)
    using automorphism-based orbit pruning under prefix stabilizers.
    """
    n = len(adj)
    identity = tuple(range(n))
    degrees = [(adj[i] & ~(1 << i)).bit_count() for i in range(n)]

    autos = {identity}
    if seed_autos:
        autos |= ir_find_automorphisms(adj, max_leaves=seed_leaf_budget, verify=verify_autos)

    best_enc = None
    best_perm = None
    best_adj = None

    # Track repeated encodings to discover more automorphisms during search
    rep_by_enc = {}

    def candidate_key(prefix_perm, x):
        """
        Heuristic to find small encodings early:
        compare adjacency bits to already-chosen labels in order.
        """
        vec = tuple(1 if ((adj[u] >> x) & 1) else 0 for u in prefix_perm)
        return (vec, degrees[x], x)

    def dfs(prefix_perm, remaining):
        nonlocal best_enc, best_perm, best_adj, autos

        if not remaining:
            perm = tuple(prefix_perm)
            enc = encoding_for_perm(adj, perm)

            # Discover automorphisms: equal enc => equal permuted matrix
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
        stabs = stabilizer_for_prefix(autos, fixed)

        # Orbit representatives among remaining vertices under current stabilizer
        choices = orbit_reps_under(remaining, stabs)

        # Try "promising" choices first
        choices.sort(key=lambda x: candidate_key(prefix_perm, x))

        for x in choices:
            new_remaining = tuple(v for v in remaining if v != x)
            dfs(prefix_perm + [x], new_remaining)

    dfs([], tuple(range(n)))
    return best_adj, best_enc, best_perm

# ============================================================
# User-facing API
# ============================================================

@lru_cache(maxsize=None)
def _canonicalize_labeled_cached(adj_tuple, seed_leaf_budget=2000):
    best_adj, best_enc, best_perm = canonicalize_bitsets_lexmin(
        adj_tuple,
        seed_autos=True,
        seed_leaf_budget=seed_leaf_budget,
        verify_autos=False,
    )
    return best_adj, best_enc

def canonicalize_matrix(M, seed_leaf_budget=2000):
    """
    Canonicalize an adjacency matrix under your Lex Order.
    Returns (canonical_matrix, canonical_encoding).
    """
    adj = matrix_to_bitsets(M)
    canon_adj, canon_enc = _canonicalize_labeled_cached(adj, seed_leaf_budget=seed_leaf_budget)
    return bitsets_to_matrix(canon_adj), canon_enc

class CanonicalGraph:
    """
    Canonical graph wrapper:
    - Always stores the lex-min canonical representative.
    - Identity is the encoding (hash/equality are trivial).
    """
    __slots__ = ("n", "_adj", "_enc")

    def __init__(self, M, seed_leaf_budget=2000):
        adj = matrix_to_bitsets(M)
        canon_adj, canon_enc = _canonicalize_labeled_cached(adj, seed_leaf_budget=seed_leaf_budget)
        self.n = len(adj)
        self._adj = canon_adj
        self._enc = canon_enc

    @classmethod
    def from_edges(cls, n, edges, seed_leaf_budget=2000):
        adj = [0] * n
        for u, v in edges:
            if u == v:
                continue
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        obj = cls.__new__(cls)
        canon_adj, canon_enc = _canonicalize_labeled_cached(tuple(adj), seed_leaf_budget=seed_leaf_budget)
        obj.n = n
        obj._adj = canon_adj
        obj._enc = canon_enc
        return obj

    @property
    def enc(self):
        return self._enc

    def adjacency_matrix(self):
        return bitsets_to_matrix(self._adj)

    def edges(self):
        out = []
        for i in range(self.n):
            b = self._adj[i] >> (i + 1)
            j = i + 1
            while b:
                lsb = b & -b
                k = lsb.bit_length() - 1
                out.append((i, j + k))
                b -= lsb
        return out

    def __hash__(self):
        return hash(self._enc)

    def __eq__(self, other):
        return isinstance(other, CanonicalGraph) and self._enc == other._enc

    def __repr__(self):
        return f"CanonicalGraph(n={self.n}, enc={self._enc})"

# ============================================================
# Canonical-safe operations (re-canonicalize on output)
# ============================================================

def add_edge(G, i, j, seed_leaf_budget=2000):
    adj = list(G._adj)
    adj[i] |= 1 << j
    adj[j] |= 1 << i
    return CanonicalGraph(bitsets_to_matrix(tuple(adj)), seed_leaf_budget=seed_leaf_budget)

def complement(G, seed_leaf_budget=2000):
    n = G.n
    all_mask = (1 << n) - 1
    adj = []
    for i in range(n):
        # flip all bits, clear self-loop
        adj.append((~G._adj[i]) & (all_mask ^ (1 << i)))
    return CanonicalGraph(bitsets_to_matrix(tuple(adj)), seed_leaf_budget=seed_leaf_budget)

def disjoint_union(G1, G2, seed_leaf_budget=2000):
    n1, n2 = G1.n, G2.n
    adj = [0] * (n1 + n2)
    for i in range(n1):
        adj[i] = G1._adj[i]
    for i in range(n2):
        adj[n1 + i] = (G2._adj[i] << n1)
    return CanonicalGraph(bitsets_to_matrix(tuple(adj)), seed_leaf_budget=seed_leaf_budget)

# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    # Two isomorphic labelings of C4
    M1 = [
        [0,1,0,1],
        [1,0,1,0],
        [0,1,0,1],
        [1,0,1,0],
    ]
    M2 = [
        [0,1,1,0],
        [1,0,0,1],
        [1,0,0,1],
        [0,1,1,0],
    ]

    G1 = CanonicalGraph(M1)
    G2 = CanonicalGraph(M2)

    print(G1 == G2)          # True
    print(G1.enc)            # canonical identity
    print(G1.adjacency_matrix())
