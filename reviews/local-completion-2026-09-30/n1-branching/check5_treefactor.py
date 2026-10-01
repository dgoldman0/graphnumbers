"""Check the tree-factor lemma (Section 5): injectivity on multisets of small
rooted trees, and the proof's explicit recovery procedure."""
import itertools, random, time
from collections import Counter
import networkx as nx
from canon import *

reg = Registry()


def rooted_trees(maxn):
    out = {}
    for n in range(2, maxn + 1):
        for T in nx.nonisomorphic_trees(n):
            adj = {v: set(T[v]) for v in T}
            for root in T:
                b, d = ball(adj, root, 10 ** 6)
                out.setdefault(ahu(b, 0), (b, d))
    return list(out.values())


def recover(adj, dist, r):
    """Implements the proof of the tree-factor lemma on a rooted ball (root 0)."""
    N0 = sorted(adj[0])
    def same(a, b):
        return not ((adj[a] & adj[b]) - {0})
    blocks = []
    for a in N0:
        for blk in blocks:
            if same(a, blk[0]):
                assert all(same(a, b) for b in blk), "relation not transitive"
                blk.append(a)
                break
        else:
            blocks.append([a])
    factors = []
    for blk in blocks:
        verts = {0}
        stack = [(0, a) for a in blk]
        while stack:
            y, x = stack.pop()
            assert dist[x] == dist[y] + 1
            verts.add(x)
            if dist[x] >= r:
                continue
            for z in adj[x]:
                if z == y:
                    continue
                # same coordinate iff edges yx and xz lie in no common 4-cycle
                if not ((adj[y] & adj[z]) - {x}):
                    stack.append((x, z))
        sub = {v: adj[v] & verts for v in verts}
        b, d = ball(sub, 0, r)
        factors.append(reg.cid(b, d))
    return Counter(factors)


t0 = time.time()
for r, maxn, maxk in ((2, 7, 2), (2, 5, 3), (3, 7, 2), (3, 5, 3)):
    trees = rooted_trees(maxn)
    cls = sorted({reg.ball_cid(b, 0, r) for b, d in trees})
    seen = {}
    count = 0
    for k in range(1, maxk + 1):
        for ms in itertools.combinations_with_replacement(cls, k):
            p = ms[0]
            for c in ms[1:]:
                p = reg.star(p, c, r)
            key = tuple(sorted(ms))
            assert p not in seen or seen[p] == key, ("COLLISION", r, seen[p], key)
            seen[p] = key
            adj, dist = reg.reps[p]
            rec = recover(adj, dist, r)
            assert rec == Counter(ms), ("recovery failed", r, ms, rec)
            count += 1
    print(f"r={r}: {len(cls)} distinct nontrivial rooted tree r-balls (trees<= {maxn} vertices); "
          f"{count} multisets of size <= {maxk}: products pairwise distinct; proof's recovery exact  [{time.time()-t0:.0f}s]", flush=True)
