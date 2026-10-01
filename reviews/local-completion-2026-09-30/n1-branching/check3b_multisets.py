"""No-cancellation check for (16) and (23) via pairwise distinctness of the
product balls of all atom multisets (sound: isomorphic balls have equal
invariants; collisions fall back to an exact VF2 test)."""
import sys, time, itertools
from collections import Counter
from math import comb, factorial
from fractions import Fraction as F
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from canon import *
from atoms import *

reg = Registry()


def multi_product_ball(factors, r):
    """factors: list of (adj, dist) rooted balls (root 0). Returns adj, dist of B_r(product)."""
    k = len(factors)
    verts = []
    def rec(i, cur, budget):
        if i == k:
            verts.append(tuple(cur)); return
        adj, dist = factors[i]
        for x in adj:
            if dist[x] <= budget:
                cur.append(x); rec(i + 1, cur, budget - dist[x]); cur.pop()
    rec(0, [], r)
    vs = set(verts)
    adjp = {v: set() for v in verts}
    for v in verts:
        for i in range(k):
            for y in factors[i][0][v[i]]:
                w = v[:i] + (y,) + v[i + 1:]
                if w in vs:
                    adjp[v].add(w)
    return ball(adjp, tuple([0] * k), r)


def invariant(adj, dist):
    G = nx.Graph()
    for v in adj:
        G.add_node(v, l0=f"{dist[v]}:{len(adj[v])}")
    for v in adj:
        for w in adj[v]:
            if v < w: G.add_edge(v, w)
    sub = nx.weisfeiler_lehman_subgraph_hashes(G, node_attr='l0', iterations=8, digest_size=16)
    return (len(adj), G.number_of_edges(), tuple(sorted(Counter(h[-1] for h in sub.values()).items()))), G, sub


def check(r, degs, M, maxtotal=None):
    t0 = time.time()
    atoms = {}
    weights = {}
    for d in degs:
        h, at = tree_cut(reg, d, r)
        for key, c in at.items():
            atoms[(d, key)] = reg.reps[c]
            weights[(d, key)] = h[c]
    keys = sorted(atoms, key=str)
    seen = {}
    ncheck = 0
    for alpha in itertools.product(range(M + 1), repeat=len(degs)):
        if maxtotal is not None and sum(alpha) > maxtotal:
            continue
        per_degree = []
        for d, a in zip(degs, alpha):
            dk = [k for k in keys if k[0] == d]
            per_degree.append(list(itertools.combinations_with_replacement(dk, a)))
        for combo in itertools.product(*per_degree):
            ms = tuple(sorted(sum(combo, ()), key=str))
            factors = [atoms[k] for k in ms]
            if factors:
                adj, dist = multi_product_ball(factors, r)
            else:
                adj, dist = {0: set()}, [0]
            inv, G, sub = invariant(adj, dist)
            if inv in seen:
                # exact fallback
                ms2, G2, sub2 = seen[inv]
                gm = GraphMatcher(G, G2, node_match=lambda a, b: True)
                raise AssertionError(f"invariant collision {ms} vs {ms2}")
            seen[inv] = (ms, None, None)
            ncheck += 1
    # consequences: exact norms
    print(f"r={r} degrees={degs} alpha_i<={M}{'' if maxtotal is None else f', |alpha|<={maxtotal}'}: "
          f"{ncheck} atom multisets give pairwise non-isomorphic product balls  [{time.time()-t0:.0f}s]", flush=True)
    return ncheck


check(2, (2,), 7)
check(2, (3,), 6)
check(2, (4,), 5)
check(2, (5,), 4)
check(3, (2,), 6)
check(3, (3,), 4)
check(3, (4,), 3)
check(4, (3,), 2)
check(2, (2, 3), 3)
check(2, (2, 4), 3)
check(2, (3, 4), 3)
check(2, (3, 5), 2)
check(2, (2, 3, 4), 2)
check(2, (2, 3, 4, 5), 1)
check(3, (2, 3), 2)
check(3, (3, 4), 2, maxtotal=3)
check(3, (2, 3, 4), 1)
