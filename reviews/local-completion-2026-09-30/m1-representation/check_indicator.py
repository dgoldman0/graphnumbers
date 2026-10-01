"""Independent check of the indicator expansion, eq. (1) of REPRESENTATION_THEOREM.md.

mode 'literal': enumerate exactly the index set of (1): t<=N, S subset I x [t]
  with every new vertex used and |S|<=N, Q subset of nonedges; NO degree pruning.
  Patterns of degree > D are kept and evaluated (they must contribute 0).
mode 'pruned': same index set but discard patterns with a vertex of degree > D
  (as the note says may be done).
"""
import sys, math, random, itertools, time
from fractions import Fraction as Fr
from collections import defaultdict
sys.path.insert(0, '.')
from glib import *
import networkx as nx


def nonempty_subsets(items):
    out = []
    for k in range(1, len(items) + 1):
        out.extend(itertools.combinations(items, k))
    return out


def expansion(H, r, D, literal):
    h = len(H)
    d = dists(H, 0)
    I = [v for v in H if d[v] < r]
    N = D * len(I)
    AH = inj_rooted(H, 0, H, 0)
    nonedges = [(u, v) for u in range(h) for v in range(u + 1, h) if v not in H[u]]
    attach = nonempty_subsets(I)
    res = defaultdict(Fr)  # canonical rooted pattern -> coefficient
    reps = {}
    nterms = 0
    def q_iter():
        # all subsets Q of nonedges; in pruned mode skip those creating degree > D
        degs0 = {v: len(H[v]) for v in H}
        def qrec(i, chosen, degs):
            if i == len(nonedges):
                yield list(chosen); return
            yield from qrec(i + 1, chosen, degs)
            u, v = nonedges[i]
            if literal or (degs[u] < D and degs[v] < D):
                degs[u] += 1; degs[v] += 1; chosen.append((u, v))
                yield from qrec(i + 1, chosen, degs)
                chosen.pop(); degs[u] -= 1; degs[v] -= 1
        yield from qrec(0, [], degs0)
    for Q in q_iter():
        base = {v: set(H[v]) for v in H}
        for u, v in Q:
            base[u].add(v); base[v].add(u)
        if not literal and maxdeg(base) > D:
            continue
        # enumerate tuples (A_1..A_t)
        def rec(t, tup, size, degs):
            nonlocal nterms
            # record the current tuple as a complete S (t new vertices)
            F = {v: set(base[v]) for v in base}
            for j, A in enumerate(tup):
                F[h + j] = set(A)
                for u in A:
                    F[u].add(h + j)
            F = {k: frozenset(s) for k, s in F.items()}
            keep = literal or maxdeg(F) <= D
            if keep:
                c = canon(F, 0)
                reps.setdefault(c, F)
                res[c] += Fr((-1) ** (size + len(Q)), AH * math.factorial(t))
                nterms += 1
            if t == N:
                return
            for A in attach:
                if size + len(A) > N:
                    continue
                nd = dict(degs)
                for u in A:
                    nd[u] += 1
                if not literal and (max(nd[u] for u in A) > D or len(A) > D):
                    continue  # every extension keeps a vertex above D: all discarded
                rec(t + 1, tup + [A], size + len(A), nd)
        rec(0, [], 0, {v: len(base[v]) for v in base})
    out = {c: v for c, v in res.items() if v != 0}
    return out, reps, nterms, N


def all_targets(D, maxn_atlas=7, nrand=60, seed=5):
    T = []
    for G in nx.graph_atlas_g()[1:]:
        if G.number_of_nodes() > maxn_atlas:
            continue
        g = mk(G.number_of_nodes(), list(G.edges()))
        if maxdeg(g) <= D:
            T.append(g)
    rng = random.Random(seed)
    for _ in range(nrand):
        n = rng.randint(8, 14)
        T.append(random_maxdeg_graph(n, D, rng.uniform(0.2, 0.9), rng))
    for n in range(3, 12):
        T.append(cycle(n)); T.append(path(n))
    return [g for g in T if maxdeg(g) <= D]


def run(D, r, literal, maxn_atlas=7, nrand=60):
    t0 = time.time()
    targets = all_targets(D, maxn_atlas, nrand)
    types = {}
    for g in targets:
        for o in g:
            b = ball(g, o, r)
            types.setdefault(canon(b, 0), b)
    checks = 0
    maxpat = 0
    bound = (D + 1) * sum(D ** j for j in range(r + 1))
    total_terms = 0
    for cH, H in types.items():
        exp, reps, nterms, N = expansion(H, r, D, literal)
        total_terms += nterms
        for c in exp:
            F = reps[c]
            maxpat = max(maxpat, len(F))
            assert is_connected(F)
            if not literal:
                assert maxdeg(F) <= D and len(F) <= bound
        for g in targets:
            for o in g:
                actual = 1 if canon(ball(g, o, r), 0) == cH else 0
                val = sum(coef * inj_rooted(reps[c], 0, g, o) for c, coef in exp.items())
                assert val == actual, (D, r, H, g, o, val, actual)
                checks += 1
    print(f"D={D} r={r} literal={literal}: types={len(types)} labelled_terms={total_terms} "
          f"max_pattern_size={maxpat} bound M={bound} checks={checks} OK  ({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    for D, r in [(1, 1), (2, 1), (3, 1), (4, 1)]:
        run(D, r, literal=True, nrand=30)
    for D, r in [(2, 2), (2, 3), (3, 1), (4, 1), (3, 2)]:
        run(D, r, literal=False, nrand=40)
