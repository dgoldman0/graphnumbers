"""Test the finite realization lemma (Step B) at fixed (D, r).

K  := linear functionals on deg-D r-ball types vanishing on all connected
      finite-graph histograms (computed from a finite sample of graphs);
K' := span of re-rooting differences inj_u(F;.) - inj_v(F;.) over patterns F
      for which both (F,u) and (F,v) are deg-D r-ball types.
K' is contained in K always (finite graphs are balanced).  Step B's argument
implies K = K' (with graphs up to M(D,r) vertices), i.e. the histogram span
has dimension |T| - dim K'.  Every balanced array annihilates K', so K = K'
means every balanced radius-r marginal is a finite signed graph combination.
"""
import sys, random, time
from collections import defaultdict
sys.path.insert(0, '.')
from glib import *
from types_gen import types_D2, types_r1, types_r2
import networkx as nx

P1, P2 = 2**61 - 1, 2**31 - 1


def rank_mod(rows, p):
    rows = [[x % p for x in r] for r in rows]
    rank = 0
    ncols = len(rows[0]) if rows else 0
    piv_rows = []
    for c in range(ncols):
        pr = None
        for i in range(rank, len(rows)):
            if rows[i][c]:
                pr = i; break
        if pr is None:
            continue
        rows[rank], rows[pr] = rows[pr], rows[rank]
        inv = pow(rows[rank][c], p - 2, p)
        rows[rank] = [(x * inv) % p for x in rows[rank]]
        for i in range(len(rows)):
            if i != rank and rows[i][c]:
                f = rows[i][c]
                rows[i] = [(a - f * b) % p for a, b in zip(rows[i], rows[rank])]
        rank += 1
    return rank


def exact_rank(rows):
    # rank over Q via fraction-free elimination on Python ints
    rows = [list(r) for r in rows if any(r)]
    if not rows:
        return 0
    ncols = len(rows[0])
    rank = 0
    for c in range(ncols):
        pr = next((i for i in range(rank, len(rows)) if rows[i][c]), None)
        if pr is None:
            continue
        rows[rank], rows[pr] = rows[pr], rows[rank]
        piv = rows[rank]
        for i in range(rank + 1, len(rows)):
            if rows[i][c]:
                f, g = rows[i][c], piv[c]
                rows[i] = [g * a - f * b for a, b in zip(rows[i], piv)]
                # reduce by gcd to keep numbers small
                from math import gcd
                gg = 0
                for x in rows[i]:
                    gg = gcd(gg, x)
                if gg > 1:
                    rows[i] = [x // gg for x in rows[i]]
        rank += 1
        rows = rows[:rank] + [r for r in rows[rank:] if any(r)]
    return rank


def sample_graphs(D, maxn_rand, nrand, seed=11, atlas_max=7):
    out = []
    for G in nx.graph_atlas_g()[1:]:
        if G.number_of_nodes() > atlas_max:
            continue
        g = mk(G.number_of_nodes(), list(G.edges()))
        if maxdeg(g) <= D and is_connected(g):
            out.append(g)
    rng = random.Random(seed)
    tries = 0
    while len(out) < nrand + 2000 and tries < 50 * nrand:
        tries += 1
        n = rng.randint(8, maxn_rand)
        g = random_maxdeg_graph(n, D, rng.uniform(0.05, 1.0), rng)
        # keep the largest component
        comps = components(g)
        big = max(comps, key=len)
        h = induced(g, big)
        if len(h) >= 2:
            out.append(h)
        if len(out) >= nrand + 1500:
            break
    return out


def run(D, r, types, maxn_rand=24, nrand=1500):
    t0 = time.time()
    idx = {canon(t, 0): i for i, t in enumerate(types)}
    n = len(types)
    # pattern matrix rows: M[(F,u)] as function of type B
    sizes = [(len(t), len(edges_of(t))) for t in types]
    M = []
    for i, F in enumerate(types):
        row = [0] * n
        for j, B in enumerate(types):
            if sizes[i][0] <= sizes[j][0] and sizes[i][1] <= sizes[j][1]:
                row[j] = inj_rooted(F, 0, B, 0)
        M.append(row)
    groups = defaultdict(list)
    for i, F in enumerate(types):
        groups[canon(F, None)].append(i)
    diffs = []
    for g, members in groups.items():
        for i in members[1:]:
            diffs.append([a - b for a, b in zip(M[i], M[members[0]])])
    dimK1 = exact_rank(diffs) if diffs else 0
    # histograms
    graphs = sample_graphs(D, maxn_rand, nrand)
    H = []
    nverts = []
    for g in graphs:
        v = [0] * n
        for o in g:
            v[idx[canon(ball(g, o, r), 0)]] += 1
        H.append(v); nverts.append(len(g))
    # sanity: K' annihilates all histograms
    for drow in diffs:
        for h in H:
            assert sum(a * b for a, b in zip(drow, h)) == 0
    rH = rank_mod(H, P1)
    rH2 = rank_mod(H, P2)
    # smallest vertex count achieving the rank
    order = sorted(range(len(H)), key=lambda i: nverts[i])
    need = None
    # incremental rank by size threshold
    for thr in sorted(set(nverts)):
        sub = [H[i] for i in range(len(H)) if nverts[i] <= thr]
        if rank_mod(sub, P1) == rH:
            need = thr; break
    bound = (D + 1) * sum(D ** j for j in range(r + 1))
    ok = (dimK1 + rH == n)
    print(f"D={D} r={r}: |T|={n} dimK'={dimK1} rank(hist)={rH} (mod p2: {rH2}) "
          f"dimK'+rank==|T|: {ok}; graphs used={len(H)}; full rank reached with <= {need} vertices; "
          f"M(D,r)={bound}  ({time.time()-t0:.1f}s)")
    return ok


if __name__ == '__main__':
    for r in range(1, 5):
        run(2, r, types_D2(r))
    for D in range(1, 5):
        run(D, 1, types_r1(D))
    run(3, 2, types_r2(3), maxn_rand=40, nrand=4000)
