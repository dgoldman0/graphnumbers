"""Check (1)-(6): exact tree-cut marginals from finite buffered trees."""
import sys, time
from fractions import Fraction
from canon import *

reg = Registry()


def S(r, d):
    q = d - 1
    return sum(q ** j for j in range(r))


def tree_cut_hist(d, r, L, all_roots=True):
    G = edge_centered_tree(d, L)
    Gc = delete_edge(G, 0, 1)
    if all_roots:
        roots = list(G)
    else:
        dist0 = bfs_dist(G, 0, r + 2)
        dist1 = bfs_dist(G, 1, r + 2)
        roots = set(dist0) | set(dist1)
    h = histogram(reg, Gc, r, 1, roots)
    h = histogram(reg, G, r, -1, roots, h)
    return clean(h), len(G)


results = {}
for d in (2, 3, 4, 5):
    for r in (1, 2, 3, 4):
        q = d - 1
        b = 1 + d * S(r, d)
        if d == 5 and r == 4:
            Ls = [2 * r - 1, 2 * r]
        else:
            Ls = [2 * r - 1, 2 * r, 2 * r + 1]
        hs = []
        for L in Ls:
            nverts = 2 * sum(q ** j for j in range(L + 1))
            allr = nverts <= 15000
            h, n = tree_cut_hist(d, r, L, all_roots=allr)
            hs.append(h)
        assert all(h == hs[0] for h in hs), (d, r, "unstable")
        h = hs[0]
        # structure
        pos = {c: v for c, v in h.items() if v > 0}
        neg = {c: v for c, v in h.items() if v < 0}
        assert len(neg) == 1 and len(pos) == r, (d, r, len(pos), len(neg))
        (cR, vR), = neg.items()
        assert vR == -2 * S(r, d), (d, r, vR)
        assert reg.size(cR) == b
        # identify each positive atom by the distance of its deficient interior vertex
        got = {}
        for c, v in pos.items():
            adj, dist = reg.reps[c]
            defic = [x for x in adj if dist[x] < r and len(adj[x]) != d]
            assert len(defic) == 1, (d, r, defic)
            j = dist[defic[0]]
            assert len(adj[defic[0]]) == d - 1
            got[j] = (v, reg.size(c))
        assert sorted(got) == list(range(r))
        for j in range(r):
            assert got[j][0] == 2 * q ** j, (d, r, j, got[j])
            assert got[j][1] == b - S(r - j, d), (d, r, j, got[j], b - S(r - j, d))
        nrm = norm(reg, h)
        assert nrm == 4 * S(r, d)
        for k in (1, 2, 3):
            formula = 2 * sum(q ** j * (b - S(r - j, d)) ** k for j in range(r)) + 2 * S(r, d) * b ** k
            assert norm(reg, h, k) == formula, (d, r, k)
        results[d, r] = (nrm, len(h))
        print(f"d={d} r={r}: types={len(h)} variation={nrm} = 4*S_r={4*S(r,d)}; Ls={Ls}; p_rk ok", flush=True)

# d=2 check against cut-line E = lim (P_n - C_n)
for r in (1, 2, 3, 4):
    n = 4 * r + 6
    hE = histogram(reg, path_graph(n), r, 1)
    hE = clean(histogram(reg, cycle_graph(n), r, -1, None, hE))
    hB, _ = tree_cut_hist(2, r, 2 * r + 1)
    assert hE == hB, r
print("B_2 = E at radii 1..4: ok")
