"""T_r P from explicit buffered grids, ALL roots (no affected-root shortcut)."""
import sys, time, pickle
from fractions import Fraction as Fr
from rb import *

E = ((0, 0), (1, 0))
results = {}
cls = Classifier()


def TrP(r, margin):
    xmin, xmax, ymin, ymax = -margin, 1 + margin, -margin, margin
    G = grid_adj(xmin, xmax, ymin, ymax)
    Gp = grid_adj(xmin, xmax, ymin, ymax, deleted=[E])
    roots = list(G)
    h = {}
    histogram(adj_nb(Gp), roots, r, cls, +1, h)
    histogram(adj_nb(G), roots, r, cls, -1, h)
    return h


for r in range(1, 5):
    t = time.time()
    h1 = TrP(r, 2 * r + 1)
    h2 = TrP(r, 2 * r + 3)
    assert h1 == h2, ("not stable", r)
    # minimal stabilization check: margin 2r (should already be stable), margin 2r-1 (should differ)
    h0 = TrP(r, 2 * r)
    hm = TrP(r, 2 * r - 1)
    b = 1 + 2 * r * (r + 1)
    var = norm(h1)
    pos = {c: v for c, v in h1.items() if v > 0}
    neg = {c: v for c, v in h1.items() if v < 0}
    print(f"r={r}: types={len(h1)} variation={var} (4r^2={4*r*r}) pos={sorted(pos.values())} neg={list(neg.values())}"
          f" stable@2r={h0 == h1} stable@2r-1={hm == h1}  time={time.time()-t:.1f}s")
    # negative atom is the intact lattice ball
    (nc, nv), = neg.items()
    B = cls.reps[nc]
    assert B.number_of_nodes() == b and nv == -2 * r * r
    for k in (1, 2, 3):
        formula = 2 * (r - 1) * (b - 2) ** k + 2 * (b - 1) ** k + (4 * r * r - 2 * r) * b ** k
        got = norm(h1, cls, k)
        print(f"   p_(r,{k}) computed={got} formula(5)={formula} {'OK' if got == formula else 'MISMATCH'}")
    # identify each positive atom with its (h,b) position, check coefficient w_b
    exp = {}
    for hh in range(r):
        for bb in range(r - hh):
            G = grid_adj(-3 * r, 3 * r + 1, -3 * r, 3 * r, deleted=[E])
            cid = cls.classify(ball(adj_nb(G), (-hh, bb), r))
            exp[cid] = exp.get(cid, 0) + (2 if bb == 0 else 4)
    exp[nc] = -2 * r * r
    print("   histogram equals formula (3):", exp == h1, " #positive (h,b) positions:", r * (r + 1) // 2)
    # regular face
    for cid, v in h1.items():
        assert is_regular_face(cls.reps[cid]) == (v < 0 or r == 1), (r, cid)
    print("   regular-face: positive atoms irregular, intact regular:", True if r >= 2 else "(r=1: all regular)")
    results[r] = h1

pickle.dump((results, cls), open('marginals.pkl', 'wb'))
