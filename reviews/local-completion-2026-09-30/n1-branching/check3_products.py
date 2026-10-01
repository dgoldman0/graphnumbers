"""Check (16), (23): no cancellation in powers and mixed products (r=2,3);
validate convolution against explicit finite Cartesian products."""
import sys, time, faulthandler
faulthandler.dump_traceback_later(240, exit=True)
from math import comb
from fractions import Fraction as F
from itertools import product as iprod
from canon import *
from atoms import *

reg = Registry()
unit = None


def unit_hist():
    return {reg.ball_cid({0: set()}, 0, 0): F(1)}


def power_list(h, n, r):
    u = {reg.cid({0: set()}, [0]): F(1)}
    out = [u]
    for _ in range(n):
        out.append(convolve(reg, out[-1], h, r))
    return out


t0 = time.time()
# ---- direct finite-product validation of the convolution engine ----
r = 2
G = edge_centered_tree(3, 2 * r - 1)
Gc = delete_edge(G, 0, 1)
n = 4 * r + 6
Pn, Cn = path_graph(n), cycle_graph(n)
direct = defaultdict(F)
for (X, sx) in ((Gc, 1), (G, -1)):
    for (Y, sy) in ((Pn, 1), (Cn, -1)):
        histogram(reg, cartesian(X, Y), r, sx * sy, None, direct)
direct = clean(direct)
hB3, _ = tree_cut(reg, 3, r)
hE = cut_line(reg, r)
conv = convolve(reg, hB3, hE, r)
assert direct == conv
print(f"direct finite product (G3\\e-G3)x(P_n-C_n) at r=2 equals local convolution: types={len(conv)}, norm={norm(reg, conv)} (expected {12*8})", flush=True)

direct = defaultdict(F)
for (X, sx) in ((Gc, 1), (G, -1)):
    for (Y, sy) in ((Gc, 1), (G, -1)):
        histogram(reg, cartesian(X, Y), r, sx * sy, None, direct)
direct = clean(direct)
conv = convolve(reg, hB3, hB3, r)
assert direct == conv
print(f"direct finite product (G3\\e-G3)^2 at r=2 equals local convolution: types={len(conv)}, norm={norm(reg, conv)} (expected 144)  [{time.time()-t0:.0f}s]", flush=True)

r = 3
G = edge_centered_tree(3, 2 * r - 1)
Gc = delete_edge(G, 0, 1)
n = 4 * r + 6
Pn, Cn = path_graph(n), cycle_graph(n)
direct = defaultdict(F)
for (X, sx) in ((Gc, 1), (G, -1)):
    for (Y, sy) in ((Pn, 1), (Cn, -1)):
        histogram(reg, cartesian(X, Y), r, sx * sy, None, direct)
direct = clean(direct)
hB3, _ = tree_cut(reg, 3, r)
hE = cut_line(reg, r)
conv = convolve(reg, hB3, hE, r)
assert direct == conv
print(f"direct finite product (G3\\e-G3)x(P_n-C_n) at r=3 equals local convolution: types={len(conv)}, norm={norm(reg, conv)} (expected {28*12})  [{time.time()-t0:.0f}s]", flush=True)

# ---- single-degree powers ----
plans = {2: {2: 5, 3: 5, 4: 4, 5: 3}, 3: {2: 4, 3: 3, 4: 2}}
for r, byd in plans.items():
    for d, N in byd.items():
        h, at = tree_cut(reg, d, r)
        pw = power_list(h, N, r)
        R = 4 * S(r, d)
        supports = []
        for k, hk in enumerate(pw):
            assert norm(reg, hk) == R ** k, (r, d, k, norm(reg, hk), R ** k)
            assert len(hk) == comb(k + r, r), (r, d, k, len(hk))
            supports.append(set(hk))
        for i in range(len(supports)):
            for j in range(i + 1, len(supports)):
                assert not (supports[i] & supports[j])
        # a mixed-sign polynomial
        coeffs = [F(1), F(-3, 2), F(2, 7), F(-5), F(1, 3), F(-1, 11)][:N + 1]
        poly = {}
        for a, hk in zip(coeffs, pw):
            poly = add(poly, {c: a * v for c, v in hk.items()})
        assert norm(reg, poly) == sum(abs(a) * R ** k for k, a in enumerate(coeffs))
        print(f"r={r} d={d}: powers 0..{N} norms (4S_r)^n with R={R}, supports C(n+r,r), disjoint; polynomial norm exact  [{time.time()-t0:.0f}s]", flush=True)

# ---- mixed products of several degrees ----
mixed_plans = [(2, (2, 3), 3), (2, (2, 4), 3), (2, (3, 4), 3), (2, (3, 5), 2), (2, (2, 3, 4), 2),
               (3, (2, 3), 2), (3, (3, 4), 2), (3, (2, 4), 2)]
for r, degs, M in mixed_plans:
    hs = {d: tree_cut(reg, d, r)[0] for d in degs}
    pw = {d: power_list(hs[d], M, r) for d in degs}
    Rs = {d: 4 * S(r, d) for d in degs}
    allsupp = {}
    import random
    random.seed(7)
    poly, coeff_norm = {}, F(0)
    for alpha in iprod(range(M + 1), repeat=len(degs)):
        if sum(alpha) > M:
            continue
        mono = None
        for d, a in zip(degs, alpha):
            mono = pw[d][a] if mono is None else convolve(reg, mono, pw[d][a], r)
        expect = 1
        nsupp = 1
        for d, a in zip(degs, alpha):
            expect *= Rs[d] ** a
            nsupp *= comb(a + r, r)
        assert norm(reg, mono) == expect, (r, degs, alpha)
        assert len(mono) == nsupp, (r, degs, alpha, len(mono), nsupp)
        for other, s in allsupp.items():
            assert not (s & set(mono)), (alpha, other)
        allsupp[alpha] = set(mono)
        a = F(random.choice([-3, -2, -1, 1, 2, 5]), random.choice([1, 2, 3, 7]))
        poly = add(poly, {c: a * v for c, v in mono.items()})
        coeff_norm += abs(a) * expect
    assert norm(reg, poly) == coeff_norm
    print(f"r={r} degrees={degs} |alpha|<={M}: monomial norms prod R_i^a_i, supports prod C(a_i+r,r), pairwise disjoint; (23) exact  [{time.time()-t0:.0f}s]", flush=True)
print("registry size", len(reg.reps))
