import sys, time, random, pickle
from fractions import Fraction as Fr
from math import comb, factorial
from rb import *

E = ((0, 0), (1, 0))
results, cls = pickle.load(open('marginals.pkl', 'rb'))


def explicit_square(r, margin):
    """T_r(P^(n) x P^(n)) from the four explicit product graphs, roots in S x S."""
    xmin, xmax, ymin, ymax = -margin, 1 + margin, -margin, margin
    G = grid_adj(xmin, xmax, ymin, ymax)
    Gp = grid_adj(xmin, xmax, ymin, ymax, deleted=[E])
    S = [v for v in G if min(abs(v[0]) + abs(v[1]), abs(v[0] - 1) + abs(v[1])) <= r]
    terms = [(Gp, Gp, 1), (Gp, G, -1), (G, Gp, -1), (G, G, 1)]
    h = {}
    for X, Y, s in terms:
        nb = product_nb(adj_nb(X), adj_nb(Y))
        for u in S:
            for v in S:
                add_hist(h, cls.classify(ball(nb, (u, v), r)), s)
    # sanity: random roots with u outside S cancel exactly
    others = [v for v in G if v not in S]
    random.seed(0)
    for _ in range(200):
        u = random.choice(others); v = random.choice(list(G))
        if random.random() < 0.5:
            u, v = v, u
            if u in S and v in S:
                continue
        tot = {}
        for X, Y, s in terms:
            add_hist(tot, cls.classify(ball(product_nb(adj_nb(X), adj_nb(Y)), (u, v), r)), s)
        assert tot == {}, "no cancellation outside S x S?"
    return h


def power(h, n, r):
    out = {cls.classify(ball(lambda v: [], 'o', r)): 1}
    for _ in range(n):
        out = convolve(out, h, cls, r)
    return out


def regular_part(h):
    return {c: v for c, v in h.items() if is_regular_face(cls.reps[c])}


t = time.time()
r = 2
h1 = results[2]
sq_explicit = explicit_square(2, 5) if "--explicit2" in sys.argv else None
sq_conv = power(h1, 2, 2)
print("r=2: explicit grid-product T_2(P^2) == atom convolution:", sq_explicit == sq_conv if sq_explicit is not None else "skipped",
      " types:", len(sq_conv), " norm:", norm(sq_conv), " (16^2=256)  t=%.1fs" % (time.time() - t))
print("     regular part:", {cls.reps[c].number_of_nodes(): v for c, v in regular_part(sq_conv).items()})
supports = {}
for n in range(0, 5 if "--n4" in sys.argv else 4):
    t = time.time()
    hn = power(h1, n, 2)
    supports[n] = set(hn)
    expected_types = comb(n + 3, 3)
    print(f"r=2 n={n}: types={len(hn)} (multisets {expected_types}) norm={norm(hn)} (16^n={16**n})"
          f" regular={[(cls.reps[c].number_of_nodes(), v) for c, v in regular_part(hn).items()]}  t={time.time()-t:.1f}s")
for a in supports:
    for b in supports:
        if a < b:
            assert not (supports[a] & supports[b]), (a, b)
print("r=2: supports of T_2(P^n), n=0..4, pairwise disjoint")

# radius three
r = 3
h1 = results[3]
t = time.time()
sq = power(h1, 2, 3)
print(f"r=3 n=2: types={len(sq)} norm={norm(sq)} (36^2={36**2}) regular={[(cls.reps[c].number_of_nodes(), v) for c, v in regular_part(sq).items()]} t={time.time()-t:.1f}s")
if '--explicit3' in sys.argv:
    t = time.time()
    sq_e = explicit_square(3, 7)
    print("r=3: explicit grid-product T_3(P^2) == atom convolution:", sq_e == sq, " t=%.1fs" % (time.time() - t))
if '--cube3' in sys.argv:
    t = time.time()
    cu = convolve(sq, h1, cls, 3)
    print(f"r=3 n=3: types={len(cu)} (multisets {comb(9,3)}) norm={norm(cu)} (36^3={36**3}) regular={[(cls.reps[c].number_of_nodes(), v) for c, v in regular_part(cu).items()]} t={time.time()-t:.1f}s")
    print("r=3: supports n=2,3 disjoint:", not (set(sq) & set(cu)))
pickle.dump((results, cls), open('marginals2.pkl', 'wb'))
