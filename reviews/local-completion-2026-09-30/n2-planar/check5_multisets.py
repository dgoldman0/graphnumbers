"""T_r(P^n) via one product per multiset of atoms (commutative, associative star_r).
Checks: which multiset products are locally regular (face lemma => only the all-intact one),
and whether distinct multisets give distinct rooted types (observed freeness, NOT claimed by the note for r>=3)."""
import sys, time, pickle, itertools
from math import factorial
from collections import Counter
from rb import *

E = ((0, 0), (1, 0))


def atoms(r):
    G = grid_adj(-3 * r, 3 * r + 1, -3 * r, 3 * r)
    Gp = grid_adj(-3 * r, 3 * r + 1, -3 * r, 3 * r, deleted=[E])
    out = []
    for hh in range(r):
        for bb in range(r - hh):
            out.append((f"A(h={hh},b={bb})", ball(adj_nb(Gp), (-hh, bb), r), 2 if bb == 0 else 4))
    out.append(("Z", ball(adj_nb(G), (0, 0), r), -2 * r * r))
    return out


def run(r, n):
    cls = Classifier()
    at = atoms(r)
    t = time.time()
    prod_cache = {(): None}
    hist = {}
    classes = {}
    for ms in itertools.combinations_with_replacement(range(len(at)), n):
        X = at[ms[0]][1]
        for i in ms[1:]:
            X = star_product_ball(X, at[i][1], r)
        cid = cls.classify(X)
        cnt = Counter(ms)
        coeff = factorial(n)
        for i, c in cnt.items():
            coeff = coeff // factorial(c) * 1
        mult = factorial(n)
        for c in cnt.values():
            mult //= factorial(c)
        w = mult
        for i in ms:
            w *= at[i][2]
        add_hist(hist, cid, w)
        classes.setdefault(cid, []).append(ms)
        if is_regular_face(X):
            print(f"   regular product: multiset={[at[i][0] for i in ms]} |V|={X.number_of_nodes()} rootdeg={root_degree(X)}")
    collisions = {c: m for c, m in classes.items() if len(m) > 1}
    nms = len(list(itertools.combinations_with_replacement(range(len(at)), n)))
    print(f"r={r} n={n}: multisets={nms} distinct types={len(classes)} collisions={len(collisions)} "
          f"norm={norm(hist)} ((4r^2)^n={(4*r*r)**n}) time={time.time()-t:.1f}s")
    reg = {c: v for c, v in hist.items() if is_regular_face(cls.reps[c])}
    print(f"   regular part coefficients: {list(reg.values())}  expected (-2r^2)^n = {(-2*r*r)**n}")
    for c, m in list(collisions.items())[:5]:
        print("   collision:", [[at[i][0] for i in ms] for ms in m])


for r, n in [(3, 2), (3, 3), (4, 2)] + ([(4, 3)] if '--big' in sys.argv else []):
    run(r, n)
    sys.stdout.flush()
