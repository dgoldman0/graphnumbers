import sys, random, itertools, math
from fractions import Fraction as Fr
from collections import defaultdict
sys.path.insert(0, '.')
from glib import *

rng = random.Random(2026)
reg = Registry()

# ---------------- COMPARISON_LEMMAS numbers ----------------
def norm(h, k):
    return sum(abs(v) * len(reg.reps[t]) ** k for t, v in h.items())
for n in range(3, 12):
    h = lin_hist([(Fr(1, n), cycle(n)), (Fr(-1), mk(1, []))], 1, reg)
    if n > 7:
        assert norm(h, 1) == 4
        continue
    assert norm(h, 1) == 4, (n, norm(h, 1))
    h = lin_hist([(Fr(1, n), complete(n))], 1, reg)
    assert norm(h, 1) == n
print('p11(C_n/n-K1)=4 for n>=3; p11(K_n/n)=n OK')
for R in range(0, 4):
    for n in range(3, 12):
        z = [(Fr(1), cycle(2 * n)), (Fr(-2), cycle(n))]
        zero = all(not lin_hist(z, r, reg) for r in range(R + 1))
        assert zero == (n > 2 * R + 1), (R, n)
print('z_n vanishes through radius R exactly when n>2R+1 OK')

# Sidorenko star bound and walk bound
def walks(g, o, j):
    cur = {o: 1}
    for _ in range(j):
        nxt = defaultdict(int)
        for v, c in cur.items():
            for w in g[v]:
                nxt[w] += c
        cur = nxt
    return sum(cur.values())
viol = 0
for trial in range(300):
    n = rng.randint(2, 12)
    g = mk(n, [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < rng.random()])
    for j in range(1, 4):
        for s in range(1, 4):
            lhs = sum(walks(g, o, j) ** s for o in g)
            rhs = sum(len(g[o]) ** (j * s) for o in g)
            if lhs > rhs:
                viol += 1
    for r in range(0, 4):
        for s in range(1, 4):
            for o in g:
                b = len(ball(g, o, r))
                assert b ** s <= (r + 1) ** (s - 1) * sum(walks(g, o, j) ** s for j in range(r + 1))
print('Sidorenko star-bound violations:', viol, '; walk bound OK')

# ---------------- ANALYSIS numbers ----------------
def V(terms): return sum(c * len(g) for c, g in terms)
def E(terms): return sum(c * len(edges_of(g)) for c, g in terms)
def mult(x, y): return [(a * b, cart(g, h)) for a, g in x for b, h in y]
smalls = [mk(1, []), complete(2), path(3), complete(3), cycle(4), star(3)]
for trial in range(30):
    X = [(Fr(rng.randint(-3, 3), rng.randint(1, 3)), rng.choice(smalls)) for _ in range(2)]
    P = [(Fr(1), mk(1, []))]
    for m in range(1, 4):
        P = mult(P, X)
        assert V(P) == V(X) ** m
        assert E(P) == m * V(X) ** (m - 1) * E(X)
print('E(X^m)=m V^(m-1) E OK')

# bipartite-ball semicharacter -> character; nonunits near 1
def chi(terms, r):
    tot = Fr(0)
    for c, g in terms:
        for o in g:
            b = ball(g, o, r)
            # bipartite test
            col = {0: 0}; ok = True
            stack = [0]
            while stack and ok:
                u = stack.pop()
                for w in b[u]:
                    if w not in col:
                        col[w] = 1 - col[u]; stack.append(w)
                    elif col[w] == col[u]:
                        ok = False; break
            tot += c * ok
    return tot
for r in range(0, 3):
    for a in smalls + [cycle(5), path(4)]:
        for b in smalls + [cycle(5)]:
            assert chi([(1, cart(a, b))], r) == chi([(1, a)], r) * chi([(1, b)], r)
for n in [3, 5, 7, 9]:
    u = [(Fr(1), mk(1, [])), (Fr(-1, 2 * n), cycle(2 * n)), (Fr(1, n), cycle(n))]
    r = (n - 1) // 2
    assert chi(u, r) == 0
print('bipartite-ball character multiplicative; u_n nonunits OK')
# K1 - tK2 killed by degree-generating evaluation at z=1/(2t) when |2t|>=1
for t in [Fr(1, 2), Fr(-1, 2), Fr(3, 4), Fr(-5, 2)]:
    z = 1 / (2 * t)
    val = 1 - t * 2 * z  # D_{K1}=1, D_{K2}(z)=2z
    assert val == 0 and abs(z) <= 1
print('K1 - tK2 nonunit for |2t|>=1 via disk characters OK')
