import sys, random, itertools, math
from fractions import Fraction as Fr
from collections import defaultdict
sys.path.insert(0, '.')
from glib import *

rng = random.Random(2026)
reg = Registry()

# ---------------- Step C: degree cutoff ----------------
def QD(g, D):
    big = {v for v in g if len(g[v]) > D}
    return {v: frozenset(w for w in g[v] if v not in big and w not in big) for v in g}

def theta(B, D, r):
    return ball(QD(B, D), 0, r)

def rand_hubby(n, rng):
    # random graph with a few hubs (unbounded degree) plus sparse background
    edges = set()
    hubs = rng.sample(range(n), max(1, n // 6))
    for h in hubs:
        for v in rng.sample(range(n), rng.randint(1, n - 1)):
            if v != h:
                edges.add((min(h, v), max(h, v)))
    for _ in range(n):
        u, v = rng.sample(range(n), 2)
        edges.add((min(u, v), max(u, v)))
    return mk(n, sorted(edges))

loc_checks = 0
for trial in range(150):
    g = rand_hubby(rng.randint(5, 13), rng)
    for D in range(1, 6):
        q = QD(g, D)
        assert maxdeg(q) <= D
        for r in range(0, 4):
            for o in g:
                direct = canon(ball(q, o, r), 0)
                local = canon(theta(ball(g, o, r + 1), D, r), 0)
                assert direct == local
                loc_checks += 1
print('StepC locality checks', loc_checks, 'OK')

# Off-by-one probe: is the r-ball (rather than (r+1)-ball) enough? (expect: no)
counter = None
for trial in range(400):
    g = rand_hubby(rng.randint(5, 10), rng)
    for D in range(1, 4):
        q = QD(g, D)
        for r in range(1, 3):
            for o in g:
                if canon(ball(q, o, r), 0) != canon(ball(QD(ball(g, o, r), D), 0, r), 0):
                    counter = (D, r)
                    break
            if counter: break
        if counter: break
    if counter: break
print('r-ball alone insufficient (expected):', counter is not None, counter)

# Signed pushforward identity and bound (3)
tail_checks = 0
for trial in range(60):
    terms = [(Fr(rng.randint(-5, 5), rng.randint(1, 4)), rand_hubby(rng.randint(4, 11), rng)) for _ in range(4)]
    for D in range(1, 7):
        for r in range(0, 3):
            a_r1 = lin_hist(terms, r + 1, reg)
            a_r = lin_hist(terms, r, reg)
            push = defaultdict(Fr)
            for t, c in a_r1.items():
                push[reg.rid(theta(reg.reps[t], D, r))] += c
            push = {k: v for k, v in push.items() if v}
            # equals the histogram of the cut-off graphs
            cut = lin_hist([(c, QD(g, D)) for c, g in terms], r, reg)
            assert push == cut
            diff = defaultdict(Fr, push)
            for t, c in a_r.items():
                diff[t] -= c
            for k in range(1, 4):
                lhs = sum(abs(v) * len(reg.reps[t]) ** k for t, v in diff.items())
                rhs = 2 * sum(abs(v) * len(reg.reps[t]) ** k for t, v in a_r1.items() if len(reg.reps[t]) > D)
                assert lhs <= rhs
                tail_checks += 1
            # truncation and theta agree on balls with |B| <= D+1 (note claims |B| <= D)
            for t in a_r1:
                B = reg.reps[t]
                if len(B) <= D + 1:
                    assert canon(theta(B, D, r), 0) == canon(ball(B, 0, r), 0)
print('StepC signed pushforward/bound checks', tail_checks, 'OK')

# ---------------- Section 5: J(c) ----------------
def cyc_ball_type(n, r):
    return ('C', n) if n <= 2 * r + 1 else ('P', 2 * r + 1)

# validate cyc_ball_type against real balls
for n in range(3, 30):
    for r in range(0, 16):
        b = ball(cycle(n), 0, r)
        if n <= 2 * r + 1:
            assert canon(b, 0) == canon(cycle(n), 0)
        else:
            assert canon(b, 0) == canon(path(2 * r + 1), r) if False else canon(b, 0) == canon(ball(path(2 * r + 1), r, r), 0)

def T_J(c, r):
    h = defaultdict(Fr)
    for j, cj in enumerate(c, start=1):
        N = 3 ** j
        h[cyc_ball_type(2 * N, r)] += cj
        h[cyc_ball_type(N, r)] -= cj
    return {k: v for k, v in h.items() if v}

def size(t):
    return t[1]

for c in [[Fr(1)] * 4, [Fr(-7, 3), Fr(0), Fr(5), Fr(1, 2)], [Fr(2), Fr(-1), Fr(-1), Fr(3)]]:
    l1 = sum(abs(x) for x in c)
    sup = 0
    for r in range(0, 3 ** len(c) * 2 + 3):
        h = T_J(c, r)
        tv = sum(abs(v) for v in h.values())
        sup = max(sup, tv)
        # monotone in r
    assert sup == 2 * l1
    for m in range(1, len(c) + 1):
        r = 3 ** m
        h = T_J(c, r)
        assert sum(abs(v) for v in h.values()) == 2 * sum(abs(x) for x in c[:m])
        for k in (1, 2, 3):
            assert sum(abs(v) * size(t) ** k for t, v in h.items()) == sum(abs(c[j]) * ((2 * 3 ** (j + 1)) ** k + (3 ** (j + 1)) ** k) for j in range(m))
        assert h.get(('C', 2 * 3 ** m), 0) == c[m - 1]
    # coordinate functional P_j at radius N_j for later j's untouched by earlier
print('J(c) checks OK')
# total variation strictly increases exactly at which radii? (for c=1)
c = [Fr(1)] * 4
prev = None
jumps = []
for r in range(0, 200):
    tv = sum(abs(v) for v in T_J(c, r).values())
    if tv != prev:
        jumps.append((r, tv)); prev = tv
print('TV jumps (r, TV) for c=1:', jumps)

# ---------------- COMPARISON_LEMMAS numbers ----------------
def norm(h, k):
    return sum(abs(v) * len(reg.reps[t]) ** k for t, v in h.items())
for n in range(3, 12):
    h = lin_hist([(Fr(1, n), cycle(n)), (Fr(-1), mk(1, []))], 1, reg)
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
