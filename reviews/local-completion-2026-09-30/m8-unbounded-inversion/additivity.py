import random
import networkx as nx
from indep import *
random.seed(7)
def star_prod(B, C, r):
    adj, verts = cartesian([B, C])
    root = verts.index((0, 0))
    b, _, _ = ball(adj, root, r)
    return b
def rand_ball(r):
    while True:
        n = random.randint(1, 8); p = random.choice([0.2, 0.35, 0.5, 0.8])
        G = nx.gnp_random_graph(n, p, seed=random.randint(0, 10**9))
        if nx.is_connected(G): break
    adj = [set(G[v]) for v in range(n)]
    root = random.randrange(n)
    b, _, _ = ball(adj, root, r)
    return b
stats = {}
for r in (2, 3):
    bad = {'d': 0, 'Q': 0, 'S': 0, 'coords': 0, 'N': 0}
    negN = 0; nonint = 0
    for trial in range(1500):
        B, C = rand_ball(r), rand_ball(r)
        P = star_prod(B, C, r)
        dB, dC, dP = len(B[0]), len(C[0]), len(P[0])
        if dP != dB + dC: bad['d'] += 1
        if square_count(P) != square_count(B) + square_count(C) + dB * dC: bad['Q'] += 1
        if sphere_counts(P, r) != ps_mul(sphere_counts(B, r), sphere_counts(C, r), r): bad['S'] += 1
        cB, NB = coordinates(B, r); cC, NC = coordinates(C, r); cP, NP = coordinates(P, r)
        if NP != NB + NC: bad['N'] += 1
        if tuple(a + b for a, b in zip(cB, cC)) != cP: bad['coords'] += 1
        if NB < 0: negN += 1
        if any(x.denominator != 1 for x in cB): nonint += 1
    stats[r] = (bad, negN, nonint)
print(stats)
# example of non-integral coordinates
for r in (2,3):
    for trial in range(2000):
        B = rand_ball(r)
        c, N = coordinates(B, r)
        if any(x.denominator != 1 for x in c):
            print('r',r,'nonintegral example: n=',len(B),'deg',len(B[0]),'coords',[str(x) for x in c]); break
