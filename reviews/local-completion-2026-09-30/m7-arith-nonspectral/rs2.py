import sys, time
from fractions import Fraction as Q
import networkx as nx
from gtools import *

R, S = rook(), shrikhande()
maxn = {1: 4, 2: 3}
for r in (1, 2):
    reg = TypeRegistry(); cache = {}
    hR = histogram(R, r, reg, Q(1, 16)); hS = histogram(S, r, reg, Q(1, 16))
    X = dict(hR)
    for k, v in hS.items(): X[k] = X.get(k, 0) - v
    one = {reg.id(rooted_from(nx.empty_graph(1), 0)): Q(1)}
    powers = [one, X]
    t0 = time.time()
    for n in range(2, maxn[r] + 1):
        powers.append(convolve(powers[-1], X, r, reg, cache))
    for n, P in enumerate(powers):
        sizes = sorted((reg.reps[k].number_of_nodes(), v) for k, v in P.items())
        print(f"r={r} n={n}: l1={l1(P)} (expect {2**n}); #types={len(P)}; (size,coef)={sizes}")
    # truncated inverse of 1+tX
    for t in (Q(1, 4), Q(-1, 3), Q(2, 5)):
        N = maxn[r] - 1
        Z = {}
        for n in range(N + 1):
            for k, v in powers[n].items():
                Z[k] = Z.get(k, 0) + (-t) ** n * v
        Z = {k: v for k, v in Z.items() if v}
        Y = dict(one)
        for k, v in X.items(): Y[k] = Y.get(k, 0) + t * v
        prod = convolve(Y, Z, r, reg, cache)
        resid = dict(prod)
        for k, v in one.items(): resid[k] = resid.get(k, 0) - v
        resid = {k: v for k, v in resid.items() if v}
        expect_resid = {k: -(-t) ** (N + 1) * v for k, v in powers[N + 1].items()}
        ok = resid == expect_resid
        var = l1(Z); expect = sum((2 * abs(t)) ** n for n in range(N + 1))
        w1 = sum(reg.reps[k].number_of_nodes() * abs(v) for k, v in Z.items()) if r == 1 else None
        expw1 = sum((2 * abs(t)) ** n * (6 * n + 1) for n in range(N + 1))
        print(f"  t={t}: residual exact={ok}; l1(partial inverse)={var} expect {expect}; p11={w1} expect {expw1 if r==1 else '-'}")
    print("time", time.time() - t0)
