import random, networkx as nx, indep, refl
from fractions import Fraction as Fr
from collections import defaultdict
rng = random.Random(23)
reg = indep.Registry()
def rand_conn(n, p):
    while True:
        G = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10**9))
        if nx.is_connected(G): return G
def rand_terms():
    return [(Fr(rng.randint(-4, 4) or 1, rng.randint(1, 3)), rand_conn(rng.randint(1, 5), rng.random())) for _ in range(rng.randint(2, 4))]
bad = defaultdict(int); cnt = 0; tight = defaultdict(lambda: Fr(0))
for _ in range(40):
    X = rand_terms()
    FX = refl.F(X); RX = [(c, refl.R_graph(G)) for c, G in X]; OX = [(c, refl.O_graph(G)) for c, G in X]
    for r in (0, 1, 2):
        hX = indep.lin_hist(X, r, reg); hX1 = indep.lin_hist(X, r + 1, reg)
        hF = indep.lin_hist(FX, r, reg)
        hR = indep.lin_hist(RX, r, reg) if r else None
        hO = indep.lin_hist(OX, r, reg) if r else None
        for k in (1, 2, 3):
            cnt += 1
            a, b = indep.wnorm(hF, k, reg), indep.wnorm(hX1, k, reg)
            if a > b: bad[('F', r, k)] += 1
            if r >= 1:
                s = indep.wnorm(hX, k, reg)
                if indep.wnorm(hR, k, reg) > (2 ** (k + 1) + 1) * s: bad[('R', r, k)] += 1
                if indep.wnorm(hO, k, reg) > 3 ** (k + 1) * s: bad[('O', r, k)] += 1
                if s: tight[('R', k)] = max(tight[('R', k)], indep.wnorm(hR, k, reg) / s); tight[('O', k)] = max(tight[('O', k)], indep.wnorm(hO, k, reg) / s)
print("checks", cnt, "failures", dict(bad))
print("max observed ratios:", {k: float(v) for k, v in tight.items()})
