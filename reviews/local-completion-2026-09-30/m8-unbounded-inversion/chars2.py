import cmath, random
import networkx as nx
from indep import *
random.seed(5)
def hist_char(adj, r, thetas):
    tot = 0
    for v in range(len(adj)):
        b, _, _ = ball(adj, v, r)
        c, _ = coordinates(b, r)
        tot += cmath.exp(1j * sum(float(x) * t for x, t in zip(c, thetas)))
    return tot
worst = 0
for trial in range(60):
    r = random.choice([2, 3])
    gs = []
    for _ in range(2):
        while True:
            n = random.randint(2, 7); G = nx.gnp_random_graph(n, random.choice([0.3, 0.5, 0.8]), seed=random.randint(0, 10**9))
            if nx.is_connected(G): break
        gs.append([set(G[v]) for v in range(n)])
    P, _ = cartesian(gs)
    th = [random.uniform(-3, 3) for _ in range(r + 1)]
    lhs = hist_char(P, r, th); rhs = hist_char(gs[0], r, th) * hist_char(gs[1], r, th)
    worst = max(worst, abs(lhs - rhs) / max(1, abs(rhs)))
print('character multiplicativity on random graph products, worst rel err', worst)
