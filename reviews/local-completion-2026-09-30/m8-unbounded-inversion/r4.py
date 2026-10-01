import cmath, math, random
import networkx as nx
from powers import *
from fractions import Fraction as Fr
r = 4
T = T_E_power(1, r, 10)
th = [0.0] * r + [math.pi]   # u = 1 on every A_{4,j}, v = -1 on L_4
chi = sum(c * cmath.exp(1j * sum(float(x) * t for x, t in zip(coordinates(adj_from_cf(cf), r)[0], th))) for cf, c in T.items())
print('r=4 character chi(E) =', chi, '  chi(1 - E/16) =', 1 - chi / 16)
T2 = T_E_power(2, r, 10)
chi2 = sum(c * cmath.exp(1j * sum(float(x) * t for x, t in zip(coordinates(adj_from_cf(cf), r)[0], th))) for cf, c in T2.items())
print('chi(E^2) =', chi2, ' l1(T_4 E^2) =', sum(abs(c) for c in T2.values()), ' types', len(T2))
random.seed(9); bad = 0
def rb():
    while True:
        n = random.randint(1, 7); G = nx.gnp_random_graph(n, random.choice([0.3, 0.5, 0.8]), seed=random.randint(0, 10**9))
        if nx.is_connected(G): break
    adj = [set(G[v]) for v in range(n)]; b, _, _ = ball(adj, random.randrange(n), r); return b
for trial in range(400):
    B, C = rb(), rb()
    adj, verts = cartesian([B, C]); P, _, _ = ball(adj, verts.index((0, 0)), r)
    if tuple(x + y for x, y in zip(coordinates(B, r)[0], coordinates(C, r)[0])) != coordinates(P, r)[0]: bad += 1
print('r=4 additivity failures on 400 random pairs:', bad)
