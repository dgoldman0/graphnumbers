import mpmath as mp
import networkx as nx
from fractions import Fraction as Q
mp.mp.dps = 50
def mpq(x): return mp.mpf(x.numerator) / x.denominator
def heat_trace(G, t):
    n = G.number_of_nodes()
    if n == 0: return mp.mpf(0)
    L = mp.matrix(n, n); idx = {v: i for i, v in enumerate(G.nodes)}
    for u, v in G.edges:
        i, j = idx[u], idx[v]; L[i, j] -= 1; L[j, i] -= 1; L[i, i] += 1; L[j, j] += 1
    E = mp.eigsy(L, eigvals_only=True)
    return mp.fsum(mp.e ** (-t * E[i]) for i in range(n))
def hline(t): return mp.e ** (-2 * t) * mp.besseli(0, 2 * t)
def trP(m, t): return mp.fsum(mp.e ** (-t * (2 - 2 * mp.cos(mp.pi * k / m))) for k in range(m))
def trC(n, t): return mp.fsum(mp.e ** (-t * (2 - 2 * mp.cos(2 * mp.pi * k / n))) for k in range(n))
def cuts_value(pos, t, n=400):
    pos = sorted(pos); pos = [p - pos[0] for p in pos]
    gaps = [b - a for a, b in zip(pos, pos[1:])]
    return mp.fsum(trP(g, t) for g in gaps) + trP(n - pos[-1], t) - trC(n, t)
