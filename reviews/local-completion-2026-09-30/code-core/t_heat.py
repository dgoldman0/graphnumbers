from fractions import Fraction as Q
import random
import mpmath as mp
import networkx as nx
from common import *
from graphlocal import Line, Finite, heat_return, path, cycle, complete, star, graph
mp.mp.dps = 60

def mpq(x): return mp.mpf(x.numerator) / x.denominator

def heat_trace(G, t):
    n = G.number_of_nodes()
    if n == 0: return mp.mpf(0)
    L = mp.matrix(n, n)
    idx = {v: i for i, v in enumerate(G.nodes)}
    for u, v in G.edges:
        i, j = idx[u], idx[v]
        L[i, j] -= 1; L[j, i] -= 1; L[i, i] += 1; L[j, j] += 1
    E = mp.eigsy(L, eigvals_only=True)
    return mp.fsum(mp.e ** (-t * E[i]) for i in range(n))

bad = []
def chk(name, iv, true):
    lo, hi = mpq(iv.lower), mpq(iv.upper)
    if not (lo - mp.mpf(10)**-50 <= true <= hi + mp.mpf(10)**-50):
        bad.append(name); print("FAIL", name, float(lo), float(true), float(hi), float(true - lo), float(hi - true))

def hline(t): return mp.e ** (-2 * t) * mp.besseli(0, 2 * t)
for t in ["0", "1/1000", "1/10", "1/2", "1", "2", "5", "10", "31/3"]:
    tt = mpq(Q(t))
    for eps in ["1e-6", "1e-12", "1e-25"]:
        c = heat_return(Line(), t, eps, max_steps=600)
        chk(f"Line t={t} eps={eps}", c.interval, hline(tt))
        if Q(t) <= 2:
            c = heat_return(Line() * Line(), t, eps, max_steps=600)
            chk(f"L^2 t={t} eps={eps}", c.interval, hline(tt) ** 2)
            c = heat_return(Line() * Line() * Line(), t, "1e-8", max_steps=600) if Q(t) <= 1 else None
            if c: chk(f"L^3 t={t}", c.interval, hline(tt) ** 3)
        c = heat_return(Line() * Q(-3, 2), t, eps, max_steps=600)
        chk(f"-3/2 L t={t} eps={eps}", c.interval, -mp.mpf(3) / 2 * hline(tt))
        c = heat_return(Line() - Finite.from_graph(cycle(5), True), t, eps, max_steps=600)
        chk(f"L - C5/5 t={t} eps={eps}", c.interval, hline(tt) - heat_trace(nx.cycle_graph(5), tt) / 5)
print("closed forms done", len(bad))

random.seed(3)
for trial in range(120):
    n = random.randint(1, 12)
    G = nx.gnp_random_graph(n, random.random(), seed=random.randint(0, 10**9))
    g = from_nx(G)
    t = random.choice(["0", "1/7", "1/2", "1", "3", "13/4"])
    tt = mpq(Q(t))
    eps = random.choice(["1e-6", "1e-15", "1e-30"])
    for normalize in (False, True):
        X = Finite.from_graph(g, normalize=normalize)
        c = heat_return(X, t, eps, max_steps=600)
        val = heat_trace(G, tt) / (n if normalize else 1)
        chk(f"G trial {trial} n={n} t={t} norm={normalize}", c.interval, val)
    # signed combination with another graph
    H = nx.gnp_random_graph(random.randint(1, 8), random.random(), seed=random.randint(0, 10**9))
    X = Finite.from_graph(g) * Q(2, 3) - Finite.from_graph(from_nx(H)) * Q(5, 4)
    c = heat_return(X, t, eps, max_steps=600)
    chk(f"signed trial {trial}", c.interval, mp.mpf(2) / 3 * heat_trace(G, tt) - mp.mpf(5) / 4 * heat_trace(H, tt))
    # product (Cartesian) with materialized oracle
    if n <= 6:
        Hs = nx.gnp_random_graph(random.randint(1, 4), 0.7, seed=random.randint(0, 10**9))
        X = Finite.from_graph(g, True) * Finite.from_graph(from_nx(Hs), True)
        P = nx.cartesian_product(G, Hs)
        c = heat_return(X, t, eps, max_steps=600)
        chk(f"product trial {trial}", c.interval, heat_trace(P, tt) / P.number_of_nodes())
print("random finite done; bad:", len(bad))
