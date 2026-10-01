import random, itertools
import numpy as np, networkx as nx, mpmath as mp
from fractions import Fraction as Q
import graphlocal as gl
from graphlocal.edge_interactions import EdgeInteraction
from graphlocal.controlled import controlled_heat
mp.mp.dps = 40

def heat_exact(G, t):
    if G.number_of_nodes() == 0: return mp.mpf(0)
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray()
    M = mp.matrix(L.tolist())
    ev = mp.eigsy(M)[0]
    return sum(mp.e ** (-t * ev[i]) for i in range(len(ev)))

def interaction_heat(G, edits, t):
    k = len(edits); tot = mp.mpf(0)
    for m in range(k + 1):
        for S in itertools.combinations(edits, m):
            H = G.copy()
            for u, v, s in S:
                if s == -1: H.remove_edge(u, v)
                else: H.add_edge(u, v)
            tot += (-1) ** (k - m) * heat_exact(H, t)
    return tot

rng = random.Random(5)
bad = 0; n_cases = 0
for trial in range(40):
    n = rng.randint(4, 8)
    while True:
        G = nx.gnp_random_graph(n, rng.uniform(0.3, 0.7), seed=rng.randint(0, 10**6))
        if nx.is_connected(G): break
    pairs = list(itertools.combinations(range(n), 2)); rng.shuffle(pairs)
    k = rng.randint(1, 3)
    edits = [(u, v, -1 if G.has_edge(u, v) else (1 if rng.random() < 0.5 else None)) for u, v in pairs[:k]]
    edits = [e for e in edits if e[2] is not None]
    if not edits: continue
    g = gl.graph(n, list(G.edges()))
    val = EdgeInteraction(g, edits, max_edits=12)
    for t in (Q(1, 10), Q(1, 2), Q(2)):
        cert = controlled_heat(val, t, "1e-9")
        ex = interaction_heat(G, edits, mp.mpf(t.numerator) / t.denominator)
        lo = mp.mpf(cert.interval.lower.numerator) / cert.interval.lower.denominator
        hi = mp.mpf(cert.interval.upper.numerator) / cert.interval.upper.denominator
        n_cases += 1
        if not (lo <= ex <= hi):
            bad += 1; print("NOT CONTAINED", trial, edits, t, lo, ex, hi)
print("random EdgeInteraction certified heat:", n_cases, "cases,", bad, "failures")

# E^2, E^3 and E*Line against closed forms
E = gl.CutLineDefect()
for t in (Q(1, 20), Q(1, 3), Q(1)):
    tt = mp.mpf(t.numerator) / t.denominator
    e = (1 - mp.e ** (-4 * tt)) / 2
    for name, val, ex in (("E^2", E * E, e ** 2), ("E^3", E * E * E, e ** 3)):
        try:
            cert = controlled_heat(val, t, "1e-8", max_steps=256)
            lo = mp.mpf(cert.interval.lower.numerator) / cert.interval.lower.denominator
            hi = mp.mpf(cert.interval.upper.numerator) / cert.interval.upper.denominator
            print(name, "t=", t, "contained:", lo <= ex <= hi, "radius", float(hi - lo) / 2, "steps", cert.steps)
        except Exception as exc:
            print(name, "t=", t, "exception", type(exc).__name__, exc)
