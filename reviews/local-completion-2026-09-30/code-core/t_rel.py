from fractions import Fraction as Q
import random, itertools
import mpmath as mp
import networkx as nx
from common import *
from t_heat import heat_trace, mpq, hline
from graphlocal import (Line, Finite, relative_heat, CutLineDefect, SparseEdgeDifference, cycle, path, graph,
                        PreparedLocal, PreparedRelativeHeat, LineCutDefect, TwoCutLineDefect, CutInteraction,
                        connected_cut_interaction)
mp.mp.dps = 60
bad = []
SL = mp.mpf(10) ** -50
def chk(name, iv, true):
    lo, hi = mpq(iv.lower), mpq(iv.upper)
    if not (lo - SL <= true <= hi + SL):
        bad.append(name); print("FAIL", name, float(lo), float(true), float(hi))

def trP(m, t):
    return mp.fsum(mp.e ** (-t * (2 - 2 * mp.cos(mp.pi * k / m))) for k in range(m))
def trC(n, t):
    return mp.fsum(mp.e ** (-t * (2 - 2 * mp.cos(2 * mp.pi * k / n))) for k in range(n))
def cuts_value(pos, t, n=600):
    pos = sorted(pos); pos = [p - pos[0] for p in pos]
    gaps = [b - a for a, b in zip(pos, pos[1:])]
    return mp.fsum(trP(g, t) for g in gaps) + trP(n - pos[-1], t) - trC(n, t)

for t in ["0", "1/1000", "1/4", "1/2", "1", "2", "4"]:
    tt = mpq(Q(t))
    for eps in ["1e-6", "1e-12", "1e-20"]:
        c = relative_heat(CutLineDefect(), t, eps, max_steps=1000)
        chk(f"E t={t} eps={eps}", c.interval, (1 - mp.e ** (-4 * tt)) / 2)
        if Q(t) <= 2:
            c = relative_heat(CutLineDefect() * Line(), t, eps, max_steps=1000)
            chk(f"EL t={t} eps={eps}", c.interval, (1 - mp.e ** (-4 * tt)) / 2 * hline(tt))
        for pos in [(0, 1), (0, 2), (0, 2, 7), (0, 3, 4, 9)]:
            if Q(t) > 2 and len(pos) > 2: continue
            c = relative_heat(LineCutDefect(pos), t, eps, max_steps=1000)
            chk(f"cuts {pos} t={t} eps={eps}", c.interval, cuts_value(pos, tt))
            c = relative_heat(connected_cut_interaction(pos), t, eps, max_steps=1000)
            k = len(pos)
            val = mp.fsum((-1) ** (k - s) * cuts_value(S, tt) for s in range(1, k + 1) for S in itertools.combinations(pos, s))
            chk(f"connected {pos} t={t} eps={eps}", c.interval, val)
        for ell in (1, 2, 5):
            c = relative_heat(TwoCutLineDefect(ell), t, eps, max_steps=1000)
            chk(f"twocut {ell} t={t}", c.interval, cuts_value((0, ell), tt))
            c = relative_heat(CutInteraction(ell), t, eps, max_steps=1000)
            chk(f"I {ell} t={t}", c.interval, cuts_value((0, ell), tt) - 2 * cuts_value((0,), tt))
print("line cut closed forms done; bad:", len(bad))

# prepared queries
for pos in [(0,), (0, 2, 7)]:
    geo = PreparedLocal(LineCutDefect(pos), radius=16)
    pr = PreparedRelativeHeat(geo, max_time=2, epsilon="1e-8")
    for t in ["0", "1/8", "1/4", "1/2", "3/4", "1", "3/2", "2"]:
        try:
            c = pr.evaluate(t)
        except Exception as e:
            print("prepared", pos, t, type(e).__name__, e); continue
        chk(f"prepared {pos} t={t}", c.interval, cuts_value(pos, mpq(Q(t))))
print("prepared done; bad:", len(bad))

random.seed(11)
for trial in range(150):
    n = random.randint(2, 12)
    G = nx.gnp_random_graph(n, random.random(), seed=random.randint(0, 10**9))
    g = from_nx(G)
    allpairs = list(itertools.combinations(range(n), 2))
    m = random.randint(1, min(4, len(allpairs)))
    chosen = random.sample(allpairs, m)
    edits = [(u, v, -1 if G.has_edge(u, v) else 1) for u, v in chosen]
    H = G.copy()
    for u, v, s in edits:
        (H.remove_edge if s < 0 else H.add_edge)(u, v)
    t = random.choice(["1/9", "1/2", "1", "5/2", "6"])
    tt = mpq(Q(t))
    eps = random.choice(["1e-6", "1e-14", "1e-25"])
    normalize = random.random() < 0.3
    X = SparseEdgeDifference(g, edits, normalize=normalize)
    true = (heat_trace(H, tt) - heat_trace(G, tt)) / (n if normalize else 1)
    c = relative_heat(X, t, eps, max_steps=2000)
    chk(f"SED trial {trial}", c.interval, true)
    # also product with finite graph
    if n <= 6:
        F = nx.gnp_random_graph(random.randint(1, 3), 0.8, seed=random.randint(0, 10**9))
        Y = Finite.from_graph(from_nx(F), True)
        tr = (heat_trace(nx.cartesian_product(H, F), tt) - heat_trace(nx.cartesian_product(G, F), tt)) / F.number_of_nodes() / (n if normalize else 1)
        c = relative_heat(X * Y, t, eps, max_steps=2000)
        chk(f"SED*Y trial {trial}", c.interval, tr)
print("random SED done; bad:", len(bad))
