import sys, random, time
import numpy as np, networkx as nx, indep
from fractions import Fraction as Fr
from scipy.optimize import linprog
sys.path.insert(0, sys.argv[1])
import reconstruct_local as RL
from verify_local_algebra import TYPES, hist as rhist, Graph

def to_nx(g):
    G = nx.Graph(); G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v in g.neighbors(u):
            G.add_edge(u, v)
    return G

def lp_opt(cols_hist, sizes, target):
    types = sorted(set(target).union(*[set(h) for h in cols_hist]))
    H = np.array([[float(h.get(t, 0)) for h in cols_hist] for t in types])
    u = np.array([float(target.get(t, 0)) for t in types])
    n = len(cols_hist)
    w = np.array(sizes, float)
    res = linprog(np.concatenate([w, w]), A_eq=np.hstack([H, -H]), b_eq=u,
                  bounds=[(0, None)] * (2 * n), method='highs')
    return res

rng = random.Random(1)
stats = []
for (maxn, maxdeg, R) in [(4, None, 1), (5, None, 1), (5, None, 2), (5, 3, 2), (6, 2, 2), (6, 3, 1), (6, None, 3), (5, 2, 0), (6, None, 1)]:
    graphs = RL.catalog(maxn, maxdeg)
    nxg = [to_nx(g) for g in graphs]
    mine = [indep.hist(G, R) for G in nxg]
    sizes = [g.n for g in graphs]
    for trial in range(6):
        k = rng.randint(1, 4)
        picks = rng.sample(range(len(graphs)), k)
        coeffs = {j: Fr(rng.randint(-5, 5) or 1, rng.randint(1, 4)) for j in picks}
        tgt_repo = {}
        for j, c in coeffs.items():
            for t, m in rhist(graphs[j], R).items():
                tgt_repo[t] = tgt_repo.get(t, 0) + c * m
        tgt_repo = {t: v for t, v in tgt_repo.items() if v}
        tgt_mine = {}
        for j, c in coeffs.items():
            for t, m in mine[j].items():
                tgt_mine[t] = tgt_mine.get(t, 0) + c * m
        tgt_mine = {t: v for t, v in tgt_mine.items() if v}
        t0 = time.time()
        try:
            res = RL.reconstruct(tgt_repo, graphs, R, search_budget=int(sys.argv[2]))
            status = 'opt'; cost = res['cost']; work = res['search_work']; rank = res['rank']
        except RL.BudgetExceeded as e:
            status = 'budget'; cost = None; work = None; rank=None
        dt = time.time() - t0
        lp = lp_opt(mine, sizes, tgt_mine)
        ok = (status != 'opt') or abs(float(cost) - lp.fun) < 1e-7
        stats.append((maxn, maxdeg, R, len(graphs), rank, status, str(cost), round(lp.fun, 6), ok, work, round(dt, 2))); print(stats[-1], flush=True)
        if not ok:
            print("DISAGREE", stats[-1], coeffs)

