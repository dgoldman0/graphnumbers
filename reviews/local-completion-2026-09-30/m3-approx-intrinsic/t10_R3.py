import numpy as np, networkx as nx, indep
from scipy.optimize import linprog
from fractions import Fraction as Fr
reg = indep.Registry()
for R, maxn, maxdeg in [(1, 3, None), (1, 4, None), (2, 5, None), (2, 6, None), (3, 7, None), (2, 6, 3)]:
    gs = indep.connected_graphs(maxn, maxdeg)
    hs = [indep.hist(g, R, reg) for g in gs]
    tgt = {reg.register(indep.ball(nx.path_graph(2*R+1), R, R)): 1}
    types = sorted(set(tgt).union(*hs))
    H = np.array([[h.get(t, 0) for h in hs] for t in types], float)
    u = np.array([tgt.get(t, 0) for t in types], float)
    w = np.array([g.number_of_nodes() for g in gs], float)
    res = linprog(np.concatenate([w, w]), A_eq=np.hstack([H, -H]), b_eq=u, bounds=[(0, None)]*(2*len(gs)), method='highs')
    c = res.x[:len(gs)] - res.x[len(gs):]
    neg = sum(-ci*wi for ci, wi in zip(c, w) if ci < -1e-9)
    support = [(g.number_of_nodes(), g.number_of_edges(), round(ci, 6)) for g, ci in zip(gs, c) if abs(ci) > 1e-9]
    print(R, maxn, maxdeg, "graphs", len(gs), "min cost", round(res.fun, 6), "neg mass", round(neg, 6), "support", support)
