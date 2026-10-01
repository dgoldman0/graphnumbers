import networkx as nx, indep
from fractions import Fraction as Fr
# (8): T_R(P_{2R+1}-P_{2R}) = delta(centered P_{2R+1}); C_{2R+2}/(2R+2) gives the same
for R in range(1, 8):
    tgt = indep.lin_hist([(1, nx.path_graph(2*R+1))], R)
    centered = indep.REG.register(indep.ball(nx.path_graph(2*R+1), R, R))
    d = indep.lin_hist([(1, nx.path_graph(2*R+1)), (-1, nx.path_graph(2*R))], R)
    c = indep.lin_hist([(Fr(1, 2*R+2), nx.cycle_graph(2*R+2))], R)
    c2 = indep.lin_hist([(Fr(1, 2*R+1), nx.cycle_graph(2*R+1))], R)
    print(R, d == {centered: 1}, c == {centered: 1}, "C_{2R+1} differs:", c2 != {centered: 1})
# uniqueness: rank of radius-R histograms of all connected graphs on <= 2R+1 vertices
for R in (1, 2, 3):
    gs = indep.connected_graphs(2*R+1)
    hs = [indep.hist(g, R) for g in gs]
    types = sorted(set().union(*hs))
    rows = [[h.get(t, 0) for h in hs] for t in types]
    rk = indep.rank_mod_p(rows)
    print("R", R, "graphs", len(gs), "rank", rk, "full column rank:", rk == len(gs))
