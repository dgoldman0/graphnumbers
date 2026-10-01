import time
import networkx as nx
from common import *
from graphlocal import BudgetExceeded
from graphlocal.graphs import isomorphic
for n, d in [(40, 4), (80, 4), (160, 4)]:
    G = from_nx(nx.random_regular_graph(d, n, seed=1)); H = from_nx(nx.random_regular_graph(d, n, seed=2))
    for budget in (1000, 4000):
        t0 = time.time()
        try:
            r = isomorphic(G, H, False, budget)
        except BudgetExceeded:
            r = "BudgetExceeded"
        dt = time.time() - t0
        print(f"n={n} d={d} budget={budget}: {r} in {dt:.2f}s  -> projected default budget (100000): {dt*100000/budget/60:.1f} min")
