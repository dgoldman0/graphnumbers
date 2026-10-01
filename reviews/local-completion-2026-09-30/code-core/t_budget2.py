import time
from graphlocal import Line, BudgetExceeded
from graphlocal.graphs import isomorphic
L = Line()
for r in (4, 5):
    (ka,) = ((L * L) * L).local(r).values
    (kb,) = (L * (L * L)).local(r).values
    n = ka.graph.n
    for budget in (n - 1, n, n + 1):
        t0 = time.time()
        try:
            res = isomorphic(ka.graph, kb.graph, True, budget)
        except BudgetExceeded:
            res = "BudgetExceeded"
        print(f"r={r} n={n} search_budget={budget}: {res} in {time.time()-t0:.2f}s")
