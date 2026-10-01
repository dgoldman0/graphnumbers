from fractions import Fraction as Q
import time
from graphlocal import *
from graphlocal.defects import _relative_plan
from graphlocal.graphs import isomorphic, induced, IsoGraph
import random

# 1. Plan monotonicity in t
viol = 0
for D in (2, 3, 4, 6):
    for q in (Q(1), Q(4), Q(1, 100)):
        for eps in (Q("1e-4"), Q("1e-8"), Q("1e-14")):
            for T in (Q(1, 2), Q(2), Q(5)):
                top = _relative_plan(T, D, q, eps, 2000)[0]
                for i in range(1, 400):
                    t = T * i / 400
                    s = _relative_plan(t, D, q, eps, 2000)[0]
                    if s > top:
                        viol += 1
                        if viol < 5: print("non-monotone", D, q, eps, T, t, s, top)
print("plan monotonicity violations:", viol)

# 2. Mappingproxy equality semantics
a = Line().local(2); b = (Finite.from_graph(path(5)) - Finite.from_graph(path(4))).local(2)
print("LocalHistogram eq:", a == b, a == CutLineDefect().local(2), a != b)

# 3. Isomorphism work per budget unit: large isomorphic balls of L^3
L3 = Line() ** 3
for r in (3, 4, 5, 6):
    h = L3.local(r)
    (key,) = h.values
    g = key.graph
    perm = [0] + random.sample(range(1, g.n), g.n - 1)
    g2 = induced(g, perm)
    t0 = time.time()
    try:
        res = isomorphic(g, g2, True)
    except BudgetExceeded:
        res = "budget"
    print(f"L^3 ball r={r} n={g.n}: iso={res} time={time.time()-t0:.2f}s")
