import time
from graphlocal import Line, Finite, path
from graphlocal.graphs import isomorphic, IsoGraph
import graphlocal.graphs as G
L = Line()
A = (L * L) * L
B = L * (L * L)
for r in (3, 4, 5):
    t0 = time.time()
    a, b = A.local(r), B.local(r)
    (ka,), (kb,) = a.values, b.values
    t1 = time.time()
    same_labels = ka.graph == kb.graph
    d = (A - B).local(r)
    print(f"r={r}: n={ka.graph.n} labeled-equal={same_labels} (A-B).local is zero: {len(d.values)==0}  time {time.time()-t1:.2f}s; isomorphic() cache info {G.isomorphic.cache_info().hits} hits")
    # count visits used
    calls = {"n": 0}
