import time, random
from graphlocal import Finite, cycle
from graphlocal.graphs import induced, isomorphic
random.seed(0)
for n in (60, 120, 240):
    g = cycle(n); perm = list(range(n)); random.shuffle(perm); h = induced(g, perm)
    t0 = time.time()
    same = Finite.from_graph(g) == Finite.from_graph(h)
    print(f"Finite(C_{n}) == Finite(relabeled C_{n}): {same} in {time.time()-t0:.2f}s (search nodes needed ~{n})", flush=True)
