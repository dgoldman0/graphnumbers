import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm import *
r = 3
P = T(Cut("P", *grid_with_edge(2*r+1)), r)
items = list(P.items())
t0 = time.time()
for i, (t1, a) in enumerate(items):
    for j, (t2, b) in enumerate(items):
        s = time.time()
        X = star_product(REG.reps[t1], REG.reps[t2], r)
        s2 = time.time()
        k = REG.key(X)
        s3 = time.time()
        tid = REG.id(X)
        s4 = time.time()
        print(i, j, X.number_of_nodes(), f"build {s2-s:.3f} key {s3-s2:.3f} id {s4-s3:.3f} bucket {len(REG.buckets[k])}", flush=True)
print("total", time.time()-t0, "iso calls", REG.iso_calls)
