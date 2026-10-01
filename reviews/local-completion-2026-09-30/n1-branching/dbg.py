import time
from canon import *
from atoms import *
import faulthandler, sys
faulthandler.dump_traceback_later(60, exit=True)
reg = Registry()
r, d = 2, 3
h, at = tree_cut(reg, d, r)
print(at, flush=True)
cur = {reg.cid({0: set()}, [0]): 1}
for k in range(1, 5):
    t = time.time()
    nxt = {}
    for c1, v1 in cur.items():
        for c2, v2 in h.items():
            t1 = time.time()
            p = reg.star(c1, c2, r)
            print(k, c1, c2, '->', p, 'size', reg.size(p), f'{time.time()-t1:.2f}s', flush=True)
            nxt[p] = nxt.get(p, 0) + v1 * v2
    cur = {c: v for c, v in nxt.items() if v}
    print('power', k, len(cur), f'{time.time()-t:.1f}s', flush=True)
