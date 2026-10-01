import time, faulthandler
from canon import *
from atoms import *
import canon
faulthandler.dump_traceback_later(100, exit=True)
reg = Registry()
orig_cid = Registry.cid
def cid_logged(self, adj, dist):
    t = time.time()
    G = self._nxg(adj, dist)
    res = orig_cid(self, adj, dist)
    dt = time.time() - t
    if dt > 1:
        print("slow cid", dt, "n", len(adj), "maxdist", max(dist), flush=True)
    return res
Registry.cid = cid_logged
# replicate check3 prefix
for r in (2, 3):
    G = edge_centered_tree(3, 2 * r - 1); Gc = delete_edge(G, 0, 1)
    n = 4 * r + 6
    Pn, Cn = path_graph(n), cycle_graph(n)
    direct = defaultdict(F) if False else defaultdict(int)
    for (X, sx) in ((Gc, 1), (G, -1)):
        for (Y, sy) in ((Pn, 1), (Cn, -1)):
            histogram(reg, cartesian(X, Y), r, sx * sy, None, direct)
    print("done direct r", r, len(reg.reps), flush=True)
    if r == 2:
        for (X, sx) in ((Gc, 1), (G, -1)):
            for (Y, sy) in ((Gc, 1), (G, -1)):
                histogram(reg, cartesian(X, Y), r, sx * sy, None, direct)
print("registry", len(reg.reps), flush=True)
# now the d=3 r=2 powers with logging of the bucket
r, d = 2, 3
h, at = tree_cut(reg, d, r)
cur = {reg.cid({0: set()}, [0]): 1}
for k in range(1, 5):
    nxt = {}
    for c1, v1 in cur.items():
        for c2, v2 in h.items():
            t = time.time()
            # compute the ball and inspect bucket before registering
            (a1, d1), (a2, d2) = reg.reps[c1], reg.reps[c2]
            verts = [(x, y) for x in a1 for y in a2 if d1[x] + d2[y] <= r]
            vs = set(verts)
            adj = {v: set() for v in verts}
            for (x, y) in verts:
                for x2 in a1[x]:
                    if (x2, y) in vs: adj[(x, y)].add((x2, y))
                for y2 in a2[y]:
                    if (x, y2) in vs: adj[(x, y)].add((x, y2))
            b, dd = ball(adj, (0, 0), r)
            print(k, c1, c2, "ball n", len(b), "m", sum(len(s) for s in b.values())//2, flush=True)
            p = reg.star(c1, c2, r)
            nxt[p] = nxt.get(p, 0) + v1 * v2
    cur = {c: v for c, v in nxt.items() if v}
    print("power", k, len(cur), flush=True)
