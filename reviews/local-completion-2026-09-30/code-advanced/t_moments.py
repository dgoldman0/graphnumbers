import random
from fractions import Fraction as Q
from math import comb
from graphlocal import *
from graphlocal.interaction_moments import interaction_moments, tree_interaction_leading
from graphlocal.interaction_bounds import interaction_geometry
import oracle

random.seed(1)
def random_graph(n, p):
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if random.random() < p]
    return graph(n, edges)

fails = 0
cases = 0
for trial in range(150):
    n = random.randint(2, 7)
    g = random_graph(n, random.choice([0.3, 0.5, 0.7]))
    present = [(u, v) for u in range(n) for v in range(u + 1, n) if g.rows[u] >> v & 1]
    absent = [(u, v) for u in range(n) for v in range(u + 1, n) if not g.rows[u] >> v & 1]
    k = random.randint(1, 4)
    pool = [(u, v, -1) for u, v in present] + [(u, v, 1) for u, v in absent]
    if len(pool) < k:
        continue
    edits = random.sample(pool, k)
    # randomize orientation of stored endpoints
    edits = [(v, u, s) if random.random() < 0.5 else (u, v, s) for u, v, s in edits]
    normalize = random.random() < 0.3
    reduce = random.random() < 0.7
    try:
        x = EdgeInteraction(g, edits, normalize=normalize, reduce_bridges=reduce)
    except Exception as e:
        print("construct exc", e); continue
    order = 9
    data = interaction_moments(x, order)
    canon = [(min(u, v), max(u, v), s) for u, v, s in edits]
    ref = oracle.laplacian_moments(g.rows, canon, order)
    scale = Q(1, n) if normalize else Q(1)
    ref = [scale * r for r in ref]
    cases += 1
    if list(data.laplacian) != ref:
        fails += 1
        print("LAPLACIAN MISMATCH", g.rows, edits, normalize, reduce, data.laplacian, ref)
    D = x.degree_bound
    if D:
        lz = oracle.lazy_moments(g.rows, canon, D, order)
        lz = [scale * r for r in lz]
        if list(data.returns) != lz:
            fails += 1
            print("RETURNS MISMATCH", g.rows, edits, data.returns, lz)
        # geometry bounds
        geo = interaction_geometry(x)
        for j, m in enumerate(lz):
            b = geo.moment_bound(j)
            prof = sum(a * Q(comb(j, r) * __import__('math').factorial(r)) / D ** r
                       for r, a in enumerate(geo.moment_profile) if r <= j)
            if abs(m) > b or b > prof:
                fails += 1
                print("GEOMETRY BOUND FAIL", g.rows, edits, j, m, b, prof)
            if geo.vanishing_order is not None and j < geo.vanishing_order and m != 0:
                fails += 1
                print("VANISHING FAIL", g.rows, edits, j, m, geo.vanishing_order)
        # also check the profile at larger D (claimed valid for D >= D0)
        for D2 in (D + 1, D + 3):
            lz2 = [scale * r for r in oracle.lazy_moments(g.rows, canon, D2, order)]
            for j, m in enumerate(lz2):
                prof = sum(a * Q(comb(j, r) * __import__('math').factorial(r)) / D2 ** r
                           for r, a in enumerate(geo.moment_profile) if r <= j)
                prof_x = sum(a * Q(comb(j, r) * __import__('math').factorial(r)) / D2 ** r
                           for r, a in enumerate(moment_profile(x)) if r <= j)
                if abs(m) > prof or abs(m) > prof_x:
                    fails += 1
                    print("PROFILE@D2 FAIL", g.rows, edits, D2, j, m, prof, prof_x)
print("cases", cases, "fails", fails)
