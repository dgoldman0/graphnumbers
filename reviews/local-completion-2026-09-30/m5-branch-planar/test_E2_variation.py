import itertools, time
import networkx as nx
from collections import defaultdict
from gcore import Registry, rooted_ball, clean
from math import comb

reg = Registry()

def factor(kind, n):
    return nx.path_graph(n) if kind == 'P' else nx.cycle_graph(n)

def product(graphs):
    G = graphs[0]
    for H in graphs[1:]:
        G = nx.cartesian_product(G, H)
    return G

def T_r_power(k, r, n):
    acc = defaultdict(int)
    for kinds in itertools.product('PC', repeat=k):
        sign = (-1) ** kinds.count('C')
        G = product([factor(t, n) for t in kinds])
        for o in G.nodes():
            B = rooted_ball(G, o, r)
            acc[reg.key(B, o)] += sign
    return clean(acc)

results = {}
for k, rs in ((2, (1, 2, 3, 4)), (3, (1, 2))):
    for r in rs:
        t0 = time.time()
        h1 = T_r_power(k, r, 2 * r + 2)
        h2 = T_r_power(k, r, 2 * r + 3)
        var = sum(abs(c) for c in h1.values())
        print(f"E^{k} r={r}: stable(n=2r+2 vs 2r+3)={h1==h2}, variation={var}, predicted (4r)^k={(4*r)**k}, "
              f"types={len(h1)}, predicted C(r+k,k)={comb(r+k,k)}, coeffs={sorted(h1.values())}, time={time.time()-t0:.1f}s", flush=True)
        results[(k, r)] = h1

# r=1 identity for E^2: 4 K12 - 8 K13 + 4 K14 rooted at centers
h = results[(2, 1)]
star = lambda j: reg.key(nx.star_graph(j), 0)
print("T_1(E^2) =", {('K1,%d' % j): h.get(star(j), 0) for j in (1, 2, 3, 4)}, "other types:", [k for k in h if k not in (star(2), star(3), star(4))])
