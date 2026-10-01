import random
import networkx as nx
import mpmath as mp
from core import lap, norm_edge
mp.mp.dps = 40
random.seed(5)
def heat(n, edges, cuts, t):
    k = len(cuts); tot = mp.mpf(0)
    for mask in range(1 << k):
        rem = set(cuts[i] for i in range(k) if mask >> i & 1)
        L = mp.matrix(lap(n, [e for e in edges if e not in rem]))
        ev, _ = mp.eigsy(L)
        tot += (-1)**(k - bin(mask).count("1")) * sum(mp.e**(-t*x) for x in ev)
    return tot
def ptail(lam, q):
    if q <= 0: return mp.mpf(1)
    return 1 - mp.e**(-lam)*sum(lam**i/mp.factorial(i) for i in range(q))
viol = 0; cnt = 0
for trial in range(120):
    n = random.randint(4, 10)
    T = nx.random_labeled_tree(n, seed=random.randint(0, 10**6))
    edges = sorted(norm_edge(e) for e in T.edges())
    D = max(d for _, d in T.degree())
    k = random.randint(2, min(5, len(edges)))
    cuts = random.sample(edges, k)
    # minimal subtree
    verts = sorted(set(x for e in cuts for x in e)); E = set(cuts)
    for v in verts[1:]:
        p = nx.shortest_path(T, verts[0], v); E |= set(norm_edge(e) for e in zip(p, p[1:]))
    deg = {}
    for a, b in E: deg[a] = deg.get(a, 0)+1; deg[b] = deg.get(b, 0)+1
    s = len(E); ell = sum(1 for d in deg.values() if d == 1)
    for t in [mp.mpf('0.2'), mp.mpf(1), mp.mpf(4)]:
        h = heat(n, edges, cuts, t); cnt += 1
        b = 2**ell * t**ell * ptail(t*D, 2*(s-ell))
        if abs(h) > b: viol += 1; print("VIOL (16)", edges, cuts, t, h, b)
print("checks", cnt, "violations", viol)
