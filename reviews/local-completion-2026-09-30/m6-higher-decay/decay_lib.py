import random
from fractions import Fraction as Q
from itertools import permutations
from math import comb, factorial
from collections import deque
import mpmath as mp
from core import lap, norm_edge
mp.mp.dps = 50
random.seed(7)

def apply(edges, edits, mask):
    E = set(norm_edge(e) for e in edges)
    for i, (u, v, s) in enumerate(edits):
        if mask >> i & 1:
            e = norm_edge((u, v))
            if s == -1: E.remove(e)
            else: E.add(e)
    return sorted(E)

def unif_moments(n, edges, edits, D, order):
    k = len(edits); res = [Q(0)]*(order+1)
    for mask in range(1 << k):
        L = lap(n, apply(edges, edits, mask))
        P = [[Q(int(i==j)) - Q(L[i][j], D) for j in range(n)] for i in range(n)]
        Pw = [[Q(int(i==j)) for j in range(n)] for i in range(n)]
        sg = (-1)**(k - bin(mask).count("1"))
        for j in range(order+1):
            res[j] += sg * sum(Pw[i][i] for i in range(n))
            Pw = [[sum(Pw[i][l]*P[l][jj] for l in range(n)) for jj in range(n)] for i in range(n)]
    return res

def heat_exact(n, edges, edits, t):
    k = len(edits); tot = mp.mpf(0)
    for mask in range(1 << k):
        L = mp.matrix(lap(n, apply(edges, edits, mask)))
        ev, _ = mp.eigsy(L)
        tot += (-1)**(k - bin(mask).count("1")) * sum(mp.e**(-t*x) for x in ev)
    return tot

def bfs_set(n, adj, src):
    dist = {s: 0 for s in src}; dq = deque(src)
    while dq:
        x = dq.popleft()
        for y in adj[x]:
            if y not in dist:
                dist[y] = dist[x] + 1; dq.append(y)
    return dist

def geometry(n, edges, edits):
    U = set(norm_edge(e) for e in edges) | set(norm_edge((u, v)) for u, v, s in edits if s == 1)
    adj = {i: [] for i in range(n)}
    for u, v in U: adj[u].append(v); adj[v].append(u)
    D = max((len(adj[i]) for i in range(n)), default=0)
    k = len(edits)
    delta = [[None]*k for _ in range(k)]
    for i, (u, v, _) in enumerate(edits):
        d = bfs_set(n, adj, [u, v])
        for j, (x, y, _) in enumerate(edits):
            if x in d: delta[i][j] = min(d[x], d[y])
    for i in range(k): delta[i][i] = 0
    taus = []
    if any(delta[i][j] is None for i in range(k) for j in range(k)):
        return D, delta, None
    for rest in permutations(range(1, k)):
        o = (0,) + rest
        taus.append(sum(delta[o[i]][o[(i+1) % k]] for i in range(k)))
    return D, delta, taus

def bound6(j, k, D, taus):
    return Q(2**k, D**k) * j * sum(comb(j - t - 1, k - 1) for t in taus if j >= k + t)

def poisson_tail(lam, q):
    if q <= 0: return mp.mpf(1)
    return 1 - mp.e**(-lam) * sum(lam**i / mp.factorial(i) for i in range(q))

