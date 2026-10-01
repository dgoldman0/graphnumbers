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

if __name__ != "__main__": raise SystemExit
viol = 0; tight = []; tests = 0; extra_vanish = 0
for trial in range(200):
    n = random.randint(3, 8)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    edges = [e for e in pairs if random.random() < 0.4]
    k = random.randint(1, 4)
    chosen = random.sample(pairs, min(k, len(pairs)))
    edits = [(u, v, -1 if (u, v) in edges else 1) for u, v in chosen]
    D, delta, taus = geometry(n, edges, edits)
    if D == 0: continue
    order = 12
    d = unif_moments(n, edges, edits, D, order)
    tests += 1
    if taus is None:
        if any(d): viol += 1; print("nonzero though disconnected", n, edges, edits, d)
        continue
    taustar = min(taus)
    for j in range(order+1):
        b = bound6(j, len(edits), D, taus)
        if abs(d[j]) > b:
            viol += 1; print("VIOL (6)", n, edges, edits, j, d[j], b)
    first = next((j for j, x in enumerate(d) if x), None)
    if first is not None and first < len(edits) + taustar:
        viol += 1; print("VIOL (7)", n, edges, edits, first, taustar)
    if first is not None and first > len(edits) + taustar: extra_vanish += 1
    # heat bounds (9) and (14) at a few times; also profile (13) at larger D
    kk = len(edits)
    for t in [mp.mpf('0.1'), mp.mpf('0.5'), mp.mpf(1), mp.mpf(3)]:
        h = heat_exact(n, edges, edits, t)
        lam = t*D
        b9 = 2**kk * t**kk / mp.factorial(kk-1) * sum(poisson_tail(lam, tau) for tau in taus)
        b9b = 2**kk * t**kk * poisson_tail(lam, taustar)
        b14 = sum(2**kk * mp.mpf(D)**tau * t**(kk+tau) / mp.factorial(kk+tau-1) for tau in taus)
        bmix = sum(min(2**kk * t**kk / mp.factorial(kk-1) * poisson_tail(lam, tau),
                       2**kk * mp.mpf(D)**tau * t**(kk+tau) / mp.factorial(kk+tau-1)) for tau in taus)
        for name, b in [("9a", b9), ("9b", b9b), ("14", b14), ("mix", bmix)]:
            if abs(h) > b * (1 + mp.mpf('1e-30')):
                viol += 1; print("VIOL heat", name, n, edges, edits, t, h, b)
        # tail (10) after M
        for M in [0, 2, 5, 9]:
            # retained partial with exact moments d_j up to M
            if M > order: continue
            retained = mp.e**(-lam) * sum(lam**j * mp.mpf(d[j].numerator)/d[j].denominator / mp.factorial(j) for j in range(M+1))
            tail = h - retained
            b10 = 2**kk * t**kk / mp.factorial(kk-1) * sum(poisson_tail(lam, max(tau, M+1-kk)) for tau in taus)
            if abs(tail) > b10 * (1 + mp.mpf('1e-30')) + mp.mpf('1e-40'):
                viol += 1; print("VIOL (10)", n, edges, edits, t, M, tail, b10)
    # profile (13) at D0=D, check at D'=D+3 using uniformized moments at D'
    D2 = D + 3
    d2 = unif_moments(n, edges, edits, D2, 10)
    A = {}
    for tau in taus:
        r = kk + tau
        A[r] = A.get(r, Q(0)) + Q(2**kk * D**tau, factorial(kk + tau - 1))
    for j in range(11):
        b = sum(a * Q(factorial(j), factorial(j - r)) / Q(D2)**r for r, a in A.items() if r <= j)
        if abs(d2[j]) > b:
            viol += 1; print("VIOL (13) at larger D", n, edges, edits, j, d2[j], b)
print("tests", tests, "violations", viol, "cases with onset beyond geometric order", extra_vanish)
