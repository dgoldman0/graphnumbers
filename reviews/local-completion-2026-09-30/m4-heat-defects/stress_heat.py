import sys, random, itertools, time
sys.path.insert(0, '/home/user/graphnumbers/python/src')
import numpy as np, mpmath as mp
def M(x):
    from fractions import Fraction
    if isinstance(x, Fraction): return mp.mpf(x.numerator)/x.denominator
    return mp.mpf(x)
from fractions import Fraction as Q
from graphlocal import Finite, Line, heat_return, cycle, path, graph, cartesian
mp.mp.dps = 40

def to_nx_lap(g):
    n = g.n; L = np.zeros((n,n))
    for u in range(n):
        for v in g.neighbors(u):
            L[u,v] = -1
        L[u,u] = g.rows[u].bit_count()
    return L
def tr_heat(g, t):
    if g.n == 0: return mp.mpf(0)
    L = mp.matrix(to_nx_lap(g).tolist())
    ev = mp.eigsy(L, eigvals_only=True)
    return mp.fsum(mp.e**(-M(t)*e) for e in ev)
def hline(t):
    t = M(t); return mp.e**(-2*t)*mp.besseli(0,2*t)

def rand_graph(n, p, rng):
    edges = [(u,v) for u in range(n) for v in range(u+1,n) if rng.random()<p]
    # ensure connected-ish by adding a path
    edges += [(i,i+1) for i in range(n-1) if rng.random()<0.6]
    return graph(n, list(set(edges)))

rng = random.Random(7)
fails = []; n_tests = 0
times = [Q(0), Q(1,10), Q(1,2), Q(1), Q(2), Q(3)]
# 1. normalized random finite graphs and signed combinations
for trial in range(25):
    g1 = rand_graph(rng.randint(2,9), 0.35, rng); g2 = rand_graph(rng.randint(2,9), 0.5, rng)
    a, b, c = Q(rng.randint(-3,3), rng.randint(1,4)), Q(rng.randint(-3,3), rng.randint(1,3)), Q(rng.randint(-2,2))
    X = a*Finite.from_graph(g1, True) + b*Finite.from_graph(g2) + c
    for t in times[:5]:
        cert = heat_return(X, t, "1e-9")
        truth = mp.mpf(a.numerator)/a.denominator*tr_heat(g1,t)/g1.n + mp.mpf(b.numerator)/b.denominator*tr_heat(g2,t) + M(c)
        n_tests += 1
        lo, hi = mp.mpf(cert.interval.lower.numerator)/cert.interval.lower.denominator, mp.mpf(cert.interval.upper.numerator)/cert.interval.upper.denominator
        if not (lo - mp.mpf('1e-25') <= truth <= hi + mp.mpf('1e-25')) or cert.interval.radius > Q(1,10**9):
            fails.append(('finite', trial, t, float(lo), float(truth), float(hi)))
print('finite/signed tests', n_tests, 'failures', fails[:5])
# 2. lattices and products with finite graphs
fails2 = []; n2 = 0
for d in (1,2,3):
    X = Line()**d
    for t in times:
        if d == 3 and t > 2: continue
        cert = heat_return(X, t, "1e-8")
        truth = hline(t)**d
        lo, hi = [mp.mpf(x.numerator)/x.denominator for x in (cert.interval.lower, cert.interval.upper)]
        n2 += 1
        ok = lo <= truth <= hi
        if not ok: fails2.append((d, t, float(lo), float(truth), float(hi)))
        print(f'L^{d} t={t}: [{mp.nstr(lo,12)}, {mp.nstr(hi,12)}] truth {mp.nstr(truth,12)} ok={ok} steps={cert.steps} radius={cert.radius}')
for H in (path(2), cycle(3), path(3)):
    X = 3*Line()*Finite.from_graph(H, True) - 2*Line()
    for t in (Q(1,10), Q(1,2), Q(1)):
        cert = heat_return(X, t, "1e-8")
        truth = 3*hline(t)*tr_heat(H,t)/H.n - 2*hline(t)
        lo, hi = [mp.mpf(x.numerator)/x.denominator for x in (cert.interval.lower, cert.interval.upper)]
        n2 += 1
        if not (lo <= truth <= hi): fails2.append(('prod', H.rows, t))
print('lattice/product tests', n2, 'failures', fails2)
