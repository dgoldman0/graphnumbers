"""T_r(E^n) from explicit products of P_m and C_m (m >= 2r+2), inclusion-exclusion,
exact rooted canonical forms.  Independent of the repo."""
import sys, time, itertools, json
from collections import Counter
from fractions import Fraction as Fr
from math import comb, factorial
from indep import *

def labelled_key(adj, verts, root, r, m, kinds):
    b, dist, order = ball(adj, root, r)
    v = verts[root]
    offs = []
    for idx in order:
        u = verts[idx]
        o = []
        for c,(a,bb) in enumerate(zip(u, v)):
            dlt = a - bb
            if kinds[c] == 'C':
                dlt %= m
                if dlt > m // 2: dlt -= m
            o.append(dlt)
        offs.append(tuple(o))
    edges = frozenset((offs[i], offs[j]) for i in range(len(b)) for j in b[i])
    return (frozenset(offs), edges), b

def T_product(k, n, r, m, cache):
    graphs = [path(m)] * k + [cycle(m)] * (n - k)
    kinds = ['P'] * k + ['C'] * (n - k)
    if n == 0:
        return Counter({canonical_form([set()]): 1})
    adj, verts = cartesian(graphs)
    H = Counter()
    for root in range(len(adj)):
        key, b = labelled_key(adj, verts, root, r, m, kinds)
        if key not in cache:
            cache[key] = canonical_form(b)
        H[cache[key]] += 1
    return H

def T_E_power(n, r, m):
    cache = {}
    total = Counter()
    for k in range(n + 1):
        Hk = T_product(k, n, r, m, cache)
        sgn = (-1) ** (n - k) * comb(n, k)
        for key, c in Hk.items():
            total[key] += sgn * c
    return {key: c for key, c in total.items() if c != 0}

def adj_from_cf(cf):
    n, edges = cf
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    return adj

if __name__ == '__main__':
    r = int(sys.argv[1]); nmax = int(sys.argv[2]); m = int(sys.argv[3]) if len(sys.argv) > 3 else 2*r+2
    supports = {}
    report = []
    for n in range(nmax + 1):
        t0 = time.time()
        T = T_E_power(n, r, m)
        l1 = sum(abs(c) for c in T.values())
        sizes = sorted({cf[0] for cf in T})
        row = {'r': r, 'n': n, 'm': m, 'types': len(T), 'l1': l1, 'pred_l1': (4*r)**n,
               'pred_types': comb(n + r, r), 'secs': round(time.time()-t0, 1)}
        if r >= 2:
            ok_coef = True; ok_coord = True
            for cf, c in T.items():
                adj = adj_from_cf(cf)
                coords, N = coordinates(adj, r)
                if N != n or any(x.denominator != 1 or x < 0 for x in coords) or sum(coords) != n:
                    ok_coord = False
                e = [int(x) for x in coords]
                pred = factorial(n)
                for x in e: pred //= factorial(x)
                pred *= 2**n * (-r)**e[-1]
                if pred != c: ok_coef = False
            row['coords_are_exponents'] = ok_coord
            row['coef_matches_multinomial'] = ok_coef
            row['distinct_coord_vectors'] = len({coordinates(adj_from_cf(cf), r)[0] for cf in T}) == len(T)
        else:
            row['support_degrees'] = sorted(len(adj_from_cf(cf)[0]) for cf in T)
            row['support_is_stars'] = all(all(len(adj_from_cf(cf)[v]) == 1 for v in range(1, cf[0])) for cf in T)
            row['coeffs_by_degree'] = {len(adj_from_cf(cf)[0]): c for cf, c in T.items()}
        supports[n] = set(T)
        report.append(row)
        print(row, flush=True)
    inter = {(a, b): len(supports[a] & supports[b]) for a in supports for b in supports if a < b}
    print('cross-power support intersections:', inter)
