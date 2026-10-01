"""Enumerate all rooted r-ball types of maximum degree <= D (independent generator)."""
import sys, itertools
sys.path.insert(0, '.')
from glib import *


def types_D2(r):
    out = []
    for a in range(r + 1):
        for b in range(a, r + 1):
            # path with a vertices on one side and b on the other, root in position a
            n = a + b + 1
            g = path(n)
            out.append(ball(g, a, r))
    for m in range(3, 2 * r + 2):
        out.append(ball(cycle(m), 0, r))
    return dedupe(out)


def dedupe(gs):
    seen = {}
    for g in gs:
        c = canon(g, 0)
        seen.setdefault(c, g)
    return list(seen.values())


def graphs_on(n, maxd):
    pairs = list(itertools.combinations(range(n), 2))
    res = []
    for mask in range(1 << len(pairs)):
        es = [pairs[j] for j in range(len(pairs)) if mask >> j & 1]
        deg = [0] * n
        ok = True
        for u, v in es:
            deg[u] += 1; deg[v] += 1
        if max(deg, default=0) <= maxd:
            res.append(es)
    return res


def types_r1(D):
    out = []
    for d in range(D + 1):
        for es in graphs_on(d, D - 1):
            g = mk(d + 1, [(0, i) for i in range(1, d + 1)] + [(u + 1, v + 1) for u, v in es])
            out.append(g)
    return dedupe(out)


def types_r2(D):
    out = {}
    for d1 in range(D + 1):
        L1 = list(range(1, d1 + 1))
        for es11 in graphs_on(d1, D - 1):
            deg = {0: d1}
            for v in L1:
                deg[v] = 1
            for u, v in es11:
                deg[u + 1] += 1; deg[v + 1] += 1
            cap = {v: D - deg[v] for v in L1}
            subsets = [A for k in range(1, d1 + 1) for A in itertools.combinations(L1, k)]
            # multisets of attachment sets, as nondecreasing index sequences
            def msets(start, cap, chosen):
                yield list(chosen)
                for i in range(start, len(subsets)):
                    A = subsets[i]
                    if all(cap[v] >= 1 for v in A) and len(A) <= D:
                        for v in A:
                            cap[v] -= 1
                        chosen.append(A)
                        yield from msets(i, cap, chosen)
                        chosen.pop()
                        for v in A:
                            cap[v] += 1
            for L2att in msets(0, dict(cap), []):
                n = 1 + d1 + len(L2att)
                base = [(0, v) for v in L1] + [(u + 1, v + 1) for u, v in es11]
                for j, A in enumerate(L2att):
                    for v in A:
                        base.append((v, 1 + d1 + j))
                d2cap = [D - len(A) for A in L2att]
                for es22 in graphs_on_caps(d2cap):
                    edges = base + [(1 + d1 + u, 1 + d1 + v) for u, v in es22]
                    g = mk(n, edges)
                    c = canon(g, 0)
                    if c not in out:
                        out[c] = g
    return list(out.values())


_cache = {}


def graphs_on_caps(caps):
    key = tuple(caps)
    if key in _cache:
        return _cache[key]
    n = len(caps)
    pairs = list(itertools.combinations(range(n), 2))
    res = []

    def rec(i, deg, chosen):
        if i == len(pairs):
            res.append(list(chosen)); return
        rec(i + 1, deg, chosen)
        u, v = pairs[i]
        if deg[u] < caps[u] and deg[v] < caps[v]:
            deg[u] += 1; deg[v] += 1; chosen.append((u, v))
            rec(i + 1, deg, chosen)
            chosen.pop(); deg[u] -= 1; deg[v] -= 1
    rec(0, [0] * n, [])
    _cache[key] = res
    return res


if __name__ == '__main__':
    import time
    for r in range(1, 5):
        print('D=2 r=%d types=%d' % (r, len(types_D2(r))))
    for D in range(1, 5):
        print('D=%d r=1 types=%d' % (D, len(types_r1(D))))
    t = time.time()
    T = types_r2(3)
    print('D=3 r=2 types=%d (%.1fs)' % (len(T), time.time() - t))
    T2 = types_r2(2)
    print('D=2 r=2 via layered generator types=%d' % len(T2))
