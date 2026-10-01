import itertools
import networkx as nx
from fractions import Fraction as Fr
from collections import defaultdict
from math import comb, factorial
import test_E2_variation as tv   # recomputes histograms (cached in module)

def returns(H, root, D, steps):
    nodes = list(H.nodes()); idx = {v: i for i, v in enumerate(nodes)}
    vec = [Fr(0)] * len(nodes); vec[idx[root]] = Fr(1)
    out = [Fr(1)]
    for _ in range(steps):
        new = [Fr(0)] * len(nodes)
        for v in nodes:
            i = idx[v]
            if vec[i] == 0: continue
            new[i] += vec[i] * (1 - Fr(H.degree(v), D))
            for w in H.neighbors(v):
                new[idx[w]] += vec[i] / D
        vec = new
        out.append(vec[idx[root]])
    return out

def moments(hist, D, steps):
    d = [Fr(0)] * (steps + 1)
    for key, c in hist.items():
        H, root = tv.reg.reps[key]
        assert max(dict(H.degree()).values()) <= D
        for j, p in enumerate(returns(H, root, D, steps)):
            d[j] += c * p
    return d

def odd_occupancy(j, k):
    tot = 0
    for ms in itertools.product(range(j + 1), repeat=k):
        if sum(ms) == j and all(m % 2 == 1 for m in ms):
            mult = factorial(j)
            for m in ms: mult //= factorial(m)
            tot += mult
    return Fr(tot, k ** j)

d2 = moments(tv.results[(2, 4)], 4, 8)
print("d_j(E^2;4), j=0..8:", [str(x) for x in d2])
print("(20) k=2:", [str(odd_occupancy(j, 2)) for j in range(9)])
print("bound (16) j(j-1)/4 respected:", all(abs(d2[j]) <= Fr(j*(j-1), 4) for j in range(9)))
d3 = moments(tv.results[(3, 2)], 6, 4)
print("d_j(E^3;6), j=0..4:", [str(x) for x in d3], " (20):", [str(odd_occupancy(j, 3)) for j in range(5)])
# also E^2 at cap 5 (non-minimal) vs binomial averaging from cap 4
d2_5 = moments(tv.results[(2, 4)], 5, 8)
w = Fr(4, 5)
avg = [sum(comb(j, m) * w**m * (1-w)**(j-m) * d2[m] for m in range(j+1)) for j in range(9)]
print("cap-5 moments equal binomial average of cap-4 moments:", d2_5 == avg)
