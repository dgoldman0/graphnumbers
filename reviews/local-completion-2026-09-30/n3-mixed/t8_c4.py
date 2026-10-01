"""Gamma (no common non-root neighbour) == 'root edges lie in no simple 4-cycle together' via explicit cycle enumeration."""
import sys, itertools, random
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm import *
exec(open(__file__.rsplit("/",1)[0] + "/t2_adversarial.py").read().split("F = fixtures()")[0].split("random.seed(20261001)")[1])
F = fixtures()
bad = cnt = 0
for name, g in F.items():
    for u in g:
        for r in (2, 3):
            B = ball(g, u, r); o = root_of(B); N = list(B[o])
            G1 = Gamma(B)
            for i, j in itertools.combinations(range(len(N)), 2):
                a, b = N[i], N[j]
                # simple 4-cycles containing edges o-a and o-b: o-a-x-b-o with x distinct from o,a,b
                in4 = any(x not in (o, a, b) and B.has_edge(a, x) and B.has_edge(x, b) for x in B.nodes)
                cnt += 1
                if G1.has_edge(i, j) == in4:
                    bad += 1
print("pairs checked", cnt, "mismatches", bad)
