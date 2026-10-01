import random
import numpy as np
import networkx as nx
from fractions import Fraction as Q
from math import factorial
from tree_exhaustive import lap_np, traces, predicted
random.seed(31337)
bad = 0; cnt = 0; maxp = 0
for trial in range(150):
    n = random.randint(8, 18)
    # bias toward high degree: preferential attachment tree
    edges = []
    for v in range(1, n):
        if random.random() < 0.5:
            u = 0 if random.random() < 0.5 else random.randrange(v)
        else:
            u = random.randrange(v)
        edges.append((u, v))
    T = nx.Graph(edges)
    k = random.randint(2, 7)
    cuts = random.sample(edges, k)
    nu, coef = predicted(T, cuts)
    order = nu
    tot = [0]*(order+1)
    for mask in range(1 << k):
        rem = set(cuts[i] for i in range(k) if mask >> i & 1)
        tr = traces(lap_np(n, [e for e in edges if e not in rem]), order)
        sg = (-1)**(k - bin(mask).count("1"))
        for j in range(order+1): tot[j] += sg*tr[j]
    first = next((j for j, x in enumerate(tot) if x), None)
    lead = Q((-1)**first * tot[first], factorial(first)) if first is not None else None
    cnt += 1
    maxp = max(maxp, abs(coef)*factorial(nu-1))
    if first != nu or lead != coef:
        bad += 1; print("MISMATCH", n, edges, cuts, first, lead, nu, coef)
print("random large trees", cnt, "bad", bad, "max p(T)", maxp)
