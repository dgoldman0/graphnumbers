import random, itertools
from collections import Counter
import networkx as nx
from fractions import Fraction as Q
random.seed(11)

def comp_classes(L, reps):
    out = Counter()
    for c in nx.connected_components(L):
        C = L.subgraph(c)
        for i, D in enumerate(reps):
            if nx.is_isomorphic(C, D):
                out[i] += 1; break
        else:
            reps.append(C.copy()); out[len(reps) - 1] += 1
    return out

def cliques_through(G, o, q):
    N = list(G.neighbors(o))
    return sum(1 for S in itertools.combinations(N, q - 1) if all(G.has_edge(u, v) for u, v in itertools.combinations(S, 2)))

reps = []; bad = 0; trials = 0
for _ in range(60):
    gs = []
    for _ in range(2):
        while True:
            n = random.randint(1, 6)
            G = nx.gnp_random_graph(n, random.choice([0.4, 0.7, 0.9]), seed=random.randint(0, 10**9))
            if nx.is_connected(G): break
        gs.append(G)
    G, H = gs
    P = nx.cartesian_product(G, H)
    for (g, h) in P.nodes:
        trials += 1
        L = P.subgraph(list(P.neighbors((g, h))))
        lhs = comp_classes(L, reps)
        rhs = comp_classes(G.subgraph(list(G.neighbors(g))), reps) + comp_classes(H.subgraph(list(H.neighbors(h))), reps)
        if lhs != rhs: bad += 1
        for q in (2, 3, 4):
            if cliques_through(P, (g, h), q) != cliques_through(G, g, q) + cliques_through(H, h, q): bad += 1
print("root-link component multisets and clique counts additive on", trials, "product roots; failures:", bad)

# 1+H: formal square root at radius one is sum_j binom(1/2,j) delta_{cone(j K1)}; unweighted l1 finite, weighted l1 infinite
from math import comb
def b(j):  # binom(1/2, j) exactly
    num = Q(1)
    for i in range(j): num *= (Q(1, 2) - i)
    return num / Q(1) / __import__("math").factorial(j)
s0 = sum(abs(b(j)) for j in range(2000)); s1 = sum(abs(b(j)) * (j + 1) for j in range(2000))
print("partial sums to 2000: unweighted", float(s0), " weighted k=1", float(s1), "(the latter grows like sqrt(J))")
