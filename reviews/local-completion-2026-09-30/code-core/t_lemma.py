from fractions import Fraction as Q
import random
import numpy as np
import networkx as nx
from common import *
from graphlocal import ball, Finite, Line, walk_observable, LocalApproximation
from graphlocal.heat import lazy_returns
from graphlocal.local import closed_walks

random.seed(2)
def full_lazy(G, root, D, steps):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n), dtype=object)
    P = [[Q(int(A[i][j]), D) if i != j else Q(D - G.degree(i), D) for j in range(n)] for i in range(n)]
    v = [Q(int(i == root)) for i in range(n)]
    out = [Q(1)]
    for _ in range(steps):
        v = [sum(v[j] * P[j][i] for j in range(n)) for i in range(n)]
        out.append(v[root])
    return out

bad = 0; strict = 0; total = 0
for trial in range(300):
    n = random.randint(2, 14)
    G = nx.gnp_random_graph(n, random.uniform(0.15, 0.6), seed=random.randint(0, 10**9))
    if G.number_of_edges() == 0: continue
    g = from_nx(G)
    D = max(d for _, d in G.degree) + random.choice([0, 0, 1])
    root = random.randrange(n)
    R = random.randint(0, 3)
    full = full_lazy(G, root, D, 2 * R + 1)
    loc = lazy_returns(ball(g, root, R), D, 2 * R + 1)
    total += 1
    if list(loc[:2 * R + 1]) != full[:2 * R + 1]:
        bad += 1; print("FAIL lemma", trial)
    if loc[2 * R + 1] != full[2 * R + 1]:
        strict += 1
    # adjacency closed walks determined by radius floor(l/2)
    for l in range(0, 7):
        A = nx.to_numpy_array(G, nodelist=range(n), dtype=object)
        Al = np.linalg.matrix_power(A, l) if l else np.eye(n, dtype=object)
        if closed_walks(ball(g, root, l // 2), l) != int(Al[root][root]):
            bad += 1; print("FAIL walks", trial, l)
print("lemma checks:", total, "bad:", bad, "cases where order 2R+1 already differs:", strict)

# walk_observable on finite element vs trace of A^l / n
for trial in range(50):
    n = random.randint(2, 10)
    G = nx.gnp_random_graph(n, 0.5, seed=random.randint(0, 10**9))
    X = Finite.from_graph(from_nx(G), normalize=True)
    for l in range(0, 6):
        A = nx.to_numpy_array(G, nodelist=range(n), dtype=object)
        Al = np.linalg.matrix_power(A, l) if l else np.eye(n, dtype=object)
        tr = Q(int(sum(Al[i][i] for i in range(n))), n)
        iv = walk_observable(l).evaluate(X.approximate(max(l // 2, 0), max(1, l)))
        if not iv.contains(tr):
            bad += 1; print("FAIL walk observable", trial, l, iv, tr)
print("walk observable bad:", bad)
