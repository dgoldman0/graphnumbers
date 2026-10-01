import networkx as nx, indep
from fractions import Fraction as Fr
reg = indep.Registry()
c = [Fr(1), Fr(-2), Fr(3, 2), Fr(5)]
def J_terms(c):
    t = []
    for j, cj in enumerate(c, 1):
        N = 3 ** j
        t += [(cj / (2 * N), nx.cycle_graph(2 * N)), (-cj / N, nx.cycle_graph(N))]
    return t
for m in (1, 2):
    r = 3 ** m
    h = indep.lin_hist(J_terms(c), r, reg)   # includes tail terms j > m (should cancel)
    tv = sum(abs(v) for v in h.values())
    cost = sum(abs(a) * G.number_of_nodes() for a, G in J_terms(c))
    print("m", m, "radius", r, "TV", tv, "expected", 2 * sum(abs(x) for x in c[:m]), "full-representation cost", cost)
