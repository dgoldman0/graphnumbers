import random
from fractions import Fraction as Q
from math import factorial
from graphlocal import *
from graphlocal.interaction_moments import interaction_moments, tree_interaction_leading
import oracle

random.seed(7)
def random_tree(n):
    return graph(n, [(i, random.randrange(i)) for i in range(1, n)])

fails = cases = 0
for trial in range(200):
    n = random.randint(2, 9)
    g = random_tree(n)
    tree_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if g.rows[u] >> v & 1]
    k = random.randint(1, min(4, len(tree_edges)))
    sel = random.sample(tree_edges, k)
    edits = [(v, u, -1) if random.random() < .5 else (u, v, -1) for u, v in sel]
    normalize = random.random() < 0.3
    x = EdgeInteraction(g, edits, normalize=normalize, reduce_bridges=random.random() < .5)
    lead = tree_interaction_leading(x)
    order = lead.order + 1
    ref = oracle.laplacian_moments(g.rows, [(min(u,v),max(u,v),s) for u,v,s in edits], order)
    scale = Q(1, n) if normalize else Q(1)
    ref = [scale * r for r in ref]
    first = next((j for j, v in enumerate(ref) if v), None)
    cases += 1
    coef = (-1) ** first * ref[first] / factorial(first) if first is not None else None
    if first != lead.order or coef != lead.heat_coefficient:
        fails += 1
        print("FAIL", g.rows, edits, normalize, "lead", lead, "first", first, "coef", coef)
print("cases", cases, "fails", fails)
