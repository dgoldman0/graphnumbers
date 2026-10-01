from fractions import Fraction as Q
import random, itertools
import numpy as np
from scipy.optimize import linprog
import networkx as nx
from common import *
from graphlocal import reconstruct, catalog, Finite, Line, LocalHistogram, path, cycle, OutOfSpan, BudgetExceeded, CutLineDefect

# catalog counts vs OEIS A001349 (connected graphs): 1,1,2,6,21,112
cat6 = catalog(6)
print("catalog counts:", [sum(g.n == n for g in cat6) for n in range(1, 7)])
# max_degree filter vs networkx atlas
atlas = [G for G in nx.graph_atlas_g() if 1 <= G.number_of_nodes() <= 6 and nx.is_connected(G)]
for D in range(0, 5):
    mine = [g for g in catalog(6, D)]
    want = [G for G in atlas if max((d for _, d in G.degree), default=0) <= D]
    if len(mine) != len(want):
        print("catalog max_degree mismatch", D, len(mine), len(want))
print("catalog degree filter checked")

def oracle_lp(target, graphs):
    r = target.radius
    cols = [g.n for g in graphs]
    hists = [LocalHistogram.from_graph(g, r) for g in graphs]
    # Build row keys via networkx to stay independent of library merge
    keys = []
    def key_index(H):
        for i, K in enumerate(keys):
            if K.number_of_nodes() == H.number_of_nodes() and K.number_of_edges() == H.number_of_edges() and nx.is_isomorphic(K, H, node_match=lambda a, b: a["r"] == b["r"]):
                return i
        keys.append(H); return len(keys) - 1
    def nxkey(g):
        H = to_nx(g); nx.set_node_attributes(H, {u: int(u == 0) for u in H}, "r"); return H
    # raw (unmerged) balls via networkx
    colvecs = []
    for g in graphs:
        G = to_nx(g); v = {}
        for root in G.nodes:
            H = nx_ball(G, root, r); nx.set_node_attributes(H, {u: int(u == root) for u in H}, "r")
            i = key_index(H); v[i] = v.get(i, 0) + 1
        colvecs.append(v)
    b = {}
    for key, c in target.values.items():
        i = key_index(nxkey(key.graph)); b[i] = b.get(i, 0) + c
    m, n = len(keys), len(graphs)
    A = np.zeros((m, n)); bb = np.zeros(m)
    for j, v in enumerate(colvecs):
        for i, c in v.items(): A[i, j] = c
    for i, c in b.items(): bb[i] = float(c)
    # min sum n_j (p_j + q_j), A(p - q) = b
    res = linprog(np.array(cols + cols, float), A_eq=np.hstack([A, -A]), b_eq=bb, bounds=[(0, None)] * (2 * n), method="highs")
    return res, A, bb

random.seed(7)
bad = 0; feas = infeas = 0
graphs_pool = list(catalog(5))
for trial in range(250):
    r = random.randint(0, 2)
    cat = random.sample(graphs_pool, random.randint(1, 8))
    # target: random combination of random graphs (sometimes outside catalog)
    srcs = random.sample(graphs_pool, random.randint(1, 3))
    target = Finite([(Q(random.randint(-4, 4), random.randint(1, 3)), g) for g in srcs]).local(r)
    if random.random() < 0.3:
        target = Line().local(r) if random.random() < 0.5 else CutLineDefect().local(r)
    res, A, bb = oracle_lp(target, cat)
    try:
        rec = reconstruct(target, cat)
    except OutOfSpan as e:
        infeas += 1
        if res.status == 0:
            print("FAIL: library says infeasible, LP feasible", trial); bad += 1
        # verify witness: annihilates all catalog columns, nonzero on target
        w = e.witness
        dot = lambda h: sum((c * w.values.get(k, Q(0)) for k, c in h.values.items()), Q(0))
        if any(dot(LocalHistogram.from_graph(g, r)) != 0 for g in cat) or dot(target) == 0 or dot(target) != e.value:
            print("FAIL witness", trial); bad += 1
        continue
    except BudgetExceeded:
        print("budget", trial); continue
    feas += 1
    if res.status != 0:
        print("FAIL: library feasible, LP infeasible", trial, res.message); bad += 1; continue
    if abs(float(rec.cost) - res.fun) > 1e-6 * max(1, abs(res.fun)):
        print("FAIL cost", trial, rec.cost, res.fun); bad += 1
    if rec.element.local(r) != target:
        print("FAIL representative", trial); bad += 1
print("reconstruct: bad", bad, "feasible", feas, "infeasible", infeas)
