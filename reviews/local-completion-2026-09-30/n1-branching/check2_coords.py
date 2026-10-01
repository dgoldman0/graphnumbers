"""Check (7)-(12): coordinates on atoms, and additivity on the whole monoid."""
import random
from fractions import Fraction as F
from canon import *
from coords import *
from atoms import *
import networkx as nx

reg = Registry()

# (12) on atoms
for d in (2, 3, 4, 5):
    for r in (2, 3, 4):
        if d == 5 and r == 4:
            continue
        h, at = tree_cut(reg, d, r)
        for key, c in at.items():
            adj, dist = reg.reps[c]
            co = tree_coords(adj, dist, r, d)
            N, cs, cR = co[0], co[1:-1], co[-1]
            assert N == 1
            if key == 'R':
                assert all(x == 0 for x in cs) and cR == 1
            else:
                assert all(cs[i] == (1 if i == key else 0) for i in range(r)) and cR == 0, (d, r, key, co)
        print(f"(12) ok d={d} r={r}")

# additivity of Q-formula (7), N_d and c's on arbitrary balls
random.seed(1)
pool_graphs = [nx.petersen_graph(), nx.complete_graph(4), nx.cycle_graph(5), nx.wheel_graph(6),
               nx.complete_bipartite_graph(2, 3), nx.grid_2d_graph(3, 3), nx.cubical_graph(),
               nx.octahedral_graph(), nx.house_x_graph(), nx.bull_graph()]
for _ in range(8):
    g = nx.gnp_random_graph(9, 0.35, seed=random.randrange(10**6))
    if nx.is_connected(g):
        pool_graphs.append(g)


def to_adj(g):
    g = nx.convert_node_labels_to_integers(g)
    return {v: set(g[v]) for v in g}


for r in (2, 3):
    balls = []
    for g in pool_graphs:
        a = to_adj(g)
        for v in list(a)[:3]:
            balls.append(reg.ball_cid(a, v, r))
    balls = list(dict.fromkeys(balls))
    checked = 0
    for i in range(len(balls)):
        for j in range(i, len(balls)):
            c1, c2 = balls[i], balls[j]
            p = reg.star(c1, c2, r)
            A1, A2, AP = reg.reps[c1], reg.reps[c2], reg.reps[p]
            d1, d2 = len(A1[0][0]), len(A2[0][0])
            assert len(AP[0][0]) == d1 + d2
            assert root_squares(AP[0]) == root_squares(A1[0]) + root_squares(A2[0]) + d1 * d2
            for d in (2, 3, 4):
                x, y, z = (tree_coords(*A, r, d) for A in (A1, A2, AP))
                assert all(zz == xx + yy for xx, yy, zz in zip(x, y, z)), (r, d)
            checked += 1
    print(f"additivity of Q-formula, N_d, c_j (d=2,3,4) on {checked} arbitrary ball pairs at r={r}: ok")
    # show some fractional / negative values
    vals = sorted({tree_coords(*reg.reps[c], r, 3)[0] for c in balls})
    print("  sample N_3 values on arbitrary balls:", [str(v) for v in vals[:8]])
