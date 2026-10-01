import itertools
from fractions import Fraction as Fr
import sympy as sp
from rb import *

E = ((0, 0), (1, 0))
Gp = grid_adj(-12, 13, -12, 12, deleted=[E])
G = grid_adj(-12, 13, -12, 12)

# ---------- radius-two table ----------
r = 2
atoms = {'A': ball(adj_nb(Gp), (0, 0), r), 'B': ball(adj_nb(Gp), (-1, 0), r),
         'C': ball(adj_nb(Gp), (0, 1), r), 'D': ball(adj_nb(G), (0, 0), r)}
rows = []
for name, X in atoms.items():
    d = root_degree(X); Q = square_count(X); s2 = sphere_vector(X)[2]
    N = Fr(3 * d - d * d + 2 * Q, 2); L2 = Fr(s2) - Fr(d * d, 2); J = J_inv(X)
    rows.append([d, N, L2, J])
    print(name, "d=%d Q=%d s2=%d N=%s L2=%s J=%d  |V|=%d |E|=%d" % (d, Q, s2, N, L2, J, X.number_of_nodes(), X.number_of_edges()))
M = sp.Matrix([[sp.Rational(str(x)) for x in row] for row in rows])
print("det =", M.det())

# additivity of (d, N, L2, J) under star_2 on random rooted 2-balls of random graphs
import random
random.seed(1)
def inv_vec(X):
    d = root_degree(X); Q = square_count(X); s2 = sphere_vector(X)[2]
    return (d, Fr(3*d - d*d + 2*Q, 2), Fr(s2) - Fr(d*d, 2), J_inv(X))
bad = 0; tested = 0
for trial in range(300):
    g1 = nx.gnp_random_graph(random.randint(1, 8), random.random(), seed=random.randint(0, 10**6))
    g2 = nx.gnp_random_graph(random.randint(1, 8), random.random(), seed=random.randint(0, 10**6))
    u = random.choice(list(g1.nodes)); v = random.choice(list(g2.nodes))
    X = ball_of_graph(g1, u, 2); Y = ball_of_graph(g2, v, 2)
    Z = star_product_ball(X, Y, 2)
    a, b, c = inv_vec(X), inv_vec(Y), inv_vec(Z)
    tested += 1
    if tuple(x + y for x, y in zip(a, b)) != c:
        bad += 1
print("random additivity tests (d,N,L2,J) under star_2:", tested, "failures:", bad)

# face lemma random tests at r=2,3
bad = 0; tested = 0
for r_ in (2, 3):
    for trial in range(300):
        g1 = nx.gnp_random_graph(random.randint(1, 9), random.random(), seed=random.randint(0, 10**6))
        g2 = random.choice([nx.cycle_graph(random.randint(3, 7)), nx.path_graph(random.randint(1, 6)),
                            nx.gnp_random_graph(random.randint(1, 8), random.random(), seed=random.randint(0, 10**6)),
                            nx.grid_2d_graph(3, 4), nx.petersen_graph()])
        u = random.choice(list(g1.nodes)); v = random.choice(list(g2.nodes))
        X = ball_of_graph(g1, u, r_); Y = ball_of_graph(g2, v, r_)
        Z = star_product_ball(X, Y, r_)
        # also compare with the ball of the full product graph
        Zfull = ball_of_graph(nx.cartesian_product(g1, g2), (u, v), r_)
        assert rooted_iso(Z, Zfull)
        tested += 1
        if is_regular_face(Z) != (is_regular_face(X) and is_regular_face(Y)):
            bad += 1
print("random face-lemma tests (r=2,3):", tested, "failures:", bad)

# ---------- radius-three collision ----------
r = 3
L = ball(adj_nb(Gp), (-1, -1), r)
R = ball(adj_nb(Gp), (0, -2), r)
Bint = ball(adj_nb(G), (0, 0), r)
for name, X in (("(-1,-1)", L), ("(0,-2)", R), ("intact", Bint)):
    print(name, "|V|=%d |E|=%d deg=%d Q=%d sphere=%s cutline coords=%s" % (
        X.number_of_nodes(), X.number_of_edges(), root_degree(X), square_count(X), sphere_vector(X),
        tuple(str(c) for c in cutline_coords(X))))
print("(-1,-1) ~ (0,-2) rooted?", rooted_iso(L, R), "  (-1,-1)~intact?", rooted_iso(L, Bint))
# sanity: cut-line coordinates of the line atoms reproduce (10) of the cut-line note
P_ = nx.path_graph(40)
for j in range(r):
    X = ball_of_graph(P_, j, r)
    print("line atom A_{3,%d} coords" % j, tuple(str(c) for c in cutline_coords(X)))
print("line ball L_3 coords", tuple(str(c) for c in cutline_coords(ball_of_graph(P_, 20, r))))
# how many radius-3 positive planar atoms collide with the intact ball in these coordinates?
cb = cutline_coords(Bint)
for hh in range(r):
    for bb in range(r - hh):
        X = ball(adj_nb(Gp), (-hh, bb), r)
        print("  atom(h=%d,b=%d): coords=%s  same-as-intact=%s" % (hh, bb, tuple(str(c) for c in cutline_coords(X)), cutline_coords(X) == cb))
