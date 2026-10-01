from fractions import Fraction as Q
import networkx as nx
from common import *
from graphlocal import Line, CutLineDefect, Finite, path, cycle
from graphlocal.interactions import TwoCutLineDefect, CutInteraction, LineCutDefect, connected_cut_interaction
from graphlocal.defects import SparseEdgeDifference

fails = []
def rep(name, ok, diff=None):
    if not ok:
        fails.append(name)
        print("FAIL", name, [(sorted(H.edges()), c) for H, v, c in (diff or [])][:4])

# Line: limit of C_n/n
for r in range(0, 6):
    n = 2 * r + 7
    C = nx.cycle_graph(n)
    oracle = rooted_hist(C, r, Q(1, n))
    ok, d = lib_vs_oracle(Line().local(r), oracle); rep(f"Line r={r}", ok, d)

# L*L vs torus C_n x C_n / n^2
for r in range(0, 4):
    n = 2 * r + 5
    T = nx.convert_node_labels_to_integers(nx.cartesian_product(nx.cycle_graph(n), nx.cycle_graph(n)))
    oracle = rooted_hist(T, r, Q(1, n * n), roots=[0])
    oracle = scale_hist(oracle, n * n)  # vertex-transitive
    oracle = scale_hist(oracle, Q(1, n * n))
    oracle = [[H, v, c * n * n] for H, v, c in oracle]
    ok, d = lib_vs_oracle((Line() * Line()).local(r), oracle); rep(f"L*L r={r}", ok, d)

# L^3 at r<=2
for r in range(0, 3):
    n = 2 * r + 4
    T = nx.convert_node_labels_to_integers(nx.cartesian_product(nx.cartesian_product(nx.cycle_graph(n), nx.cycle_graph(n)), nx.cycle_graph(n)))
    oracle = rooted_hist(T, r, Q(1), roots=[0])
    ok, d = lib_vs_oracle((Line() ** 3).local(r), oracle); rep(f"L^3 r={r}", ok, d)

# E = lim P_n - C_n
for r in range(0, 6):
    n = 2 * r + 6
    oracle = merge_hists(rooted_hist(nx.path_graph(n), r), rooted_hist(nx.cycle_graph(n), r, Q(-1)))
    ok, d = lib_vs_oracle(CutLineDefect().local(r), oracle); rep(f"E r={r}", ok, d)

# E*L vs (P_n - C_n) x C_N / N
for r in range(0, 4):
    n, N = 2 * r + 5, 2 * r + 4
    PN = nx.convert_node_labels_to_integers(nx.cartesian_product(nx.path_graph(n), nx.cycle_graph(N)))
    CN = nx.convert_node_labels_to_integers(nx.cartesian_product(nx.cycle_graph(n), nx.cycle_graph(N)))
    oracle = merge_hists(rooted_hist(PN, r, Q(1, N)), rooted_hist(CN, r, Q(-1, N)))
    ok, d = lib_vs_oracle((CutLineDefect() * Line()).local(r), oracle); rep(f"E*L r={r}", ok, d)

# E*E (crossing cuts) vs (P_n - C_n) x (P_n - C_n)
for r in range(0, 4):
    n = 2 * r + 5
    terms = []
    for A, sa in [(nx.path_graph(n), 1), (nx.cycle_graph(n), -1)]:
        for B, sb in [(nx.path_graph(n), 1), (nx.cycle_graph(n), -1)]:
            G = nx.convert_node_labels_to_integers(nx.cartesian_product(A, B))
            terms.append(rooted_hist(G, r, Q(sa * sb)))
    oracle = merge_hists(*terms)
    ok, d = lib_vs_oracle((CutLineDefect() * CutLineDefect()).local(r), oracle); rep(f"E*E r={r}", ok, d)

# Two cut defect: C_n with two deleted edges enclosing ell vertices, minus C_n
def cut_cycle(n, cuts):
    G = nx.cycle_graph(n)
    for p in cuts:  # cut edge (p-1, p) mod n
        G.remove_edge((p - 1) % n, p % n)
    return G
for ell in range(1, 7):
    for r in range(0, 6):
        n = ell + 2 * r + 6
        oracle = merge_hists(rooted_hist(cut_cycle(n, [0, ell]), r), rooted_hist(nx.cycle_graph(n), r, Q(-1)))
        ok, d = lib_vs_oracle(TwoCutLineDefect(ell).local(r), oracle); rep(f"TwoCut ell={ell} r={r}", ok, d)
        # interaction = two-cut - cut0 - cut_ell
        oracle_i = merge_hists(oracle, rooted_hist(cut_cycle(n, [0]), r, Q(-1)), rooted_hist(nx.cycle_graph(n), r, Q(1)),
                               rooted_hist(cut_cycle(n, [ell]), r, Q(-1)), rooted_hist(nx.cycle_graph(n), r, Q(1)))
        ok, d = lib_vs_oracle(CutInteraction(ell).local(r), oracle_i); rep(f"I ell={ell} r={r}", ok, d)

# LineCutDefect for several position sets
import itertools
for pos in [(0,), (0, 1), (0, 2, 7), (-3, 0, 1, 5), (2, 3, 4), (0, 10)]:
    for r in range(0, 5):
        span = max(pos) - min(pos)
        n = span + 2 * r + 8
        oracle = merge_hists(rooted_hist(cut_cycle(n, [p - min(pos) for p in pos]), r), rooted_hist(nx.cycle_graph(n), r, Q(-1)))
        ok, d = lib_vs_oracle(LineCutDefect(pos).local(r), oracle); rep(f"LineCut {pos} r={r}", ok, d)
        # full connected inclusion-exclusion
        k = len(pos)
        terms = []
        for size in range(1, k + 1):
            for S in itertools.combinations(pos, size):
                sign = (-1) ** (k - size)
                terms.append(rooted_hist(cut_cycle(n, [p - min(pos) for p in S]), r, Q(sign)))
                terms.append(rooted_hist(nx.cycle_graph(n), r, Q(-sign)))
        oracle_c = merge_hists(*terms)
        ok, d = lib_vs_oracle(connected_cut_interaction(pos).local(r), oracle_c); rep(f"connected {pos} r={r}", ok, d)

# Finite and products: K2/2 * K2/2 == C4/4 ; P3 x K3, etc.
import random
random.seed(5)
for trial in range(60):
    G1 = nx.gnp_random_graph(random.randint(1, 5), 0.6, seed=random.randint(0, 999))
    G2 = nx.gnp_random_graph(random.randint(1, 4), 0.6, seed=random.randint(0, 999))
    r = random.randint(0, 3)
    X, Y = Finite.from_graph(from_nx(G1)), Finite.from_graph(from_nx(G2))
    P = nx.convert_node_labels_to_integers(nx.cartesian_product(G1, G2))
    oracle = rooted_hist(P, r)
    ok, d = lib_vs_oracle((X * Y).local(r), oracle); rep(f"product trial {trial} r={r}", ok, d)
    ok, d = lib_vs_oracle((X * Y).finite().local(r), oracle); rep(f"materialized product trial {trial}", ok, d)
print("fails:", len(fails))
