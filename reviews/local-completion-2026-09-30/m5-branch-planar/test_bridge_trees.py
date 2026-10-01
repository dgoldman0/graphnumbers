import itertools, random, sys, time
import networkx as nx
from gcore import *

def check(G, F, radii, label=""):
    is_tree, L, _ = quotient_leaf_edges(G, F)
    assert is_tree, "selected set must be bridges"
    k, ell = len(F), len(L)
    sign = (-1) ** (k - ell)
    a = interaction(G, F, component_vector)
    b = interaction(G, L, component_vector)
    b = {kk: sign * v for kk, v in b.items()}
    if a != b:
        print("COMPONENT FAIL", label, G.edges(), F, L); return False
    for r in radii:
        a = interaction(G, F, lambda H: histogram(H, r))
        b = interaction(G, L, lambda H: histogram(H, r))
        b = {kk: sign * v for kk, v in b.items()}
        if a != b:
            print("HIST FAIL", label, r, G.edges(), F, L); return False
    return True

t0 = time.time()
ncases = 0; nproper = 0
for n in range(2, 9):
    for T in nx.nonisomorphic_trees(n):
        E = list(T.edges())
        for m in range(1, len(E) + 1):
            for F in itertools.combinations(E, m):
                _, L, _ = quotient_leaf_edges(T, list(F))
                ncases += 1
                nproper += (len(L) < len(F))
                radii = [] if n > 7 else [1, 2]
                assert check(T, list(F), radii, f"n={n}")
    print("n", n, "cases so far", ncases, "proper", nproper, "time", round(time.time()-t0,1), flush=True)
print("exhaustive trees up to 8 vertices OK:", ncases, "cut sets;", nproper, "proper reductions")
