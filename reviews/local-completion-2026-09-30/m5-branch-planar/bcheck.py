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
        print("COMPONENT FAIL", label, list(G.edges()), F, L); return False
    for r in radii:
        a = interaction(G, F, lambda H: histogram(H, r))
        b = interaction(G, L, lambda H: histogram(H, r))
        b = {kk: sign * v for kk, v in b.items()}
        if a != b:
            print("HIST FAIL", label, r, list(G.edges()), F, L); return False
    return True
