"""E^2 + 2P: face projection vs norm across radii (the E/P obstruction to the face method)."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm import *
for r in (1, 2, 3, 4, 5):
    E = T(Cut("E", *tree_with_edge(2, 2*r+1)), r)
    P = T(Cut("P", *grid_with_edge(2*r+1)), r)
    bgE = [t for t, c in E.items() if c < 0][0]; bgP = [t for t, c in P.items() if c < 0][0]
    LL = REG.id(star_product(REG.reps[bgE], REG.reps[bgE], r))
    EE = convolve(E, E, r)
    X = add(EE, scale(P, 2))
    fp = face_projection(X, r) if r >= 2 else None
    shared = len(set(EE) & set(P))
    print(f"r={r}: L*L==Z {LL==bgP}; ||T_r(E^2+2P)||_1 = {l1(X)}; face part {fp}; types shared by E^2 and P: {shared}; ||E^2||={l1(EE)} ||P||={l1(P)}")
