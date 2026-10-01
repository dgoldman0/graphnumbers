from fractions import Fraction as Q
from math import factorial
from itertools import product
from core import *

def first_nonzero(I):
    for j, x in enumerate(I):
        if x: return j
    return None

def heat_lead(I):
    j = first_nonzero(I)
    return j, Q((-1)**j * I[j], factorial(j)) if j is not None else None

def by_letters(n, edges, edits, N):
    k = len(edits); c = cross_moments(n, edges, edits, N); sig=[s for *_ ,s in edits]
    out = {}
    for m in range(k, N+1):
        sub = 0
        for labels in product(range(k), repeat=m):
            if len(set(labels)) != k: continue
            sg = 1
            for l in labels: sg *= sig[l]
            for gaps in compositions(N-m, m):
                p = 1
                for j in range(m):
                    p *= c[gaps[j]][labels[j]][labels[(j+1)%m]]
                    if not p: break
                sub += sg*p
        out[m] = Q(N, m)*sub
    return out

def s_gamma(n, edges, edits, maxa):
    c = cross_moments(n, edges, edits, maxa)
    k = len(edits)
    S = {}
    for i in range(k):
        for j in range(k):
            if i == j: continue
            s = next((a for a in range(maxa+1) if c[a][i][j] != 0), None)
            S[(i+1, j+1)] = (s, c[s][i][j] if s is not None else None)
    return S

# 1. K4 with path edges (0,1),(1,2),(2,3)
K4 = [(i,j) for i in range(4) for j in range(i+1,4)]
ed = [(0,1,-1),(1,2,-1),(2,3,-1)]
I = brute_moments(4, K4, ed, 8)
print("K4 path:", I, heat_lead(I), s_gamma(4, K4, ed, 4))
# 2. paw fixture used by the repo verifier
paw = [(0,1),(0,2),(0,3),(1,2)]
ed2 = [(0,1,-1),(0,3,-1),(1,2,-1)]
I = brute_moments(4, paw, ed2, 8)
print("paw fixture:", I, heat_lead(I), s_gamma(4, paw, ed2, 4))
print("  paw by letters at 4:", by_letters(4, paw, ed2, 4))
print("  K4 by letters at 4:", by_letters(4, K4, ed, 4))
# 3. cancellation example
G6 = [(3,5),(2,4),(1,5),(0,5),(1,2),(1,3),(3,4),(4,5)]
ed3 = [(3,5,-1),(2,4,-1),(1,5,-1)]
I = brute_moments(6, G6, ed3, 9)
print("cancel triple:", I, heat_lead(I))
print("  s,gamma:", s_gamma(6, G6, ed3, 6))
print("  by letters at 6:", by_letters(6, G6, ed3, 6))
# 4. four-defect cancellation
G5 = [(0,1),(0,2),(0,4),(1,2),(1,3)]
ed4 = [(0,1,-1),(0,2,-1),(0,4,-1),(1,3,-1)]
I = brute_moments(5, G5, ed4, 9)
print("cancel quad:", I, heat_lead(I))
print("  s,gamma:", s_gamma(5, G5, ed4, 6))
print("  by letters at 5:", by_letters(5, G5, ed4, 5))
print("  by letters at 6:", by_letters(5, G5, ed4, 6))
# 5. binary quartet
BQ = [(0,1),(0,2),(0,3),(1,4),(1,5)]
ed5 = [(0,2,-1),(0,3,-1),(1,4,-1),(1,5,-1)]
I = brute_moments(6, BQ, ed5, 8); print("binary quartet:", I, heat_lead(I))
# spider (2,1,1,1): center 0, arm 0-1-2, arms 0-3, 0-4, 0-5; terminal edges (1,2),(0,3),(0,4),(0,5)
SP = [(0,1),(1,2),(0,3),(0,4),(0,5)]
ed6 = [(1,2,-1),(0,3,-1),(0,4,-1),(0,5,-1)]
I = brute_moments(6, SP, ed6, 8); print("spider 2111:", I, heat_lead(I))
# spider (2,2,2): 0 center; 0-1-2, 0-3-4, 0-5-6 terminal (1,2),(3,4),(5,6)
SP3 = [(0,1),(1,2),(0,3),(3,4),(0,5),(5,6)]
ed7 = [(1,2,-1),(3,4,-1),(5,6,-1)]
I = brute_moments(7, SP3, ed7, 10); print("spider 222:", I, heat_lead(I), Q(1,20160))
SP4 = [(0,1),(1,2),(0,3),(3,4),(0,5),(5,6),(0,7),(7,8)]
ed8 = [(1,2,-1),(3,4,-1),(5,6,-1),(7,8,-1)]
I = brute_moments(9, SP4, ed8, 12); print("spider 2222:", I, heat_lead(I), Q(1,6652800))
# 6. time sign reversal fixture
TS = [(0,1),(0,2),(0,3),(0,4),(1,2)]
ed9 = [(0,1,-1),(0,3,-1),(0,4,-1)]
I = brute_moments(5, TS, ed9, 9); print("sign reversal:", I, heat_lead(I))
