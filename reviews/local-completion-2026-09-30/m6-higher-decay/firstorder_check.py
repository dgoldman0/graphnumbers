import random
from fractions import Fraction as Q
from itertools import product
from core import brute_moments, cross_moments
random.seed(2024)

def first_order_test(n, edges, edits, maxa):
    k = len(edits)
    c = cross_moments(n, edges, edits, maxa)
    INF = None
    w = [[None]*k for _ in range(k)]; g = [[None]*k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j: w[i][j], g[i][j] = 1, 2; continue
            s = next((a for a in range(maxa+1) if c[a][i][j] != 0), None)
            if s is not None: w[i][j], g[i][j] = 1 + s, c[s][i][j]
    # DP over words: state (first, mask, last, weight) -> sum of prod gamma with sign and 1/m
    # find nu by BFS on weight up to cap
    cap = 4*k + 2*maxa
    full = (1 << k) - 1
    # accumulate by (weight, m): sum of products over closed covering words
    acc = {}
    for first in range(k):
        states = {(1 << first, first, 0, 1): Q(1)}  # mask,last,weight,m
        while states:
            new = {}
            for (mask, last, wt, m), val in states.items():
                # close
                if mask == full and w[last][first] is not None:
                    W = wt + w[last][first]
                    if W <= cap:
                        acc[(W, m)] = acc.get((W, m), Q(0)) + val * g[last][first]
                for nx in range(k):
                    if w[last][nx] is None: continue
                    W = wt + w[last][nx]
                    if W + 1 > cap: continue
                    key = (mask | (1 << nx), nx, W, m + 1)
                    new[key] = new.get(key, Q(0)) + val * g[last][nx]
            states = new
    if not acc: return None, None, w, g
    nu = min(W for W, m in acc)
    Inu = nu * sum(Q((-1)**m, m) * v for (W, m), v in acc.items() if W == nu)
    return nu, Inu, w, g

def three_formula(w, g):
    a, b, c = w[0][1], w[1][2], w[2][0]
    cands = []
    if None not in (a, b, c): cands.append(a + b + c)
    if None not in (a, b): cands.append(2*(a+b))
    if None not in (b, c): cands.append(2*(b+c))
    if None not in (c, a): cands.append(2*(c+a))
    if not cands: return None, None
    nu = min(cands)
    val = Q(0)
    if None not in (a, b, c) and a+b+c == nu: val += -2 * g[0][1]*g[1][2]*g[2][0]
    if None not in (a, b) and 2*(a+b) == nu: val += g[0][1]**2 * g[1][2]**2
    if None not in (b, c) and 2*(b+c) == nu: val += g[1][2]**2 * g[2][0]**2
    if None not in (c, a) and 2*(c+a) == nu: val += g[2][0]**2 * g[0][1]**2
    return nu, nu * val

bad = 0; tests = 0; cancels = 0; ties = 0
for trial in range(400):
    n = random.randint(3, 8)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    edges = [e for e in pairs if random.random() < 0.5]
    k = random.randint(2, 4)
    if len(edges) < k: continue
    chosen = random.sample(edges, k)
    edits = [(u, v, -1) if random.random() < .5 else (v, u, -1) for u, v in chosen]
    nu, Inu, w, g = first_order_test(n, edges, edits, n)
    order = (nu if nu is not None else 8)
    order = min(order, 12)
    I = brute_moments(n, edges, edits, order)
    tests += 1
    if nu is None:
        if any(I): bad += 1; print("nonzero but no covering word", edges, edits)
        continue
    if nu > 12: continue
    if any(I[:nu]): bad += 1; print("LOWER BOUND FAIL", n, edges, edits, nu, I)
    if I[nu] != Inu: bad += 1; print("(4) FAIL", n, edges, edits, nu, Inu, I[nu])
    if Inu == 0: cancels += 1
    if k == 3:
        nu3, I3 = three_formula(w, g)
        if (nu3, I3) != (nu, Inu): bad += 1; print("(6)/(7) FAIL", edges, edits, nu3, I3, nu, Inu)
        a, b, c = w[0][1], w[1][2], w[2][0]
        cnt = 0
        if None not in (a,b,c) and a+b+c == nu: cnt += 1
        for p, q in [(a,b),(b,c),(c,a)]:
            if None not in (p,q) and 2*(p+q) == nu: cnt += 1
        if cnt >= 2: ties += 1
print("tests", tests, "bad", bad, "cases where (4) vanishes (onset delayed)", cancels, "three-defect tie cases", ties)
