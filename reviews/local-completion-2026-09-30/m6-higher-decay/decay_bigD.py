import random
from decay_lib import unif_moments, geometry, bound6
random.seed(11)
viol = 0; tests = 0
for trial in range(120):
    n = random.randint(3, 7)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    edges = [e for e in pairs if random.random() < 0.45]
    k = random.randint(1, 4)
    chosen = random.sample(pairs, min(k, len(pairs)))
    edits = [(u, v, -1 if (u, v) in edges else 1) for u, v in chosen]
    D, delta, taus = geometry(n, edges, edits)
    if D == 0 or taus is None: continue
    for Dp in (D + 1, D + 4):
        d = unif_moments(n, edges, edits, Dp, 11)
        tests += 1
        for j in range(12):
            if abs(d[j]) > bound6(j, len(edits), Dp, taus):
                viol += 1; print("VIOL", edges, edits, Dp, j, d[j])
print("tests", tests, "violations", viol)
