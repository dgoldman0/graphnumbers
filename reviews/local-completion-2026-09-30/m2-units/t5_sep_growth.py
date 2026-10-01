import itertools, time
from gl import *
t0 = time.time()
types5 = all_rooted(5)
for r in (1, 2, 3):
    tr = [t for t in types5 if ecc(REG.reps[t]) <= r]
    pats = [p for p in all_patterns(4) if pattern_ecc(p) <= r]
    keys = {}
    for t in tr:
        kap = cumulants_from_moments(walks(REG.reps[t], r))[1:]
        keys[t] = tuple(kap) + tuple(K(f, t) for f in pats)
    print(f"r={r}: {len(tr)} ball types, {len(pats)} patterns(<=4 edges): distinct keys {len(set(keys.values()))}")
    # K values integral?

# growth lemma (4)
worst = 0
for t in [x for x in types5 if REG.reps[x].n >= 2][:40]:
    B = REG.reps[t]; Delta = max(len(a) for a in B.adj)
    P = G(1, [])
    for n in range(1, 4):
        P = cart(P, B)
        for r in range(1, 4):
            sz = ball(P, 0, r).n
            bound = sum((n * Delta) ** j for j in range(r + 1))
            assert sz <= bound <= (r + 1) * (1 + n * Delta) ** r, (t, n, r, sz, bound)
print("growth bound (4) holds on 40 balls, n<=3, r<=3; time", round(time.time() - t0, 1))
