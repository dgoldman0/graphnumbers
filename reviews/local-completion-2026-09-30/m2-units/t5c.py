from gl import *
types5 = all_rooted(5)
for r in (2,3):
    tr = [t for t in types5 if ecc(REG.reps[t]) <= r]
    pats = [p for p in all_patterns(4) if pattern_ecc(p) <= r]
    keys = defaultdict(list)
    for t in tr:
        kap = cumulants_from_moments(walks(REG.reps[t], r))[1:]
        keys[tuple(kap) + tuple(K(f, t) for f in pats)].append(t)
    for k, ts in keys.items():
        if len(ts) > 1:
            print("r=%d collision:" % r, [(REG.reps[t].n, REG.reps[t].edges()) for t in ts])
