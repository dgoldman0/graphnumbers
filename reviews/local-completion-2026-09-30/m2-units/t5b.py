from gl import *
types5 = all_rooted(5)
for r in (1,2,3):
    tr = [t for t in types5 if ecc(REG.reps[t]) <= r]
    pats = [canon_multi(REG.reps[t].n, REG.reps[t].edges()) for t in tr if REG.reps[t].n>1]
    keys = {t: tuple(h(p, t) for p in pats) for t in tr}
    print(r, len(tr), "distinct rooted-hom vectors using the balls themselves as patterns:", len(set(keys.values())))
