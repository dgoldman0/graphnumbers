import itertools, time
from gl import *
t0=time.time()
types5 = all_rooted(5)
for r in (1,2,3):
    tr = [t for t in types5 if ecc(REG.reps[t]) <= r]
    prod = {}
    bad3 = bad2 = badsub = 0
    for a,b in itertools.product(tr, repeat=2):
        c = star(a,b,r); prod[(a,b)] = c
        A,B,C = REG.reps[a],REG.reps[b],REG.reps[c]
        if C.n < A.n + B.n - 1: bad3 += 1
        if C.n > A.n*B.n: badsub += 1
        if C.deg(0) != A.deg(0)+B.deg(0): bad2 += 1
    # cancellativity: for each C, map B -> B*C must be injective on tr
    noncancel = 0
    for c in tr:
        seen = {}
        for b in tr:
            p = prod[(b,c)]
            if p in seen: noncancel += 1; print("NONCANCEL", r, seen[p], b, c)
            seen[p] = b
    # walk-cumulant additivity (open walks j<=r) and closed (j<=2r+1); size bound (12)
    badk = badm = bad12 = 0
    for (a,b),c in prod.items():
        ka = cumulants_from_moments(walks(REG.reps[a], r)); kb = cumulants_from_moments(walks(REG.reps[b], r)); kc = cumulants_from_moments(walks(REG.reps[c], r))
        if any(kc[j] != ka[j]+kb[j] for j in range(1,r+1)): badk += 1
        L = 2*r+1
        qa = cumulants_from_moments(walks(REG.reps[a], L, True)); qb = cumulants_from_moments(walks(REG.reps[b], L, True)); qc = cumulants_from_moments(walks(REG.reps[c], L, True))
        if any(qc[j] != qa[j]+qb[j] for j in range(1,L+1)): badm += 1
        Cg = REG.reps[c]
        if Cg.n > sum(walks(Cg, r)): bad12 += 1
    # sharpness: closed-walk cumulant of length 2r+2 not additive in general
    nonadd = 0
    for (a,b),c in prod.items():
        L = 2*r+2
        qa = cumulants_from_moments(walks(REG.reps[a], L, True)); qb = cumulants_from_moments(walks(REG.reps[b], L, True)); qc = cumulants_from_moments(walks(REG.reps[c], L, True))
        if qc[L] != qa[L]+qb[L]: nonadd += 1
    print(f"r={r}: balls {len(tr)}, products {len(prod)}; (3) fails {bad3}; |BD|<=|B||D| fails {badsub}; deg additivity fails {bad2}; noncancellative {noncancel}; open-walk cumulant add fails {badk}; closed j<=2r+1 fails {badm}; (12) fails {bad12}; closed j=2r+2 nonadditive {nonadd}")
print("registry size", len(REG.reps), "time", time.time()-t0)
