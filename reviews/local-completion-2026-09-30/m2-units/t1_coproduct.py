import random, itertools, math, time
from gl import *
random.seed(1)
t0=time.time()
types5 = all_rooted(5)
print("rooted connected types <=5 vertices:", len(types5))
for r in (1,2):
    tr = [t for t in types5 if ecc(REG.reps[t]) <= r]
    pats = [p for p in all_patterns(4) if pattern_ecc(p) <= r]
    print(f"r={r}: balls {len(tr)}, patterns(<=4 edges, ecc<=r) {len(pats)}")
    pairs = list(itertools.product(tr, repeat=2))
    random.shuffle(pairs)
    pairs = pairs[:400]
    bad_hom = bad_K = 0; n_hom = 0
    for a, b in pairs:
        c = star(a, b, r)
        for f in pats:
            lhs = h(f, c)
            rhs = sum(m * h(f1, a) * h(f2, b) for (f1, f2), m in coproduct(f, 2, False).items())
            n_hom += 1
            if lhs != rhs: bad_hom += 1
            if K(f, c) != K(f, a) + K(f, b): bad_K += 1
    print(f"  pairs {len(pairs)}  hom-coproduct checks {n_hom} failures {bad_hom}; K additivity failures {bad_K}")
    # coassociativity, counit, grading
    bad=0
    for f in pats:
        lhs=Counter(); rhs=Counter()
        for (x,y),m in coproduct(f,2,False).items():
            for (x1,x2),mm in coproduct(x,2,False).items(): lhs[(x1,x2,y)]+=m*mm
            for (y1,y2),mm in coproduct(y,2,False).items(): rhs[(x,y1,y2)]+=m*mm
        if not (lhs==rhs==coproduct(f,3,False)): bad+=1
    print("  coassociativity failures", bad)
print("time", time.time()-t0)
