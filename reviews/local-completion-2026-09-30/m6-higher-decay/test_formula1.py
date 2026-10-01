import random, sys
from fractions import Fraction as Q
from core import *
random.seed(12345)
nfail = 0; ntests = 0
for trial in range(300):
    n = random.randint(2, 7)
    allpairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    edges = [e for e in allpairs if random.random() < 0.5]
    k = random.randint(1, min(4, len(allpairs)))
    chosen = random.sample(allpairs, k)
    mode = random.choice(["del", "ins", "mixed"])
    edits = []
    for e in chosen:
        present = e in edges
        if mode == "del" and not present: continue
        if mode == "ins" and present: continue
        s = -1 if present else +1
        # random orientation
        if random.random() < 0.5: e = (e[1], e[0])
        edits.append((e[0], e[1], s))
    if not edits: continue
    order = 7 if len(edits) <= 3 else 6
    bf = brute_moments(n, edges, edits, order)
    tf = cyclic_formula_transfer(n, edges, edits, order)
    ntests += 1
    if [Q(x) for x in bf] != tf:
        nfail += 1
        print("FAIL transfer", n, edges, edits, bf, tf)
    if len(edits) <= 3 and order <= 7 and trial % 5 == 0:
        lf = cyclic_formula_bruteforce(n, edges, edits, min(order, 6))
        if [Q(x) for x in bf[:7]] != lf[:7]:
            nfail += 1
            print("FAIL literal", n, edges, edits, bf, lf)
print("tests", ntests, "fail", nfail)
# weighted perturbations
nf2=0
for trial in range(60):
    n = random.randint(2, 6)
    allpairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    edges = [e for e in allpairs if random.random() < 0.5]
    k = random.randint(1, min(3, len(allpairs)))
    chosen = random.sample(allpairs, k)
    edits = [(u, v, 1) for u, v in chosen]
    w = [Q(random.randint(-5, 5), random.randint(1, 4)) for _ in edits]
    bf = weighted_brute(n, edges, edits, 6, w)
    tf = cyclic_formula_transfer(n, edges, edits, 6, weights=w)
    if bf != tf:
        nf2 += 1; print("FAIL weighted", n, edges, edits, w, bf, tf)
print("weighted fails", nf2)
