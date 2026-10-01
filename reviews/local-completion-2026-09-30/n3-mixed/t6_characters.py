"""Evaluate the joint characters (9)-(11) on computed histograms; check multiplicativity on products."""
import sys, time, random, cmath
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm import *

random.seed(3)
r = int(sys.argv[1])
names = ["B3", "B4", "P"]
degs = [3, 4]


def make(name):
    M = 2 * r + 1
    if name == "P":
        return Cut("P", *grid_with_edge(M))
    return Cut(name, *tree_with_edge(int(name[1:]), M))


H = {n: T(make(n), r) for n in names}
c = {"B3": 2 * sum(2 ** j for j in range(r)), "B4": 2 * sum(3 ** j for j in range(r)), "P": 2 * r * r}
prods = {("B3", "P"): convolve(H["B3"], H["P"], r), ("B3", "B4"): convolve(H["B3"], H["B4"], r),
         ("B4", "P"): convolve(H["B4"], H["P"], r)}
nu_cache = {}


def s_val(t, z, w):
    B = REG.reps[t]
    if not interior_regular(B, r):
        return 0
    if t not in nu_cache:
        nu_cache[t] = nu(B)
    m = nu_cache[t]
    v = 1
    for zi, d in zip(z, degs):
        e = m.get(complete_id(d), 0)
        v *= zi ** e if e else 1
    e2 = m.get(complete_id(2), 0)
    v *= w ** e2 if e2 else 1
    return v


def chi(h, z, w):
    return sum(cf * s_val(t, z, w) for t, cf in h.items())


maxerr = 0
for trial in range(200):
    z = [cmath.rect(random.random(), random.uniform(0, 6.283)) for _ in degs]
    w = cmath.rect(random.random(), random.uniform(0, 6.283))
    if trial == 0:
        z, w = [0, 0], 0
    vals = {n: chi(H[n], z, w) for n in names}
    maxerr = max(maxerr, abs(vals["B3"] + c["B3"] * z[0]), abs(vals["B4"] + c["B4"] * z[1]), abs(vals["P"] + c["P"] * w * w))
    for (a, b), h in prods.items():
        maxerr = max(maxerr, abs(chi(h, z, w) - vals[a] * vals[b]))
print(f"r={r}: max |chi(B3)+c z1|, |chi(B4)+c z2|, |chi(P)+c w^2|, |chi(xy)-chi(x)chi(y)| over 200 random (z,w): {maxerr:.3e}")
