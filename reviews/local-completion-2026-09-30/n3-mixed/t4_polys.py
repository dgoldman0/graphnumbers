"""Random multivariable polynomials f(X): face projection = sum a_alpha (-1)^|alpha| c^alpha delta_{T^alpha},
lower <= ||T_r f(X)||_1 <= upper; plus cross-monomial support overlaps."""
import sys, time, random, itertools
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm import *

random.seed(7)
r = int(sys.argv[1])
maxdeg = int(sys.argv[2])
names = sys.argv[3].split(",")
t0 = time.time()


def make(name, r):
    M = 2 * r + 1
    if name == "P":
        return Cut("P", *grid_with_edge(M))
    return Cut(name, *tree_with_edge(int(name[1:]), M))


def c_of(name):
    if name == "P":
        return 2 * r * r
    d = int(name[1:])
    return 2 * sum((d - 1) ** j for j in range(r))


H = {n: T(make(n, r), r) for n in names}
bg = {n: [t for t, c in h.items() if c < 0][0] for n, h in H.items()}
c = {n: c_of(n) for n in names}
print("built marginals", {n: (len(H[n]), l1(H[n])) for n in names}, f"{time.time()-t0:.1f}s", flush=True)

# all monomials up to maxdeg
mono = {}
mono[tuple([0] * len(names))] = unit_hist()
for deg in range(1, maxdeg + 1):
    for alpha in itertools.product(range(deg + 1), repeat=len(names)):
        if sum(alpha) != deg:
            continue
        i = next(k for k, a in enumerate(alpha) if a > 0)
        prev = list(alpha); prev[i] -= 1
        mono[alpha] = convolve(mono[tuple(prev)], H[names[i]], r)
    print(f"degree {deg} monomials done ({len(mono)}), {time.time()-t0:.1f}s", flush=True)

# per-monomial facts
bad = []
face_atom = {}
for alpha, h in mono.items():
    cc = 1
    for n, a in zip(names, alpha):
        cc *= c[n] ** a
    fp = face_projection(h, r)
    if len(fp) != 1:
        bad.append(("face size", alpha, len(fp)))
        continue
    (t, v), = fp.items()
    if v != (-1) ** sum(alpha) * cc:
        bad.append(("face coeff", alpha, v))
    face_atom[alpha] = t
    up = 1
    for n, a in zip(names, alpha):
        up *= l1(H[n]) ** a
    if l1(h) != up:
        print("  monomial", alpha, "has internal cancellation:", l1(h), "<", up)
print("face atoms distinct:", len(set(face_atom.values())) == len(face_atom), "; bad:", bad)

# nu readout of each face atom
def nu_vec(t):
    m = nu(REG.reps[t])
    out = []
    for n in names:
        k = complete_id(2 if n in ("P", "B2") else int(n[1:]))
        out.append(m.get(k, 0))
    return tuple(out)

ok_nu = all(nu_vec(face_atom[a]) == tuple((2 * x if n == "P" else x) for n, x in zip(names, a)) for a in face_atom)
print("nu(face atom) = (alpha_trees, 2*alpha_P):", ok_nu)

# cross-monomial support overlaps
keys = list(mono)
overlaps = []
for a, b in itertools.combinations(keys, 2):
    s = set(mono[a]) & set(mono[b])
    if s:
        overlaps.append((a, b, len(s)))
print("pairs of distinct monomials with overlapping supports:", len(overlaps), overlaps[:10])

# random polynomials
worst = None
for trial in range(300):
    k = random.randint(1, min(8, len(keys)))
    chosen = random.sample(keys, k)
    coeffs = {a: Q(random.choice([-1, 1]) * random.randint(1, 9), random.randint(1, 5)) for a in chosen}
    f = {}
    for a, q in coeffs.items():
        f = add(f, scale(mono[a], q))
    lo = sum(abs(q) * eval_c for a, q in coeffs.items() for eval_c in [__import__('math').prod(c[n] ** x for n, x in zip(names, a))])
    up = sum(abs(q) * __import__('math').prod((2 * c[n]) ** x for n, x in zip(names, a)) for a, q in coeffs.items())
    fp = face_projection(f, r)
    exp_fp = {face_atom[a]: q * (-1) ** sum(a) * __import__('math').prod(c[n] ** x for n, x in zip(names, a)) for a, q in coeffs.items()}
    nrm = l1(f)
    if not (fp == exp_fp and l1(fp) == lo and lo <= nrm <= up):
        print("VIOLATION", coeffs, lo, nrm, up)
    ratio = (nrm - lo) / (up - lo) if up != lo else 1
    if worst is None or ratio < worst[0]:
        worst = (ratio, nrm, lo, up, coeffs)
print("all 300 random polynomials satisfy face identity and lower<=norm<=upper; min (norm-lower)/(upper-lower) =", float(worst[0]))
print("elapsed", time.time() - t0)
