"""Section 1-3 checks: exact variation, affected-root counts, face separation, Gamma of backgrounds."""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm import *

out = []
def check(cond, msg):
    out.append((bool(cond), msg))
    print(("OK  " if cond else "FAIL"), msg, flush=True)

t0 = time.time()
for d in (2, 3, 4, 5):
    for r in (1, 2, 3, 4):
        if d == 5 and r == 4:
            continue
        L = 2 * r + 1
        cut = Cut(f"B{d}", *tree_with_edge(d, L))
        h = T(cut, r)
        cut2 = Cut(f"B{d}", *tree_with_edge(d, L + 1))
        h2 = T(cut2, r)
        S = sum((d - 1) ** j for j in range(r))
        aff = cut.affected(r)
        # affected roots <-> distance to nearer endpoint <= r-1
        dist_near = {v: min(nx.shortest_path_length(cut.before, v, u) for u in cut.e) for v in cut.before}
        pred = {v for v, dv in dist_near.items() if dv <= r - 1}
        check(set(aff) == pred and len(aff) == 2 * S, f"B{d} r={r}: affected roots = {{dist<=r-1}}, count {len(aff)} = 2*sum = {2*S}")
        check(h == h2, f"B{d} r={r}: histogram stable under larger buffer")
        check(l1(h) == 4 * S, f"B{d} r={r}: ||T_r||_1 = {l1(h)} vs 4*sum={4*S}")
        neg = [t for t, c in h.items() if c < 0]
        check(len(neg) == 1 and h[neg[0]] == -2 * S, f"B{d} r={r}: unique negative atom with coeff -c_d(r)={-2*S}")
        if r >= 2:
            ok = all(interior_regular(REG.reps[t], r) == (c < 0) for t, c in h.items())
            check(ok, f"B{d} r={r}: face membership <=> negative (background)")
            g = nu(REG.reps[neg[0]])
            check(g == Counter({complete_id(d): 1}), f"B{d} r={r}: Gamma(background) = K_{d}")
        check(sum(h.values()) == 0, f"B{d} r={r}: vertex mass 0")

for r in (1, 2, 3, 4, 5):
    M = 2 * r + 1
    cut = Cut("P", *grid_with_edge(M))
    h = T(cut, r)
    h2 = T(Cut("P", *grid_with_edge(M + 1)), r)
    aff = cut.affected(r)
    pred = {v for v in cut.before if min(abs(v[0]), abs(v[0] - 1)) + abs(v[1]) <= r - 1}
    check(set(aff) == pred and len(aff) == 2 * r * r, f"P r={r}: affected = lens, count {len(aff)} = 2r^2")
    check(h == h2, f"P r={r}: stable under larger buffer")
    check(l1(h) == 4 * r * r, f"P r={r}: ||T_r P||_1 = {l1(h)} vs 4r^2 = {4*r*r}")
    neg = [t for t, c in h.items() if c < 0]
    check(len(neg) == 1 and h[neg[0]] == -2 * r * r, f"P r={r}: unique negative atom coeff -2r^2")
    print(f"   P r={r}: number of types {len(h)} (positive {len(h)-1}); r(r+1)/2 = {r*(r+1)//2}")
    if r >= 2:
        ok = all(interior_regular(REG.reps[t], r) == (c < 0) for t, c in h.items())
        check(ok, f"P r={r}: face membership <=> negative")
        g = nu(REG.reps[neg[0]])
        check(g == Counter({complete_id(2): 2}), f"P r={r}: Gamma(background) = 2K_2")
    check(sum(h.values()) == 0, f"P r={r}: vertex mass 0")

print("failures:", [m for ok, m in out if not ok])
print("elapsed", time.time() - t0)
