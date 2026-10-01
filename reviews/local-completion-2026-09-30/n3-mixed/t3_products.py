"""Products of buffered finite cuts: direct product-graph histograms vs convolution;
face projection, Gamma coordinates of surviving atoms, cancellation, and the E/P collision."""
import sys, time, json
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm import *

t0 = time.time()
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2
FULL = len(sys.argv) > 2 and sys.argv[2] == "full"


def make(name, r):
    M = 2 * r + 1
    if name == "P":
        return Cut("P", *grid_with_edge(M))
    d = int(name[1:])
    return Cut(name, *tree_with_edge(d, M))


def c_of(name, r):
    if name == "P":
        return 2 * r * r
    d = int(name[1:])
    return 2 * sum((d - 1) ** j for j in range(r))


r = R
cuts = {n: make(n, r) for n in ("B2", "B3", "B4", "P")}
H = {n: T(c, r) for n, c in cuts.items()}
bg = {n: [t for t, c in h.items() if c < 0][0] for n, h in H.items()}
for n in H:
    print(f"T_{r} {n}: types {len(H[n])}, norm {l1(H[n])}, c={c_of(n, r)}")

results = {}
pairs = [("B3", "B4"), ("B3", "P"), ("B2", "P"), ("B4", "P"), ("B3", "B3"), ("P", "P"), ("B2", "B2")]
for a, b in pairs:
    ca, cb = cuts[a], cuts[b]
    if FULL and r == 2 and (a, b) in (("B3", "P"), ("B2", "P")):
        ra, rb = None, None
        tag = "all roots"
    else:
        ra, rb = ca.near(r), cb.near(r)  # superset of affected roots (dist <= r-1) plus a ring
        tag = f"roots within {r} of the cut endpoints ({len(ra)}x{len(rb)})"
    hd, nz = T_product_direct(ca, cb, r, ra, rb)
    hc = convolve(H[a], H[b], r)
    same = hd == hc
    up = l1(H[a]) * l1(H[b])
    lo = c_of(a, r) * c_of(b, r)
    fp = face_projection(hd, r)
    bgprod = REG.id(star_product(REG.reps[bg[a]], REG.reps[bg[b]], r))
    fp_ok = fp == {bgprod: c_of(a, r) * c_of(b, r)}
    nu_bg = nu(REG.reps[bgprod])
    nu_readable = {f"K{UREG.reps[k].number_of_nodes()}" if nx.is_isomorphic(UREG.reps[k], nx.complete_graph(UREG.reps[k].number_of_nodes())) else f"id{k}": v for k, v in nu_bg.items()}
    pos = sum(c for c in hd.values() if c > 0)
    neg = -sum(c for c in hd.values() if c < 0)
    print(f"[{a}*{b} r={r}] {tag}: direct==conv {same}; nonzero roots {nz}; types {len(hd)}; "
          f"norm {l1(hd)}; lower c*c {lo}; upper ||.||*||.|| {up}; attains upper {l1(hd)==up}; "
          f"face projection = (c_a c_b) delta_(bg*bg): {fp_ok}; nu(bg*bg) {nu_readable}; mass {sum(hd.values())}", flush=True)
    results[(a, b)] = hd

# The E/P collision: the face projection cannot separate E^2 from P (L_r * L_r = Z_r)
LL = REG.id(star_product(REG.reps[bg["B2"]], REG.reps[bg["B2"]], r))
print("L_r * L_r == Z_r as rooted types:", LL == bg["P"])
EE = results[("B2", "B2")]
combo = add(EE, scale(H["P"], 2))
print(f"E^2 + 2P at r={r}: face projection {face_projection(combo, r)} ; norm {l1(combo)} ; types {len(combo)}")
EP = results[("B2", "P")]
E3 = convolve(EE, H["B2"], r)
combo3 = add(E3, scale(EP, 2))
print(f"E^3 + 2EP at r={r}: face projection {face_projection(combo3, r)} ; norm {l1(combo3)}")
print("elapsed", time.time() - t0)
