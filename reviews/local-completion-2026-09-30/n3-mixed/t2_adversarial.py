"""Adversarial tests of the face lemma (4) and the root-edge product lemma (6) on arbitrary rooted balls."""
import sys, time, random
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm import *

random.seed(20261001)


def fixtures():
    F = {}
    F["K1"] = nx.empty_graph(1)
    for n in range(2, 6):
        F[f"K{n}"] = nx.complete_graph(n)
    for n in range(3, 8):
        F[f"C{n}"] = nx.cycle_graph(n)
    for n in range(2, 6):
        F[f"P{n}"] = nx.path_graph(n)
    F["diamond"] = nx.diamond_graph()          # chorded square
    F["paw"] = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3)])
    F["K23"] = nx.complete_bipartite_graph(2, 3)
    F["K33"] = nx.complete_bipartite_graph(3, 3)
    F["K222"] = nx.complete_multipartite_graph(2, 2, 2)  # octahedron
    F["W5"] = nx.wheel_graph(6)
    F["W4"] = nx.wheel_graph(5)
    F["petersen"] = nx.petersen_graph()
    F["Q3"] = nx.hypercube_graph(3)
    F["prism3"] = nx.circular_ladder_graph(3)
    F["moebius4"] = nx.circular_ladder_graph(4)
    F["book3"] = nx.Graph([(0, 1)] + [(0, i) for i in range(2, 5)] + [(1, i) for i in range(2, 5)])
    F["K4-chord-sq"] = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 4), (4, 3)])
    F["star3"] = nx.star_graph(3)
    F["tri_lattice"] = nx.triangular_lattice_graph(2, 3)
    king = nx.grid_2d_graph(3, 3)
    for x in range(2):
        for y in range(2):
            king.add_edge((x, y), (x + 1, y + 1))
            king.add_edge((x + 1, y), (x, y + 1))
    F["king3"] = king
    F["grid33"] = nx.grid_2d_graph(3, 3)
    F["grid44"] = nx.grid_2d_graph(4, 4)
    F["3reg10"] = nx.random_regular_graph(3, 10, seed=1)
    F["4reg9"] = nx.random_regular_graph(4, 9, seed=2)
    F["tree"] = nx.random_labeled_tree(9, seed=3) if hasattr(nx, "random_labeled_tree") else nx.random_tree(9, seed=3)
    F["dodeca"] = nx.dodecahedral_graph()
    F["icosa"] = nx.icosahedral_graph()
    for i in range(12):
        n = random.randint(5, 11)
        p = random.choice([0.25, 0.4, 0.55, 0.7])
        F[f"gnp{i}"] = nx.gnp_random_graph(n, p, seed=100 + i)
    # relabel to ints
    return {k: nx.convert_node_labels_to_integers(g) for k, g in F.items()}


F = fixtures()
rooted = [(name, g, u) for name, g in F.items() for u in g.nodes]
print("rooted fixtures:", len(rooted))

stats = Counter()
fails = []
t0 = time.time()


def gamma_disjoint_ok(D, B, C):
    G1 = Gamma(D)
    G2 = nx.disjoint_union(Gamma(B), Gamma(C))
    iso = nx.is_isomorphic(G1, G2)
    cm = component_multiset(G1) == component_multiset(Gamma(B)) + component_multiset(Gamma(C))
    return iso, cm


for r, npairs in ((2, 6000), (3, 2500), (1, 1500), (4, 600)):
    for _ in range(npairs):
        (n1, g, u), (n2, h, v) = random.choice(rooted), random.choice(rooted)
        B, C = ball(g, u, r), ball(h, v, r)
        D = product_ball(g, h, (u, v), r)
        D2 = star_product(B, C, r)
        if not nx.vf2pp_is_isomorphic(D, D2, node_label="dist"):
            fails.append(("star_r not well defined", r, n1, u, n2, v))
        fB, fC, fD = interior_regular(B, r), interior_regular(C, r), interior_regular(D, r)
        stats[(r, "face", fB, fC)] += 1
        if fD != (fB and fC):
            fails.append(("face", r, n1, u, n2, v))
        iso, cm = gamma_disjoint_ok(D, B, C)
        stats[(r, "gamma_ok", iso and cm)] += 1
        if r >= 2 and not (iso and cm):
            fails.append(("gamma", r, n1, u, n2, v))
        # hypothetical 'share a 4-cycle' version: component counts additive?
        G1, Gb, Gc = Gamma_share4(D), Gamma_share4(B), Gamma_share4(C)
        add_ok = component_multiset(G1) == component_multiset(Gb) + component_multiset(Gc)
        stats[(r, "share4_additive", add_ok)] += 1
    print(f"r={r} done, elapsed {time.time()-t0:.1f}s, fails so far {len(fails)}", flush=True)

for k in sorted(stats, key=str):
    print(k, stats[k])
print("FAILS:", fails[:20], len(fails))

# exhaustive radius-2 sweep over a sub-collection containing triangles / chorded squares
sub = ["K3", "K4", "diamond", "paw", "K23", "W4", "book3", "K4-chord-sq", "C4", "C5", "king3", "tri_lattice", "K222", "P3", "star3"]
cnt = 0
bad = 0
for a in sub:
    for b in sub:
        g, h = F[a], F[b]
        for u in g:
            for v in h:
                for r in (2, 3):
                    B, C = ball(g, u, r), ball(h, v, r)
                    D = product_ball(g, h, (u, v), r)
                    iso, cm = gamma_disjoint_ok(D, B, C)
                    fo = interior_regular(D, r) == (interior_regular(B, r) and interior_regular(C, r))
                    cnt += 1
                    if not (iso and cm and fo):
                        bad += 1
print("exhaustive sweep pairs:", cnt, "bad:", bad)

# r = 1 counterexample to Gamma additivity (the note restricts to r >= 2)
B = ball(F["P3"], 1, 1); C = ball(F["P3"], 1, 1)
D = product_ball(F["P3"], F["P3"], (1, 1), 1)
print("r=1: Gamma(P3 center)^2 components:", nx.number_connected_components(Gamma(D)),
      "vs sum", nx.number_connected_components(Gamma(B)) + nx.number_connected_components(Gamma(C)))
print("elapsed", time.time() - t0)
