"""Exact radius-1 convolution powers of T_1(X) using a complete invariant, and
radius-2 checks on actual product graphs.

Radius-1 ball = cone over the root link, so its rooted type is determined by the
isomorphism class of the link, i.e. by the multiset of iso classes of link components.
Component classes are resolved by explicit isomorphism against a registry (small graphs).
Convolution B *_1 C is computed by literally forming the product of the two balls and
taking the radius-1 ball, then canonicalizing -- no use of the link-additivity claim.
"""
from fractions import Fraction as Q
from math import comb
from collections import Counter
import networkx as nx
from gtools import ball, rooted_from, root_of, link_components, rook, shrikhande, whash

comp_reps = []
def comp_class(C):
    for i, D in enumerate(comp_reps):
        if C.number_of_nodes() == D.number_of_nodes() and C.number_of_edges() == D.number_of_edges() and nx.is_isomorphic(C, D):
            return i
    comp_reps.append(C); return len(comp_reps) - 1

def r1_type(B):
    o = root_of(B)
    return tuple(sorted(Counter(comp_class(c) for c in link_components(B, o)).items()))

type_reps = {}
def canon(B):
    k = r1_type(B)
    type_reps.setdefault(k, B)
    return k

def star1(k1, k2):
    B, C = type_reps[k1], type_reps[k2]
    P = nx.cartesian_product(B, C)
    P2 = nx.Graph(); P2.add_nodes_from(P.nodes); P2.add_edges_from(P.edges)
    return canon(ball(P2, (root_of(B), root_of(C)), 1))

cache = {}
def conv(a, b):
    out = Counter()
    for x, cx in a.items():
        for y, cy in b.items():
            key = tuple(sorted((x, y)))
            if key not in cache: cache[key] = star1(*key)
            out[cache[key]] += cx * cy
    return {k: v for k, v in out.items() if v}

R, S = rook(), shrikhande()
def hist1(G, scale):
    h = Counter()
    for o in G.nodes: h[canon(ball(G, o, 1))] += scale
    return dict(h)
hR, hS = hist1(R, Q(1, 16)), hist1(S, Q(1, 16))
X = Counter(hR); X.subtract(hS); X = {k: v for k, v in X.items() if v}
one = {canon(rooted_from(nx.empty_graph(1), 0)): Q(1)}
K3, C6 = comp_class(nx.complete_graph(3)), comp_class(nx.cycle_graph(6))
powers = [one, X]
NMAX = 7
for n in range(2, NMAX + 1):
    powers.append(conv(powers[-1], X))
l1 = lambda a: sum(abs(v) for v in a.values())
size = lambda k: type_reps[k].number_of_nodes()
for n, P in enumerate(powers):
    ok = True
    for k, v in P.items():
        d = dict(k); nK3, nC6 = d.get(K3, 0), d.get(C6, 0)
        a, b = nK3 // 2, nC6
        ok &= (set(d) <= {K3, C6} and nK3 % 2 == 0 and a + b == n and v == comb(n, a) * (-1) ** b and size(k) == 6 * n + 1)
    print(f"r=1 n={n}: #types={len(P)} l1={l1(P)} (2^n={2**n}); all types cone(2a K3 + b C6), coeff C(n,a)(-1)^b, size 6n+1: {ok}", flush=True)
for t in (Q(1, 4), Q(-1, 3), Q(9, 20), Q(-49, 100)):
    N = NMAX - 1
    Z = Counter()
    for n in range(N + 1):
        for k, v in powers[n].items(): Z[k] += (-t) ** n * v
    Z = {k: v for k, v in Z.items() if v}
    Y = Counter(one)
    for k, v in X.items(): Y[k] += t * v
    prod = Counter(conv(dict(Y), Z))
    for k in one: prod[k] -= 1
    resid = {k: v for k, v in prod.items() if v}
    exp_res = {k: -(-t) ** (N + 1) * v for k, v in powers[N + 1].items()}
    p11 = sum(size(k) * abs(v) for k, v in Z.items())
    p12 = sum(size(k) ** 2 * abs(v) for k, v in Z.items())
    print(f"  t={t}: residual == -(-t)^(N+1) X^(N+1): {resid == exp_res}; l1(partial)={l1(Z)} == sum_(n<=N)(2|t|)^n: "
          f"{l1(Z) == sum((2*abs(t))**n for n in range(N+1))}; p11 matches sum (2|t|)^n (6n+1): "
          f"{p11 == sum((2*abs(t))**n*(6*n+1) for n in range(N+1))}; p12 matches: {p12 == sum((2*abs(t))**n*(6*n+1)**2 for n in range(N+1))}", flush=True)
# t = 1/2 and -1/2 obstructions: link characters z^{#K3} w^{#C6}-type evaluation
def char(a, z, w):
    tot = 0
    for k, v in a.items():
        d = dict(k); tot += v * (z ** d.get(K3, 0)) * (w ** d.get(C6, 0))
    return tot
for t, z, w in ((Q(1, 2), 1j, 1), (Q(-1, 2), 1, -1)):
    Y = Counter(one)
    for k, v in X.items(): Y[k] += t * v
    # z counts K3 components (R = Q^2 has 2 K3 per root -> z^2), w counts C6 components
    print(f"  character (z per K3 comp={z}, w per C6 comp={w}) of 1+tX at t={t}:", char(dict(Y), z, w))

# radius 2: actual products, WL-hash sanity (vertex transitivity is guaranteed: Cayley graphs of Z_4^4)
def strip(G):
    H = nx.Graph(); H.add_nodes_from(G.nodes); H.add_edges_from(G.edges); return H
facs = {(1, 0): R, (0, 1): S, (2, 0): strip(nx.cartesian_product(R, R)),
        (1, 1): strip(nx.cartesian_product(R, S)), (0, 2): strip(nx.cartesian_product(S, S))}
seen = {}
for key, G in facs.items():
    hs = {whash(ball(G, o, 2)) for o in G.nodes}
    B0 = ball(G, next(iter(G.nodes)), 2)
    seen[key] = (hs, B0.number_of_nodes(), B0.number_of_edges(), r1_type(ball(B0, root_of(B0), 1)))
    print(f"r=2 {key}: |V|={G.number_of_nodes()}, #distinct WL hashes of root 2-balls={len(hs)}, 2-ball |V|,|E|={B0.number_of_nodes()},{B0.number_of_edges()}, radius-1 truncation={seen[key][3]}", flush=True)
r1s = [v[3] for v in seen.values()]
print("radius-1 truncations of the five product 2-balls pairwise distinct:", len(set(r1s)) == 5)
