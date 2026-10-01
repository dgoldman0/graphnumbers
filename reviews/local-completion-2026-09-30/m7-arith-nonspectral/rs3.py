"""Radius-1 powers of T_1(X) up to n=6 and radius-2 checks via actual product graphs."""
import time, itertools
from fractions import Fraction as Q
from math import comb
import networkx as nx
from gtools import *

R, S = rook(), shrikhande()
# ---- radius 1: exact convolution powers
reg = TypeRegistry(); cache = {}
hR = histogram(R, 1, reg, Q(1, 16)); hS = histogram(S, 1, reg, Q(1, 16))
X = dict(hR)
for k, v in hS.items(): X[k] = X.get(k, 0) - v
one = {reg.id(rooted_from(nx.empty_graph(1), 0)): Q(1)}
powers = [one, X]
for n in range(2, 7):
    powers.append(convolve(powers[-1], X, 1, reg, cache))
def link_signature(B):
    o = root_of(B)
    comps = link_components(B, o)
    nK3 = sum(1 for c in comps if nx.is_isomorphic(c, nx.complete_graph(3)))
    nC6 = sum(1 for c in comps if nx.is_isomorphic(c, nx.cycle_graph(6)))
    return nK3, nC6, len(comps)
for n, P in enumerate(powers):
    ok = True
    for k, v in P.items():
        nK3, nC6, nc = link_signature(reg.reps[k])
        a, b = nK3 // 2, nC6
        ok &= (nK3 % 2 == 0 and nc == nK3 + nC6 and a + b == n and v == comb(n, a) * (-1) ** b
               and reg.reps[k].number_of_nodes() == 6 * n + 1)
    print(f"r=1 n={n}: #types={len(P)} l1={l1(P)} (2^n={2**n}); each type = cone(2aK3+bC6) with coeff C(n,a)(-1)^b: {ok}", flush=True)
for t in (Q(1, 4), Q(-1, 3), Q(9, 20)):
    N = 5
    Z = {}
    for n in range(N + 1):
        for k, v in powers[n].items():
            Z[k] = Z.get(k, 0) + (-t) ** n * v
    Y = dict(one)
    for k, v in X.items(): Y[k] = Y.get(k, 0) + t * v
    prod = convolve(Y, Z, 1, reg, cache)
    resid = {k: v for k, v in prod.items() if v != (1 if k in one else 0)}
    resid = dict(prod);
    for k in one: resid[k] -= 1
    resid = {k: v for k, v in resid.items() if v}
    exp_res = {k: -(-t) ** (N + 1) * v for k, v in powers[N + 1].items()}
    p11 = sum(reg.reps[k].number_of_nodes() * abs(v) for k, v in Z.items())
    print(f"  t={t}: (1+tX)*partial == 1 - (-t)^(N+1) X^(N+1): {resid == exp_res}; "
          f"l1={l1(Z)} vs sum (2|t|)^n = {sum((2*abs(t))**n for n in range(N+1))}; "
          f"p11={p11} vs {sum((2*abs(t))**n*(6*n+1) for n in range(N+1))}", flush=True)

# ---- radius 2: actual products of the two 16-vertex graphs
def strip(G):
    H = nx.Graph(); H.add_nodes_from(G.nodes); H.add_edges_from(G.edges); return H
prods = {(0, 0): None}
facs = {(1, 0): R, (0, 1): S, (2, 0): strip(nx.cartesian_product(R, R)),
        (1, 1): strip(nx.cartesian_product(R, S)), (0, 2): strip(nx.cartesian_product(S, S))}
reg2 = TypeRegistry()
types = {}
t0 = time.time()
for key, G in facs.items():
    hashes = set()
    for o in G.nodes:
        hashes.add(whash(ball(G, o, 2)))
    # vertex-transitivity sanity: all root 2-balls share one WL hash; confirm with explicit isomorphism for a few roots
    nodes = list(G.nodes)
    B0 = ball(G, nodes[0], 2)
    iso_ok = all(rooted_iso(B0, ball(G, nodes[i], 2)) for i in (1, len(nodes) // 3, len(nodes) - 1))
    tid = reg2.id(B0)
    types[key] = tid
    print(f"r=2 product {key}: |V|={G.number_of_nodes()} distinct WL hashes over roots={len(hashes)}; sample isos ok={iso_ok}; "
          f"ball size={B0.number_of_nodes()} type id={tid}  ({time.time()-t0:.1f}s)", flush=True)
print("distinct radius-2 types for (1,0),(0,1),(2,0),(1,1),(0,2):", len(set(types.values())) == 5)
# T_2(X)^2 = T_2(R^2) - 2 T_2(RS) + T_2(S^2): l1 = 4 if three distinct types
print("radius-2 l1 of X^2 =", 1 + 2 + 1, "(requires the three n=2 types distinct:", len({types[(2,0)], types[(1,1)], types[(0,2)]}) == 3, ")")
# cross-check star_2 convolution against the actual product ball for R*S
cache2 = {}
rid = reg2.id(ball(R, 0 if 0 in R.nodes else list(R.nodes)[0], 2)); sid = reg2.id(ball(S, (0, 0), 2))
rs = star_r(reg2.reps[rid], reg2.reps[sid], 2, reg2)
print("star_2(B_R, B_S) equals the 2-ball of R box S:", rs == types[(1, 1)], flush=True)
