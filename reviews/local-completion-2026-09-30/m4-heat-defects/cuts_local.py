from collections import Counter
from fractions import Fraction as F
import itertools, networkx as nx

def rooted_type_deg2(G, o, r):
    """Exact rooted type of the induced r-ball of a max-degree<=2 graph."""
    dist = nx.single_source_shortest_path_length(G, o, cutoff=r)
    B = G.subgraph(dist.keys())
    k = B.number_of_nodes()
    if k == 1: return ('P', 1, 0)
    degs = dict(B.degree())
    ends = [v for v in B if degs[v] == 1]
    if not ends:
        return ('C', k)   # cycle; vertex-transitive
    # path: position of root from one end
    a = nx.shortest_path_length(B, ends[0], o)
    return ('P', k, min(a, k-1-a))

def T(G, r, coeff=1):
    c = Counter()
    for o in G.nodes():
        c[rooted_type_deg2(G, o, r)] += coeff
    return c

def add(*cs):
    out = Counter()
    for coef, c in cs:
        for k, v in c.items(): out[k] += coef*v
    return {k: v for k, v in out.items() if v != 0}

def size(key): return key[1]
def var(h): return sum(abs(v) for v in h.values())
def pnorm(h, k): return sum(abs(v)*size(key)**k for key, v in h.items())

def TL(r): return Counter({('P', 2*r+1, r): 1})
def TE_formula(r):
    if r == 0: return {}
    h = Counter({('P', r+j+1, j): 2 for j in range(r)})
    h[('P', 2*r+1, r)] -= 2*r
    return dict(h)
def Pn(n): return nx.path_graph(n)
def Cn(n): return nx.cycle_graph(n)

# 1. formula (6): exact threshold in n
print("== Cut line formula (6) ==")
for r in range(0, 7):
    ok_ns = [n for n in range(max(3, 2*r-1), 2*r+8) if add((1, T(Pn(n), r)), (-1, T(Cn(n), r))) == TE_formula(r)]
    print(f"r={r}: n values (in tested range) where T_r(P_n-C_n) equals (6): {ok_ns}; claimed threshold n>=2r+2={2*r+2}; var={var(TE_formula(r))} (claim 4r={4*r}); p_r,2={pnorm(TE_formula(r),2)} vs formula {2*sum((r+j+1)**2 for j in range(r))+2*r*(2*r+1)**2}")

# 2. interactions
print("== Two cuts ==")
def T_I_formula(ell, r):  # P_ell - ell L - E
    return add((1, T(Pn(ell), r)), (-ell, TL(r)), (-1, Counter(TE_formula(r))))
def T_D_formula(ell, r):
    return add((1, T(Pn(ell), r)), (-ell, TL(r)), (1, Counter(TE_formula(r))))
bad = []
for ell in range(1, 9):
    first = None
    for r in range(0, ell + 4):
        nmin = max(ell + 2*r, 2*r + 2, ell + 1, 3)
        for n in range(nmin, nmin + 3):
            G2 = nx.cycle_graph(n); G2.remove_edge(n-1, 0); G2.remove_edge(ell-1, ell)
            G1 = nx.cycle_graph(n); G1.remove_edge(n-1, 0)
            Ifin = add((1, T(G2, r)), (1, T(Cn(n), r)), (-2, T(G1, r)))
            Dfin = add((1, T(G2, r)), (-1, T(Cn(n), r)))
            if Ifin != T_I_formula(ell, r) or Dfin != T_D_formula(ell, r):
                bad.append((ell, r, n))
        hI = T_I_formula(ell, r)
        if hI and first is None: first = r
        if r <= ell // 2 and hI: bad.append(('nonzero below threshold', ell, r))
        if r >= ell:
            if var(hI) != 4*r or var(T_D_formula(ell, r)) != 4*r + 2*ell: bad.append(('var', ell, r, var(hI), var(T_D_formula(ell,r))))
    print(f"ell={ell}: first nonzero radius={first}, claimed floor(ell/2)+1={ell//2+1}; var at r=ell-1: {var(T_I_formula(ell, ell-1)) if ell>1 else None} (4r={4*(ell-1)})")
print("two-cut mismatches:", bad)
# check whether the stated sufficient n is also needed (one below)
below = []
for ell in range(1, 7):
    for r in range(1, 6):
        nmin = max(ell + 2*r, 2*r + 2)
        n = nmin - 1
        if n <= ell or n < 3: continue
        G2 = nx.cycle_graph(n); G2.remove_edge(n-1, 0); G2.remove_edge(ell-1, ell)
        G1 = nx.cycle_graph(n); G1.remove_edge(n-1, 0)
        Ifin = add((1, T(G2, r)), (1, T(Cn(n), r)), (-2, T(G1, r)))
        below.append(Ifin == T_I_formula(ell, r))
print("at n = threshold-1, identity holds in", sum(below), "of", len(below), "cases")

# 3. inclusion-exclusion (13)
print("== Higher inclusion-exclusion (13) ==")
def cut_cycle(n, cutpos):  # cut edges {x-1,x} (x>0) and wrap edge for x=0
    G = nx.cycle_graph(n)
    for x in cutpos:
        G.remove_edge((x-1) % n, x)
    return G
results = []
for positions in [(0,1,2), (0,1,3), (0,2,7), (0,3,4,9), (0,1,2,3), (0,2,5,6,11), (0,1,4,6,7), (-3,0,4,5)]:
    pos = sorted(positions); base = pos[0]; pos = [p-base for p in pos]
    k = len(pos); span = pos[-1]
    for r in range(0, 6):
        n = span + 2*r + 2
        total = Counter()
        for s in range(1, k+1):
            for S in itertools.combinations(pos, s):
                sign = (-1)**(k - s)
                Dfin = add((1, T(cut_cycle(n, S), r)), (-1, T(Cn(n), r)))
                for key, v in Dfin.items(): total[key] += sign*v
        total = {kk: v for kk, v in total.items() if v}
        target = {kk: (-1)**k * v for kk, v in T_I_formula(span, r).items()}
        results.append((tuple(positions), r, total == target))
print("all (13) checks pass:", all(x[2] for x in results), "count", len(results), [x for x in results if not x[2]][:5])
