"""Exact finite checks for APPLICATION_ASSESSMENT.md.

These fixtures test the mathematical observations recorded in the
assessment: edge-space duality for bond cuts, the alternating sign law on
graphs of maximum degree at most two and its failure at branch points, the
sign-definite adjacency interaction, the fixed sign of the two-bond
spanning-tree interaction, a prime circulant whose small balls are square
lattice balls, and a non-unique factorization of a positive mixture.
The literature verdicts in the note are not checked here.
No graphlocal or third-party modules are imported.
"""
from fractions import Fraction as Q
from itertools import combinations, product
from math import factorial
from pathlib import Path
import argparse
import hashlib
import json


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def edge_list(edges):
    out, seen = [], set()
    for u, v in edges:
        require(u != v, "loops are not simple edges")
        e = (min(u, v), max(u, v))
        require(e not in seen, "duplicate simple edge")
        seen.add(e)
        out.append(e)
    return out


def adjacency(n, edges):
    a = [[0]*n for _ in range(n)]
    for u, v in edges:
        a[u][v] = a[v][u] = 1
    return a


def laplacian(n, edges):
    m = [[0]*n for _ in range(n)]
    for u, v in edges:
        m[u][u] += 1
        m[v][v] += 1
        m[u][v] -= 1
        m[v][u] -= 1
    return m


def edge_gram(edges, orientation=None):
    """K = B^T B, where column e of B is +1 at its tail and -1 at its head."""
    orientation = orientation or [0]*len(edges)
    signs = []
    for (u, v), flip in zip(edges, orientation):
        tail, head = (v, u) if flip else (u, v)
        signs.append({tail: 1, head: -1})
    return [[sum(a[w]*b[w] for w in a if w in b) for b in signs] for a in signs]


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def power_traces(m, top):
    n = len(m)
    if n == 0:
        return [0]*(top+1)
    p = [[int(i == j) for j in range(n)] for i in range(n)]
    out = []
    for _ in range(top+1):
        out.append(sum(p[i][i] for i in range(n)))
        p = matmul(p, m)
    return out


def principal(m, keep):
    return [[m[i][j] for j in keep] for i in keep]


def subsets(items):
    for size in range(len(items)+1):
        yield from combinations(items, size)


# Fixtures. HI is the HIGHER_INTERACTION_GEOMETRY section 6 graph:
# the star K_{1,4} plus the edge 12, with cuts 01, 03 and 04.
PATH7 = [(i, i+1) for i in range(6)]
CYCLE8 = [(i, (i+1) % 8) for i in range(8)]
STAR3 = [(0, 1), (0, 2), (0, 3)]
SPIDER = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6)]
HI = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)]
G7 = [(0, 1), (0, 3), (0, 4), (0, 6), (1, 4), (1, 5), (2, 4), (2, 5), (2, 6),
      (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]
K4 = list(combinations(range(4), 2))
K5 = list(combinations(range(5), 2))
WHEEL6 = [(0, i) for i in range(1, 6)] + [(i, i % 5+1) for i in range(1, 6)]
PETERSEN = ([(i, (i+1) % 5) for i in range(5)] + [(i, i+5) for i in range(5)]
            + [(5+i, 5+(i+2) % 5) for i in range(5)])


def edge_space_duality():
    cases = [("path P7", 7, PATH7, [(1, 2), (4, 5), (5, 6)]),
             ("cycle C8", 8, CYCLE8, [(0, 1), (3, 4), (5, 6)]),
             ("star K13", 4, STAR3, STAR3),
             ("spider 222", 7, SPIDER, [(0, 1), (0, 3), (0, 5)]),
             ("HI fixture", 5, HI, [(0, 1), (0, 3), (0, 4)]),
             ("G7", 7, G7, [(0, 4), (2, 4), (3, 5)])]
    records = []
    for name, n, edges, cuts in cases:
        edges, cuts = edge_list(edges), edge_list(cuts)
        k_full = edge_gram(edges)
        for cut in subsets(cuts):
            rest = [e for e in edges if e not in cut]
            keep = [i for i, e in enumerate(edges) if e not in cut]
            tl = power_traces(laplacian(n, rest), 10)
            tk = power_traces(principal(k_full, keep), 10)
            require(tl[0]-tk[0] == n-len(rest), name+": zeroth trace offset")
            require(tl[1:] == tk[1:], name+": nonzero spectra differ")
        records.append({"fixture": name, "cut_subsets": 2**len(cuts), "powers_checked": "1..10"})
    for k in (2, 3, 4):
        offset = sum((-1)**(k-s)*(s) for s in range(k+1) for _ in combinations(range(k), s))
        constant = sum((-1)**(k-s) for s in range(k+1) for _ in combinations(range(k), s))
        require(offset == 0 and constant == 0, "affine term does not cancel")
    return {"identity": "tr(L_{G-S}^n) = tr(K[E-S]^n) for n>=1, and the zeroth traces differ by |V|-|E-S|",
            "affine_term_cancels_for_k": [2, 3, 4], "fixtures": records}


def z_matrix_orientations():
    cases = [("path P5", [(i, i+1) for i in range(4)]),
             ("cycle C5", [(i, (i+1) % 5) for i in range(5)]),
             ("cycle C6", [(i, (i+1) % 6) for i in range(6)]),
             ("triangle", [(0, 1), (1, 2), (0, 2)]),
             ("star K13", STAR3),
             ("spider 222", SPIDER),
             ("HI fixture", HI),
             ("K4", K4)]
    records = []
    for name, edges in cases:
        edges = edge_list(edges)
        degree = {}
        for u, v in edges:
            degree[u] = degree.get(u, 0)+1
            degree[v] = degree.get(v, 0)+1
        exists = any(all(x <= 0 for i, row in enumerate(edge_gram(edges, o))
                         for j, x in enumerate(row) if i != j)
                     for o in product((0, 1), repeat=len(edges)))
        require(exists == (max(degree.values()) <= 2), name+": orientation criterion")
        records.append({"fixture": name, "max_degree": max(degree.values()),
                        "nonpositive_offdiagonal_orientation_exists": exists})
    for o in product((0, 1), repeat=3):
        k = edge_gram(STAR3, o)
        require(k[0][1]*k[0][2]*k[1][2] == 1, "three edges at a vertex")
    return {"criterion": "some orientation makes K a Z-matrix iff maximum degree <= 2",
            "three_edges_at_a_vertex_offdiagonal_product": 1, "fixtures": records}


def heat_trace_interval(n, edges, t, terms):
    lap = laplacian(n, edges)
    spectral_bound = 2*max((lap[i][i] for i in range(n)), default=0)
    traces = power_traces(lap, terms)
    total, term = Q(0), Q(1)
    for m in range(terms+1):
        if m:
            term = term*(-t)/m
        total += term*traces[m]
    x = t*spectral_bound
    require(terms+2 > x, "too few terms for the tail bound")
    tail = n*x**(terms+1)/factorial(terms+1)*Q(terms+2)/(terms+2-x)
    return total-tail, total+tail


def heat_interaction_interval(n, edges, cuts, t, terms):
    lo = hi = Q(0)
    for cut in subsets(cuts):
        a, b = heat_trace_interval(n, [e for e in edges if e not in cut], t, terms)
        if (len(cuts)-len(cut)) % 2 == 0:
            lo, hi = lo+a, hi+b
        else:
            lo, hi = lo-b, hi-a
    return lo, hi


def sign_of(interval):
    lo, hi = interval
    require(lo > 0 or hi < 0, "interval does not determine the sign")
    return 1 if lo > 0 else -1


def sign_law():
    times = [Q(1, 4), Q(1), Q(3)]
    cases = [("path P7", 7, PATH7, [(1, 2), (4, 5)], True),
             ("path P7", 7, PATH7, [(0, 1), (2, 3), (5, 6)], True),
             ("cycle C8", 8, CYCLE8, [(0, 1), (4, 5)], True),
             ("cycle C8", 8, CYCLE8, [(0, 1), (3, 4), (5, 6)], True),
             ("cycle C8", 8, CYCLE8, [(0, 1), (2, 3), (4, 5), (6, 7)], True),
             ("star K13", 4, STAR3, STAR3, False),
             ("HI fixture", 5, HI, [(0, 1), (0, 3), (0, 4)], False)]
    records = []
    for name, n, edges, cuts, low_degree in cases:
        edges, cuts = edge_list(edges), edge_list(cuts)
        predicted = (-1)**len(cuts)
        signs = []
        for t in times:
            interval = heat_interaction_interval(n, edges, cuts, t, 140)
            signs.append(sign_of(interval))
            records.append({"fixture": name, "cuts": cuts, "t": str(t),
                            "interval": [float(interval[0]), float(interval[1])],
                            "sign": signs[-1], "sign_law_prediction": predicted})
        if low_degree:
            require(all(s == predicted for s in signs), name+": sign law fails at degree <= 2")
        else:
            require(any(s != predicted for s in signs), name+": expected a branch-point failure")
    return {"law": "(-1)^k H_F(t) > 0 for k cuts in one component when maximum degree <= 2",
            "times": [str(t) for t in times], "records": records}


def adjacency_interaction():
    cases = [("HI fixture", 5, HI, [(0, 1), (0, 3), (0, 4)]),
             ("path P5", 5, [(i, i+1) for i in range(4)], [(0, 1), (3, 4)]),
             ("K4", 4, K4, [(0, 1), (2, 3)]),
             ("C5 plus chord", 5, [(i, (i+1) % 5) for i in range(5)]+[(0, 2)],
              [(0, 2), (3, 4)])]
    top = 14
    records = []
    for name, n, edges, cuts in cases:
        edges, cuts = edge_list(edges), edge_list(cuts)
        index = {e: i for i, e in enumerate(cuts)}
        full = (1 << len(cuts))-1
        nbrs = [[] for _ in range(n)]
        for u, v in edges:
            nbrs[u].append(v)
            nbrs[v].append(u)
        covering = [0]*(top+1)
        for start in range(n):
            state = {(start, 0): 1}
            for m in range(top+1):
                covering[m] += state.get((start, full), 0)
                nxt = {}
                for (v, mask), c in state.items():
                    for w in nbrs[v]:
                        e = (min(v, w), max(v, w))
                        key = (w, mask | (1 << index[e]) if e in index else mask)
                        nxt[key] = nxt.get(key, 0)+c
                state = nxt
        inclusion = [0]*(top+1)
        for cut in subsets(cuts):
            tr = power_traces(adjacency(n, [e for e in edges if e not in cut]), top)
            for m in range(top+1):
                inclusion[m] += (-1)**(len(cuts)-len(cut))*tr[m]
        require(inclusion == [(-1)**len(cuts)*c for c in covering], name+": covering-walk identity")
        onset = next(m for m, c in enumerate(covering) if c)
        records.append({"fixture": name, "cuts": cuts, "onset": onset,
                        "leading_covering_walks": covering[onset]})
    return {"identity": "sum_S (-1)^(k-|S|) tr A(G-S)^m = (-1)^k #closed walks of length m using every cut edge",
            "powers_checked": "0..%d" % top, "fixtures": records}


def spanning_trees(n, edges):
    lap = laplacian(n, edges)
    m = [[Q(lap[i][j]) for j in range(1, n)] for i in range(1, n)]
    det = Q(1)
    for c in range(n-1):
        p = next((r for r in range(c, n-1) if m[r][c] != 0), None)
        if p is None:
            return 0
        if p != c:
            m[c], m[p] = m[p], m[c]
            det = -det
        det *= m[c][c]
        for r in range(c+1, n-1):
            f = m[r][c]/m[c][c]
            if f:
                for j in range(c, n-1):
                    m[r][j] -= f*m[c][j]
    require(det.denominator == 1 and det >= 0, "matrix-tree determinant")
    return int(det)


def spanning_tree_interactions():
    cases = [("K4", 4, K4), ("K5", 5, K5), ("wheel W6", 6, WHEEL6),
             ("Petersen", 10, PETERSEN), ("G7", 7, G7)]
    pair_records, triple_counts, triple_examples = [], {"+": 0, "-": 0, "0": 0}, {}
    for name, n, edges in cases:
        edges = edge_list(edges)
        cache = {}

        def tau(cut):
            key = frozenset(cut)
            if key not in cache:
                cache[key] = spanning_trees(n, [e for e in edges if e not in key])
            return cache[key]
        strict = equal = 0
        for e, f in combinations(edges, 2):
            if tau((e, f)) == 0:
                continue
            lhs, rhs = tau(())*tau((e, f)), tau((e,))*tau((f,))
            require(lhs <= rhs, name+": two-bond spanning-tree sign")
            strict, equal = strict+(lhs < rhs), equal+(lhs == rhs)
        pair_records.append({"fixture": name, "pairs_strict": strict, "pairs_equal": equal})
        for e, f, g in combinations(edges, 3):
            if tau((e, f, g)) == 0:
                continue
            num = tau((e, f, g))*tau((e,))*tau((f,))*tau((g,))
            den = tau((e, f))*tau((e, g))*tau((f, g))*tau(())
            key = "+" if num > den else "-" if num < den else "0"
            triple_counts[key] += 1
            triple_examples.setdefault(key, {"fixture": name, "cuts": [e, f, g],
                                             "ratio": str(Q(num, den))})
    require(triple_counts["+"] > 0 and triple_counts["-"] > 0, "three-bond signs")
    e, f = (0, 4), (2, 4)
    g7 = edge_list(G7)
    taus = [spanning_trees(7, [x for x in g7 if x not in cut]) for cut in ((), (e,), (f,), (e, f))]
    heat = {}
    for t in (Q(1, 2), Q(1)):
        interval = heat_interaction_interval(7, g7, [e, f], t, 140)
        heat[str(t)] = {"interval": [float(interval[0]), float(interval[1])], "sign": sign_of(interval)}
    require(heat["1/2"]["sign"] == 1 and heat["1"]["sign"] == -1, "G7 heat pair sign change")
    return {"pair_law": "tau(G) tau(G-e-f) <= tau(G-e) tau(G-f) whenever G-e-f is connected",
            "pair_fixtures": pair_records,
            "three_bond_log_tau_signs": triple_counts, "three_bond_examples": triple_examples,
            "G7_pair_04_24": {"spanning_tree_ratio": str(Q(taus[0]*taus[3], taus[1]*taus[2])),
                              "heat_interaction": heat}}


def circulant_square_lattice(N=101, k=10, max_radius=6):
    require(all(N % d for d in range(2, int(N**0.5)+1)), "N must be prime")
    steps = {1, N-1, k, N-k}
    records = []
    for r in range(1, max_radius+1):
        ball = [(a, b) for a in range(-r, r+1) for b in range(-r, r+1) if abs(a)+abs(b) <= r]
        phi = {p: (p[0]+k*p[1]) % N for p in ball}
        dist, frontier = {0: 0}, [0]
        for d in range(r):
            nxt = []
            for v in frontier:
                for s in steps:
                    w = (v+s) % N
                    if w not in dist:
                        dist[w] = d+1
                        nxt.append(w)
            frontier = nxt
        ok = len(set(phi.values())) == len(ball) and set(phi.values()) == set(dist)
        ok = ok and all((((phi[p]-phi[q]) % N) in steps) == (abs(p[0]-q[0])+abs(p[1]-q[1]) == 1)
                        for p in ball for q in ball if p != q)
        records.append({"radius": r, "ball_vertices": len(ball), "rooted_ball_equals_square_lattice": ok})
    agree = [x["radius"] for x in records if x["rooted_ball_equals_square_lattice"]]
    require(agree[:4] == [1, 2, 3, 4], "small circulant balls should be lattice balls")
    return {"graph": "C_%d(1,%d), vertex-transitive and Cartesian-prime because %d is prime" % (N, k, N),
            "radii": records}


def mixture_factorization():
    def mul(a, b):
        out = [0]*(len(a)+len(b)-1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i+j] += x*y
        return out
    left = mul([Q(1, 3)]*3, [Q(1, 2), 0, 0, Q(1, 2)])
    right = mul([Q(1, 2), Q(1, 2)], [Q(1, 3), 0, Q(1, 3), 0, Q(1, 3)])
    require(left == right == [Q(1, 6)]*6, "mixture factorizations")
    return {"identity": "(1+H+H^2)/3 * (1+H^3)/2 = (1+H)/2 * (1+H^2+H^4)/3 = (1+...+H^5)/6",
            "factors_positive_mass_one": True, "factor_sets_differ": True}


def serialize(value):
    if isinstance(value, Q):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("application_assessment_results.json"))
    args = parser.parse_args()
    results = {
        "status": "passed",
        "arithmetic": "Python standard library; exact integers and rationals, certified heat-series tails",
        "edge_space_duality": edge_space_duality(),
        "z_matrix_orientations": z_matrix_orientations(),
        "sign_law": sign_law(),
        "adjacency_interaction": adjacency_interaction(),
        "spanning_tree_interactions": spanning_tree_interactions(),
        "circulant_square_lattice": circulant_square_lattice(),
        "mixture_factorization": mixture_factorization(),
        "scope": "Finite fixtures for the assessment's new observations; literature verdicts are not checked.",
        "proof_review": "The general arguments in APPLICATION_ASSESSMENT.md remain unreviewed.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(serialize(results), indent=2, sort_keys=True)+"\n")
    print("PASS: application assessment fixtures:", args.output)


if __name__ == "__main__":
    main()
