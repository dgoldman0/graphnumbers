"""Exact finite checks for GHOST_EDGE.md.

The fixtures check the finite inputs to the note: cycle balls stabilize to
path balls, lattice-power ball sizes and root degrees, the coefficient
identities that refute the old quadratic and stub-inverse identities, the
unit threshold for 1 + tG, the divergence of the complete-graph route,
Cartesian additivity of the edge and four-clique counts, and the
rook-Shrikhande phantom. The general proofs in the note are not checked
here. No graphlocal or third-party modules are imported.
"""
from fractions import Fraction as Q
from itertools import combinations, product
from math import comb
from pathlib import Path
import argparse
import hashlib
import json


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def graph(n, edges):
    g = [set() for _ in range(n)]
    for u, v in edges:
        require(u != v and v not in g[u], "simple edge expected")
        g[u].add(v)
        g[v].add(u)
    return g


def cycle(n):
    return graph(n, [(i, (i+1) % n) for i in range(n)])


def cartesian(g, h):
    n, m = len(g), len(h)
    edges = [(a*m+b, c*m+b) for a in range(n) for c in g[a] if a < c for b in range(m)]
    edges += [(a*m+b, a*m+d) for a in range(n) for b in range(m) for d in h[b] if b < d]
    return graph(n*m, edges)


def edge_count(g):
    return sum(len(row) for row in g)//2


def ball(g, root, r):
    dist, frontier = {root: 0}, [root]
    for d in range(r):
        nxt = []
        for v in frontier:
            for w in g[v]:
                if w not in dist:
                    dist[w] = d+1
                    nxt.append(w)
        frontier = nxt
    return dist


def four_cliques(g):
    return sum(1 for q in combinations(range(len(g)), 4)
               if all(b in g[a] for a, b in combinations(q, 2)))


def lattice_ball_size(j, r):
    return sum(1 for p in product(range(-r, r+1), repeat=j) if sum(map(abs, p)) <= r)


def cycle_route():
    records = []
    for r in range(1, 5):
        for n in range(2*r+2, 2*r+9):
            g = cycle(n)
            for root in range(n):
                b = ball(g, root, r)
                inner = sum(1 for v in b for w in g[v] if w in b)//2
                require(len(b) == 2*r+1 and inner == 2*r, "cycle ball is not a path ball")
                require(sorted(b.values()) == [0]+[d for d in range(1, r+1) for _ in (0, 1)],
                        "cycle ball is not centred")
        records.append({"radius": r, "cycle_lengths": [2*r+2, 2*r+8], "ball": "path on %d vertices" % (2*r+1)})
    return {"claim": "every rooted r-ball of C_n with n > 2r+1 is the centred path, so C_n/n has the line's marginals",
            "vertex_mass": 1, "edges_per_vertex": 1, "records": records}


def lattice_powers():
    records = []
    for j in range(5):
        for r in range(1, 4):
            size = lattice_ball_size(j, r)
            formula = sum(2**i*comb(j, i)*comb(r, i) for i in range(min(r, j)+1))
            require(size == formula, "lattice ball formula")
            require(size <= (2*j+1)**r, "polynomial ball bound")
            if r == 1:
                require(size == 2*j+1, "radius-one ball size")
            records.append({"dimension": j, "radius": r, "ball_size": size})
    tori = []
    for j, r in ((1, 2), (2, 1), (2, 2), (3, 1), (3, 2)):
        m = 2*r+3
        g = cycle(m)
        for _ in range(j-1):
            g = cartesian(g, cycle(m))
        require(len(ball(g, 0, r)) == lattice_ball_size(j, r) and len(g[0]) == 2*j,
                "torus ball differs from lattice ball")
        tori.append({"dimension": j, "radius": r, "torus_side": m, "root_degree": 2*j})
    return {"ball_size": "|B_r(Z^j)| = sum_i 2^i C(j,i) C(r,i), polynomial in j of degree r",
            "root_degree": "2j, so the ball types of distinct powers differ at every radius r >= 1",
            "records": records, "torus_approximants": tori}


def polynomial_identities():
    def mul(a, b):
        out = [0]*(len(a)+len(b)-1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i+j] += x*y
        return out
    ghost = [-1, 1]                      # G = U(Z) - 1 in the basis U(Z^j)
    square_minus_twice = [x-2*y for x, y in zip(mul(ghost, ghost), ghost+[0])]
    require(square_minus_twice == [3, -4, 1], "G^2 - 2G")
    stub_product = mul([0, 1], [2, -1])  # U(Z) (2 - U(Z))
    require(stub_product == [0, 2, -1] and stub_product != [1, 0, 0], "stub inverse")
    other_ghost = {"vertex_mass": Q(1, 2)*1-Q(1, 2), "edges_per_vertex": Q(1, 2)*2}
    require(other_ghost == {"vertex_mass": 0, "edges_per_vertex": 1}, "second unit-edge ghost")
    return {"G^2-2G_coefficients_on_U(Z^0),U(Z^1),U(Z^2)": square_minus_twice,
            "U(Z)(2-U(Z))_coefficients": stub_product,
            "second_ghost_with_unit_edge_count": "(U(Z^2)-1)/2"}


def unit_threshold():
    samples = [Q(-1), Q(0), Q(1, 4), Q(49, 100), Q(1, 2), Q(51, 100), Q(1), Q(2)]
    complex_samples = [(Q(1, 2), Q(1)), (Q(1, 4), Q(3)), (Q(3, 5), Q(-2))]
    records = []
    for re, im in [(t, Q(0)) for t in samples]+complex_samples:
        if re == 0 and im == 0:
            unit = True
        else:
            # The zero of (1-t)+t w is w0=(t-1)/t; |w0|<=1 iff |t-1|^2 <= |t|^2.
            unit = (re-1)**2+im**2 > re**2+im**2
        require(unit == (re < Q(1, 2)), "unit threshold")
        records.append({"t": [str(re), str(im)], "unit": unit})
    return {"claim": "1 + tG is a unit exactly when Re t < 1/2", "records": records}


def complete_graph_route():
    records = []
    for n in (3, 5, 10, 20, 50, 100):
        c = Q(2, n*(n-1))
        mass, edges, p12 = c*n, c*n*(n-1)/2, c*n*n**2
        require(edges == 1 and mass == Q(2, n-1) and p12 == Q(2*n*n, n-1), "complete-graph route")
        records.append({"n": n, "vertex_mass": str(mass), "edge_count": str(edges),
                        "p_1_2": str(p12)})
    require(all(Q(records[i+1]["p_1_2"]) > Q(records[i]["p_1_2"]) for i in range(len(records)-1)),
            "p_1_2 should grow")
    return {"sequence": "(2/(n(n-1))) K_n", "records": records,
            "conclusion": "p_{1,2} grows like 2n, so the sequence is unbounded and has no limit"}


def ghost_strata():
    k2, k4 = graph(2, [(0, 1)]), graph(4, list(combinations(range(4), 2)))
    p3, c4 = graph(3, [(0, 1), (1, 2)]), cycle(4)
    pairs = []
    for name, g, h in (("P3xC4", p3, c4), ("K4xK2", k4, k2), ("K4xP3", k4, p3), ("C4xC4", c4, c4)):
        prod = cartesian(g, h)
        require(edge_count(prod) == len(g)*edge_count(h)+edge_count(g)*len(h), name+": edge Leibniz")
        require(four_cliques(prod) == len(g)*four_cliques(h)+four_cliques(g)*len(h),
                name+": four-clique Leibniz")
        pairs.append(name)
    rook = graph(16, [(4*a+b, 4*c+d) for a, b, c, d in product(range(4), repeat=4)
                      if 4*a+b < 4*c+d and (a == c or b == d)])
    steps = {(1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3)}
    shri = graph(16, [(4*a+b, 4*c+d) for a, b, c, d in product(range(4), repeat=4)
                      if 4*a+b < 4*c+d and ((c-a) % 4, (d-b) % 4) in steps])
    require(edge_count(rook) == edge_count(shri) == 48, "rook and Shrikhande edge counts")
    q4 = Q(four_cliques(rook), 16)-Q(four_cliques(shri), 16)
    require(q4 == Q(1, 2), "Q4(X)")
    return {"leibniz_pairs_checked": pairs,
            "X_rook_minus_shrikhande": {"vertex_mass": 0, "edges_per_vertex": 0, "Q4": str(q4)},
            "cut_line_fixture": {"vertex_mass": 0, "edge_count": -1}}


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
                        default=Path(__file__).with_name("ghost_edge_results.json"))
    args = parser.parse_args()
    results = {
        "status": "passed",
        "arithmetic": "Python standard library; exact integers and rationals",
        "cycle_route": cycle_route(),
        "lattice_powers": lattice_powers(),
        "polynomial_identities": polynomial_identities(),
        "unit_threshold": unit_threshold(),
        "complete_graph_route": complete_graph_route(),
        "ghost_strata": ghost_strata(),
        "scope": "Finite inputs to GHOST_EDGE.md; the general proofs are not checked here.",
        "proof_review": "The note's arguments remain unreviewed.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(serialize(results), indent=2, sort_keys=True)+"\n")
    print("PASS: ghost edge fixtures:", args.output)


if __name__ == "__main__":
    main()
