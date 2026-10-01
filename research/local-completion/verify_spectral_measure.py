"""Exact, standalone fixtures for SPECTRAL_MEASURE_HOMOMORPHISM.md.

No graphlocal or third-party imports. Matrix calculations, permutation
determinants, rational arithmetic, and explicit graph rewiring are the
oracles. Finite calculations do not certify an infinite moment condition.
"""
from collections import Counter
from fractions import Fraction as Q
from itertools import permutations
from math import comb, factorial, prod
from pathlib import Path
import argparse
import hashlib
import json


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def graph(n, edges):
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        require(0 <= u < n and 0 <= v < n and u != v, "invalid edge")
        require(v not in neighbors[u], "duplicate edge")
        neighbors[u].add(v)
        neighbors[v].add(u)
    return tuple(tuple(sorted(row)) for row in neighbors)


def path(n):
    return graph(n, ((i, i + 1) for i in range(n - 1)))


def cycle(n):
    require(n >= 3, "simple cycle needs at least three vertices")
    return graph(n, [(i, i + 1) for i in range(n - 1)] + [(0, n - 1)])


def edge_set(g):
    return {(u, v) for u, row in enumerate(g) for v in row if u < v}


def cartesian(g, h):
    n, m = len(g), len(h)
    edges = [(u * m + w, v * m + w) for u, v in edge_set(g)
             for w in range(m)]
    edges += [(w * m + u, w * m + v) for u, v in edge_set(h)
              for w in range(n)]
    return graph(n * m, edges)


def identity(n, scale=1):
    return [[scale * int(i == j) for j in range(n)] for i in range(n)]


def laplacian(g):
    result = [[0] * len(g) for _ in g]
    for u, row in enumerate(g):
        result[u][u] = len(row)
        for v in row:
            result[u][v] = -1
    return result


def adjacency(g):
    return [[int(j in row) for j in range(len(g))] for row in g]


def shifted(matrix, scalar):
    return [[scalar * int(i == j) - value for j, value in enumerate(row)]
            for i, row in enumerate(matrix)]


def multiply(a, b):
    """Dense left factor with sparse-row right factor, exact integers."""
    sparse = [[(j, v) for j, v in enumerate(row) if v] for row in b]
    result = [[0] * len(b) for _ in a]
    for i, row in enumerate(a):
        for k, value in enumerate(row):
            if value:
                for j, other in sparse[k]:
                    result[i][j] += value * other
    return result


def powers(matrix, order):
    result = [identity(len(matrix))]
    for _ in range(order):
        result.append(multiply(result[-1], matrix))
    return result


def trace(matrix):
    return sum(matrix[i][i] for i in range(len(matrix)))


def traces(matrix, order):
    current = identity(len(matrix))
    result = [len(matrix)]
    for _ in range(order):
        current = multiply(current, matrix)
        result.append(trace(current))
    return result


def atom_moments(atoms, order):
    return [sum(weight * value**j for value, weight in atoms.items())
            for j in range(order + 1)]


def binomial_moments(a, b):
    return [sum(comb(j, i) * a[i] * b[j-i] for i in range(j+1))
            for j in range(min(len(a), len(b)))]


def bernstein_row(moments, n):
    return [comb(n, k) * sum((-1)**i * comb(n-k, i) * moments[k+i]
                            for i in range(n-k+1)) for k in range(n+1)]


def check_elevation(rows):
    for n, row in enumerate(rows[:-1]):
        following = rows[n+1]
        require(row == [Q(n+1-k, n+1) * following[k]
                        + Q(k+1, n+1) * following[k+1] for k in range(n+1)],
                "Bernstein elevation disagrees")
        require(sum(map(abs, row)) <= sum(map(abs, following)),
                "Bernstein variations decreased")


def bernstein_fixtures():
    fixtures = {
        "K1": (path(1), 1, {0: 1}),
        "K2": (path(2), 1, {0: 1, 2: 1}),
        "P3": (path(3), 2, {0: 1, 1: 1, 3: 1}),
        "C3": (cycle(3), 2, {0: 1, 3: 2}),
        "C4": (cycle(4), 2, {0: 1, 2: 2, 4: 1}),
        "star4": (graph(4, [(0, 1), (0, 2), (0, 3)]), 3,
                  {0: 1, 1: 2, 4: 1}),
    }
    order = 12
    output = {}
    for name, (g, cap, atoms) in fixtures.items():
        L = laplacian(g)
        lp, qp = powers(L, order), powers(shifted(L, 2*cap), order)
        raw = [trace(p) for p in lp]
        require(raw == atom_moments(atoms, order), name + " atomic moments")
        scaled = [Q(value, (2*cap)**j) for j, value in enumerate(raw)]
        rows = [bernstein_row(scaled, n) for n in range(order+1)]
        for n, row in enumerate(rows):
            direct = [Q(comb(n, k) * trace(multiply(lp[k], qp[n-k])),
                        (2*cap)**n) for k in range(n+1)]
            require(row == direct, name + " direct matrix Bernstein")
            require(all(value >= 0 for value in row), name + " positivity")
            require(sum(row) == len(g), name + " mass")
        check_elevation(rows)
        output[name] = {"degree_cap": cap, "vertices": len(g),
                        "atoms": atoms, "row_12": rows[-1]}

    signed_atoms = {0: Q(1, 6), 1: Q(1, 6), 3: Q(1, 6),
                    4: Q(-1, 6), 5: Q(-1, 6), 7: Q(-1, 6)}
    signed = atom_moments(signed_atoms, 32)
    rows = [bernstein_row([Q(v, 8**j) for j, v in enumerate(signed)], n)
            for n in range(33)]
    for n, row in enumerate(rows):
        direct = [sum(weight * comb(n, k) * (Q(value, 8)**k)
                      * (1-Q(value, 8))**(n-k)
                      for value, weight in signed_atoms.items())
                  for k in range(n+1)]
        require(row == direct, "signed atomic Bernstein")
        require(sum(map(abs, row)) <= 1, "signed spectral variation")
    check_elevation(rows)

    line_moments = [Q(comb(2*j, j), 4**j) for j in range(33)]
    line_rows = [bernstein_row(line_moments, n) for n in range(33)]
    for n, row in enumerate(line_rows):
        beta = [Q(comb(2*k, k) * comb(2*(n-k), n-k), 4**n)
                for k in range(n+1)]
        require(row == beta and sum(row) == 1, "line beta-integral row")
    check_elevation(line_rows)
    return {"finite_matrix_horizon": order, "finite_fixtures": output,
            "signed_fixture": "E U(P3)", "signed_row_horizon": 32,
            "signed_variation_lower_bounds":
                {n: sum(map(abs, rows[n])) for n in (1, 4, 8, 16, 32)},
            "line_beta_formula_horizon": 32,
            "finite_prefix_is_not_an_infinite_bound_certificate": True}


def product_fixtures():
    output = []
    for name, g, h in (("P3 x K2", path(3), path(2)),
                       ("P3 x C4", path(3), cycle(4)),
                       ("C3 x C4", cycle(3), cycle(4))):
        order = 12
        a, b = traces(laplacian(g), order), traces(laplacian(h), order)
        actual = traces(laplacian(cartesian(g, h)), order)
        require(actual == binomial_moments(a, b), name + " moment product")
        output.append({"fixture": name, "moments": actual})

    order, n = 10, 12
    E = [0] + [-Q(4**j, 2) for j in range(1, order+1)]
    delta = [a-b for a, b in zip(traces(laplacian(path(n)), order),
                                 traces(laplacian(cycle(n)), order))]
    require(delta == E, "stabilized cut-line moments")
    e2 = [0] * (order+1)
    for g, cg in ((path(n), 1), (cycle(n), -1)):
        for h, ch in ((path(n), 1), (cycle(n), -1)):
            values = traces(laplacian(cartesian(g, h)), order)
            e2 = [v+cg*ch*w for v, w in zip(e2, values)]
    expected_e2 = atom_moments({0: Q(1, 4), 4: Q(-1, 2), 8: Q(1, 4)}, order)
    require(e2 == expected_e2 == binomial_moments(E, E), "crossing-cut moments")
    resolvents = []
    for k in range(1, 7):
        atoms = {4*a: Q((-1)**a * comb(k, a), 2**k) for a in range(k+1)}
        require(sum(map(abs, atoms.values())) == 1, "cut power variation")
        for s in (Q(1, 2), Q(1), Q(3)):
            actual = sum(w/(s+v) for v, w in atoms.items())
            expected = 2**k * factorial(k) / prod(s+4*a for a in range(k+1))
            require(actual == expected, "cut power resolvent")
            resolvents.append({"k": k, "s": s, "value": actual})
    for j in range(1, 17):
        # Integral of -p'/2 on (0,4), paired with the spectral shift.
        require(-Q(4**j, 2) == atom_moments({0: Q(1, 2), 4: Q(-1, 2)}, j)[j],
                "spectral-shift derivative sign")
    return {"finite_products": output, "cut_line_n": n,
            "crossing_cut_host_vertices": n*n, "cut_moment_horizon": order,
            "cut_line_moments": E, "crossing_cut_moments": e2,
            "resolvent_fixtures": resolvents,
            "cut_line_shift": {"interval": [0, 4], "value": "-1/2",
                               "integral": -2, "moment_horizon": 16}}


def ball_signature(g, root, radius):
    distance, frontier = {root: 0}, [root]
    while frontier:
        following = []
        for u in frontier:
            if distance[u] < radius:
                for v in g[u]:
                    if v not in distance:
                        distance[v] = distance[u]+1
                        following.append(v)
        frontier = following
    selected = set(distance)
    # For induced balls in cycles these distinguish full cycles and centered paths.
    degrees = [sum(v in selected for v in g[u]) for u in selected]
    return len(selected), tuple(sorted(degrees)), tuple(sorted(distance.values()))


def chebyshev_cycle_values(n, horizon):
    """Return tr(T_k(A/2))/n by the integer recurrence for 2 T_k."""
    A = adjacency(cycle(n))
    previous, current = identity(n, 2), A
    values = [Q(1), Q(trace(A), 2*n)]
    for _ in range(1, horizon):
        following = multiply(current, A)
        following = [[v-w for v, w in zip(row, old)]
                     for row, old in zip(following, previous)]
        previous, current = current, following
        values.append(Q(trace(current), 2*n))
    return values


def cycle_counterexample():
    rewiring = []
    for n in (3, 9, 27):
        original = edge_set(cycle(2*n))
        deleted = {(n-1, n), (0, 2*n-1)}
        inserted = {(0, n-1), (n, 2*n-1)}
        edited = original-deleted | inserted
        expected = edge_set(cycle(n)) | {(u+n, v+n) for u, v in edge_set(cycle(n))}
        require(edited == expected and len(original ^ edited) == 4,
                "four-edge cycle rewiring")
        for r in range(13):
            if n > 2*r+1:
                require(ball_signature(cycle(n), 0, r)
                        == ball_signature(cycle(2*n), 0, r),
                        "cycle local cancellation")
        rewiring.append({"n": n, "deleted": sorted(deleted),
                         "inserted": sorted(inserted), "budget": Q(2, n)})

    cheb = {n: chebyshev_cycle_values(n, 27) for n in (3, 6, 9, 18, 27, 54)}
    for n, values in cheb.items():
        for k, value in enumerate(values):
            require(value == int(k % n == 0), "cycle Chebyshev trace oracle")
    witnesses = []
    for m in range(1, 4):
        k = 3**m
        observed = sum(cheb[3**j][k]-cheb[2*3**j][k] for j in range(1, 4))
        require(observed == m, "unbounded spectral-functional witness")
        witnesses.append({"m": m, "polynomial_degree": k,
                          "uniform_norm_on_0_4": 1, "matrix_value": observed})

    # Direct walk counts on the line: each signed cycle difference selects
    # endpoint displacements divisible by N but not 2N. Terms with N>k vanish.
    horizon = 48
    d = []
    for k in range(horizon+1):
        value, n = Q(0), 3
        while n <= k:
            value += Q(sum(comb(k, i) for i in range(k+1)
                           if (2*i-k) % n == 0 and (2*i-k) % (2*n) != 0), 2**k)
            n *= 3
        require(abs(value) <= k, "edit-profile moment bound")
        d.append(value)
    # Independent finite matrices verify the first part of the walk calculation.
    matrix_d = [Q(0)] * 13
    for n in (3, 9):
        a, b = traces(adjacency(cycle(n)), 12), traces(adjacency(cycle(2*n)), 12)
        matrix_d = [v + Q(x, n*2**k) - Q(y, 2*n*2**k)
                    for k, (v, x, y) in enumerate(zip(matrix_d, a, b))]
    require(matrix_d == d[:13], "walk/matrix moments of cycle limit")
    a = [sum((-1)**i * comb(j, i) * d[i] for i in range(j+1))/2**j
         for j in range(horizon+1)]
    rows = [bernstein_row(a, n) for n in range(horizon+1)]
    check_elevation(rows)
    return {"rewiring": rewiring, "local_cancellation_radii": [0, 12],
            "chebyshev_matrix_witnesses": witnesses,
            "walk_moment_horizon": horizon, "matrix_moment_horizon": 12,
            "profile": [0, 2], "degree_cap": 2, "edit_budget_bound": 1,
            "partial_budgets": {m: 1-Q(1, 3**m) for m in range(1, 7)},
            "bernstein_variation_lower_bounds":
                {n: sum(map(abs, rows[n])) for n in (3, 9, 27, 48)},
            "infinite_nonmeasure_conclusion_uses_written_Chebyshev_proof": True}


def polynomial_multiply(a, b):
    result = [0] * (len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i+j] += x*y
    return result


def determinant_polynomial(matrix):
    """det(z I-L) by permutations; coefficients in decreasing order."""
    n = len(matrix)
    result = [0] * (n+1)
    for perm in permutations(range(n)):
        inversions = sum(perm[i] > perm[j] for i in range(n) for j in range(i+1, n))
        term = [(-1)**inversions]
        for i, j in enumerate(perm):
            term = polynomial_multiply(term, [-matrix[i][j], int(i == j)])
        result = [a+b for a, b in zip(result, term)]
    return list(reversed(result))


def newton_polynomial(matrix):
    n = len(matrix)
    moments = traces(matrix, n)
    coefficients = [1]
    for k in range(1, n+1):
        value = -sum(coefficients[k-i]*moments[i] for i in range(1, k+1))
        require(value % k == 0, "Newton coefficient integrality")
        coefficients.append(value // k)
    return coefficients


def integer_factors(coefficients, maximum=8):
    remaining, roots = list(coefficients), []
    for root in range(maximum+1):
        while len(remaining) > 1:
            divided = [remaining[0]]
            for coefficient in remaining[1:]:
                divided.append(coefficient+root*divided[-1])
            if divided[-1]:
                break
            roots.append(root)
            remaining = divided[:-1]
    return roots, remaining


def evaluate(coefficients, x):
    value = 0
    for coefficient in coefficients:
        value = value*x+coefficient
    return value


def heat_interval(uniformized_integer_moments, cap, time, variation):
    """Rational uniformization interval with a geometric Poisson-tail bound."""
    M = len(uniformized_integer_moments)-1
    lam = time*cap
    require(M+2 > lam, "insufficient heat truncation")
    term, S, numerator = Q(1), Q(1), Q(uniformized_integer_moments[0])
    for j in range(1, M+1):
        term *= lam/j
        S += term
        numerator += term * Q(uniformized_integer_moments[j], cap**j)
    tail = term*lam/(M+1)/(1-lam/(M+2))
    require(abs(numerator) <= variation*S, "retained trace bound")
    return ((numerator-variation*tail)/(S+tail),
            (numerator+variation*tail)/(S+tail))


def outward_decimal(interval, digits=12):
    scale = 10**digits
    lower = interval[0]*scale
    upper = interval[1]*scale
    lo, hi = lower.numerator//lower.denominator, -((-upper.numerator)//upper.denominator)
    require(Q(lo, scale) <= interval[0] <= interval[1] <= Q(hi, scale),
            "outward decimal rounding")
    def render(value):
        return ("-" if value < 0 else "") + str(abs(value)//scale) + "." + str(abs(value)%scale).zfill(digits)
    return [render(lo), render(hi)]


def interaction_fixture():
    edges = {(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)}
    cuts = [(0, 1), (0, 3), (0, 4)]
    cap, horizon = 4, 100
    uniformized = [0] * (horizon+1)
    integer_atoms, other_factors = Counter(), Counter()
    records = []
    for mask in range(8):
        deleted = {edge for i, edge in enumerate(cuts) if mask >> i & 1}
        L = laplacian(graph(5, sorted(edges-deleted)))
        determinant, newton = determinant_polynomial(L), newton_polynomial(L)
        require(determinant == newton, "independent characteristic polynomials")
        roots, remaining = integer_factors(determinant)
        sign = (-1)**(3-mask.bit_count())
        for root in roots:
            integer_atoms[root] += sign
        if remaining != [1]:
            other_factors[tuple(remaining)] += sign
        current = traces(shifted(L, cap), horizon)
        uniformized = [a+sign*b for a, b in zip(uniformized, current)]
        records.append({"mask": mask, "sign": sign, "deleted": sorted(deleted),
                        "characteristic_polynomial": determinant,
                        "integer_roots": roots, "remaining_factor": remaining})
    integer_atoms = {r: w for r, w in integer_atoms.items() if w}
    require(integer_atoms == {1: 2, 2: -2, 4: 2, 5: -1}, "interaction integer atoms")
    require(dict(other_factors) == {(1, -7, 13, -5): 1, (1, -4, 2): -2},
            "interaction algebraic atoms")
    cubic, quadratic = [1, -7, 13, -5], [1, -4, 2]
    brackets = [(Q(5188, 10000), Q(5189, 10000)), (Q(2), Q(3)), (Q(4), Q(5))]
    for lo, hi in brackets:
        require(evaluate(cubic, lo)*evaluate(cubic, hi) < 0, "cubic root bracket")
    qbrackets = [(Q(5857, 10000), Q(5858, 10000)),
                 (Q(34142, 10000), Q(34143, 10000))]
    for lo, hi in qbrackets:
        require(evaluate(quadratic, lo)*evaluate(quadratic, hi) < 0, "quadratic root bracket")
    # These brackets contain all roots by degree. No atom coincidences remain.
    require(qbrackets[0][0]-brackets[0][1] > Q(66, 1000), "lowest-atom gap")
    exp_lower = sum(Q(66, 25)**j/factorial(j) for j in range(13))
    require(exp_lower > 13, "eventual-sign exponential bound")
    intervals = {}
    for time, expected_sign in ((Q(1), 1), (Q(3), -1)):
        bounds = heat_interval(uniformized, cap, time, variation=40)
        require(bounds[0] > 0 if expected_sign > 0 else bounds[1] < 0,
                "certified interaction heat sign")
        require(bounds[1]-bounds[0] < Q(1, 10**12), "heat enclosure width")
        intervals[str(time)] = {"exact_rational_interval": bounds,
                                "outward_decimal_interval": outward_decimal(bounds)}
    return {"edges": sorted(edges), "cuts": cuts, "subset_factorizations": records,
            "integer_atoms": integer_atoms,
            "other_atoms": [{"polynomial": list(p), "weight_per_root": w}
                             for p, w in sorted(other_factors.items())],
            "cubic_root_brackets": brackets, "quadratic_root_brackets": qbrackets,
            "spectral_total_variation": 14, "heat_degree_cap": cap,
            "heat_moment_horizon": horizon, "heat_intervals": intervals,
            "exp_2_64_rational_lower_bound": exp_lower,
            "positive_for_all_times_at_least": 40,
            "proven_zero_intervals": [[1, 3], [3, 40]],
            "exact_number_of_zeros_claimed": False}


def serializable(value):
    if isinstance(value, Q):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("spectral_measure_results.json"))
    args = parser.parse_args()
    result = {"status": "passed", "arithmetic": "Python standard-library exact integers and fractions",
              "bernstein": bernstein_fixtures(), "products_and_defects": product_fixtures(),
              "controlled_nonmeasure": cycle_counterexample(), "interaction": interaction_fixture(),
              "proof_review": "Independent proof review deferred; finite fixtures are not universal proofs.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(serializable(result), indent=2, sort_keys=True)+"\n")
    print("PASS: spectral-measure fixtures; exact result record:", args.output)


if __name__ == "__main__":
    main()
