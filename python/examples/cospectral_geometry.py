"""Exact cospectral graphs with distinct local graph-number coordinates.

Run with ``PYTHONPATH=python/src python python/examples/cospectral_geometry.py``
from the repository root. Uses only Python's standard library and graphlocal.
No numerical eigenvalue calculation or timing comparison is involved.

The 4-by-4 rook graph and the Shrikhande graph both satisfy A^2 = 4I + 2J.
Their common adjacency and Laplacian spectra therefore give identical scalar
spectral traces, including heat traces, while rooted radius-one histograms
and K4 counts distinguish them. This is a representation example: ordinary
graph algorithms can also distinguish their neighborhoods.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
import json

from graphlocal import (LocalHistogram, ball, cartesian, complete, cycle,
                       disjoint_union, graph, isomorphic, path)
from graphlocal.graphs import components, induced, integer


def rook_graph():
    """The graph K4 Cartesian K4, with vertex (x,y) encoded as 4*x+y."""
    return cartesian(complete(4), complete(4))


def shrikhande_graph():
    """Cayley graph of Z4^2 for steps +/-(1,0), +/-(0,1), +/-(1,1)."""
    steps = ((1, 0), (3, 0), (0, 1), (0, 3), (1, 1), (3, 3))
    return graph(16, ((4 * x + y, 4 * ((x + dx) % 4) + (y + dy) % 4)
                      for x in range(4) for y in range(4)
                      for dx, dy in steps
                      if 4 * x + y < 4 * ((x + dx) % 4) + (y + dy) % 4))


def count_cliques(g, size):
    """Count unlabeled vertex subsets inducing a complete graph, exactly."""
    integer(size, "clique size", 1)
    return sum(all(g.rows[u] & (1 << v) for u, v in combinations(vertices, 2))
               for vertices in combinations(range(g.n), size))


def diagonal_moments(g, max_power, *, laplacian=False):
    """Return every diagonal entry of A**m or (D-A)**m, for 0 <= m <= M.

    Multiplication uses integer arithmetic; no spectrum is assumed. Keeping
    individual diagonal entries also verifies rooted closed-walk counts.
    """
    integer(max_power, "maximum power")
    neighbors = tuple(tuple(g.neighbors(u)) for u in range(g.n))
    powers = [[int(u == v) for v in range(g.n)] for u in range(g.n)]
    result = [tuple(powers[u][u] for u in range(g.n))]
    for _ in range(max_power):
        if laplacian:
            powers = [[len(neighbors[u]) * powers[u][v]
                       - sum(powers[w][v] for w in neighbors[u])
                       for v in range(g.n)] for u in range(g.n)]
        else:
            powers = [[sum(powers[w][v] for w in neighbors[u])
                       for v in range(g.n)] for u in range(g.n)]
        result.append(tuple(powers[u][u] for u in range(g.n)))
    return tuple(result)


def verify_geometry(max_power=12):
    """Verify all reported identities and return a JSON-serializable report.

    The exact symmetric-matrix identity, regularity and trace determine the
    full spectra, not just the finitely many checked moments. On constants
    A acts as 6; on their orthogonal complement A**2=4I, hence A has roots
    +2 and -2. Dimension 15 and trace zero give multiplicities 6 and 9.
    For the combinatorial Laplacian 6I-A these become 0, 4, 8.
    """
    integer(max_power, "maximum power")
    checks = 0

    def require(condition, description):
        nonlocal checks
        if not condition:
            raise AssertionError(description)
        checks += 1

    rook, shrikhande = rook_graph(), shrikhande_graph()
    graphs = {"rook": rook, "shrikhande": shrikhande}
    expected_neighbors = {"rook": disjoint_union(complete(3), complete(3)),
                          "shrikhande": cycle(6)}
    graph_reports = {}
    for name, g in graphs.items():
        require(g.n == 16 and g.edges == 48, f"{name}: order and edge count")
        require(all(row.bit_count() == 6 for row in g.rows), f"{name}: regularity")
        require(len(tuple(components(g))) == 1, f"{name}: connectedness")
        for u in range(g.n):
            for v in range(g.n):
                # Symmetry turns matrix entry (A**2)[u,v] into a bit count.
                require((g.rows[u] & g.rows[v]).bit_count() == 4 * int(u == v) + 2,
                        f"{name}: A^2=4I+2J at ({u},{v})")
            neighborhood = induced(g, g.neighbors(u))
            require(isomorphic(neighborhood, expected_neighbors[name]),
                    f"{name}: induced neighborhood at {u}")
            require(isomorphic(ball(g, u, 1), ball(g, 0, 1), rooted=True),
                    f"{name}: rooted radius-one type at {u}")

        four_cliques = count_cliques(g, 4)
        require(four_cliques == (8 if name == "rook" else 0), f"{name}: K4 count")
        adjacency = diagonal_moments(g, max_power)
        laplacian = diagonal_moments(g, max_power, laplacian=True)
        for power in range(max_power + 1):
            adjacency_trace = 6 ** power + 6 * 2 ** power + 9 * (-2) ** power
            laplacian_trace = int(power == 0) + 6 * 4 ** power + 9 * 8 ** power
            require(sum(adjacency[power]) == adjacency_trace,
                    f"{name}: adjacency moment {power}")
            require(sum(laplacian[power]) == laplacian_trace,
                    f"{name}: Laplacian moment {power}")
            require(all(Fraction(value) == Fraction(adjacency_trace, 16)
                        for value in adjacency[power]),
                    f"{name}: rooted closed walks of length {power}")
        graph_reports[name] = {
            "vertices": g.n,
            "edges": g.edges,
            "degree": 6,
            "strongly_regular_parameters": [16, 6, 2, 2],
            "matrix_identity": "A^2 = 4I + 2J",
            "induced_neighborhood": "2 K3" if name == "rook" else "C6",
            "K4_count": four_cliques,
            "K4_count_per_vertex": str(Fraction(four_cliques, g.n)),
            "adjacency_trace_moments": [sum(values) for values in adjacency],
            "laplacian_trace_moments": [sum(values) for values in laplacian],
        }

    histograms = {name: LocalHistogram.from_graph(g, 1, normalize=True)
                  for name, g in graphs.items()}
    require(all(len(hist.values) == 1 and hist.mass == 1
                for hist in histograms.values()), "one normalized rooted type per graph")
    difference = histograms["rook"] - histograms["shrikhande"]
    require(len(difference.values) == 2, "distinct radius-one rooted types")
    require(difference.truncate(0).norm(0) == 0, "radius-zero histograms coincide")
    for weight in (0, 1, 2):
        require(difference.norm(weight) == 2 * 7 ** weight,
                f"normalized radius-one difference with weight {weight}")

    # Every K4 containing a root corresponds to a triangle among its neighbors.
    # Division by four corrects for counting the four possible roots of each K4.
    def rooted_k4_density(rooted):
        return Fraction(count_cliques(induced(rooted, rooted.neighbors(0)), 3), 4)

    for name, hist in histograms.items():
        require(hist.evaluate(rooted_k4_density)
                == Fraction(graph_reports[name]["K4_count"], 16),
                f"{name}: radius-one K4 observable")
    require(histograms["rook"].evaluate(rooted_k4_density)
            - histograms["shrikhande"].evaluate(rooted_k4_density) == Fraction(1, 2),
            "K4 observable separates normalized graph elements")

    # Every clique of size >= 3 in a Cartesian product lies in one coordinate
    # fiber. Thus Q4(G square H) = |H| Q4(G) + |G| Q4(H), and Q4/|V| is
    # additive on normalized finite graphs under Cartesian multiplication.
    factor = path(2)
    product_counts = {}
    for name, g in graphs.items():
        product = cartesian(g, factor)
        count = count_cliques(product, 4)
        require(count == factor.n * count_cliques(g, 4) + g.n * count_cliques(factor, 4),
                f"{name}: Cartesian K4 count identity with K2")
        product_counts[name] = count
    require(Fraction(product_counts["rook"] - product_counts["shrikhande"], 32)
            == Fraction(1, 2), "normalized K4 difference survives multiplication by K2/2")

    return {
        "success": True,
        "exact_checks": checks,
        "arithmetic": "integer and rational; no numerical eigensolver",
        "checked_moment_orders": [0, max_power],
        "graphs": graph_reports,
        "common_adjacency_spectrum": [[6, 1], [2, 6], [-2, 9]],
        "common_laplacian_spectrum": [[0, 1], [4, 6], [8, 9]],
        "spectra_certified_by": "symmetry, regularity, A^2=4I+2J, dimension, trace",
        "common_normalized_heat_trace": "(1 + 6 exp(-4t) + 9 exp(-8t)) / 16",
        "normalized_difference": {
            "radius_zero_norm": "0",
            "radius_one_total_variation": str(difference.norm(0)),
            "radius_one_p_1_1": str(difference.norm(1)),
            "radius_one_p_1_2": str(difference.norm(2)),
            "K4_per_vertex_difference": "1/2",
        },
        "cartesian_product_check": {
            "factor": "K2",
            "product_vertices": 32,
            "product_K4_counts": product_counts,
            "normalized_K4_difference": "1/2",
            "general_identity": "Q4(G square H) = |H| Q4(G) + |G| Q4(H)",
        },
        "interpretation": "Local graph coordinates distinguish these cospectral graphs; "
                          "scalar adjacency and Laplacian spectral traces do not.",
        "scope": "Representation demonstration; no computational speed or novelty claim.",
    }


if __name__ == "__main__":
    print(json.dumps(verify_geometry(), indent=2))
