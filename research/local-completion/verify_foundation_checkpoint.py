"""Independent finite fixtures for the October foundation proof checkpoint.

No graphlocal or third-party imports. Small graph isomorphism is exhaustive
permutation enumeration. The JSON contains witnesses and scope, not assertion
counts. Universal claims are proved in the accompanying notes.
"""
import argparse
from collections import Counter, defaultdict, deque
from fractions import Fraction as Q
from functools import lru_cache
from itertools import combinations, permutations
import json
from math import prod
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def graph(n, edges):
    rows = [0] * n
    for u, v in edges:
        require(0 <= u < n and 0 <= v < n and u != v, "invalid edge")
        rows[u] |= 1 << v
        rows[v] |= 1 << u
    return tuple(rows)


def adjacent(rows, u):
    return [v for v in range(len(rows)) if rows[u] & (1 << v)]


def induced(rows, order):
    return graph(len(order), ((i, j) for i, j in combinations(range(len(order)), 2)
                             if rows[order[i]] & (1 << order[j])))


def distances(rows, root):
    result, queue = {root: 0}, deque([root])
    while queue:
        u = queue.popleft()
        for v in adjacent(rows, u):
            if v not in result:
                result[v] = result[u] + 1
                queue.append(v)
    return result


def encode(rows, order):
    return sum(1 << k for k, (i, j) in enumerate(combinations(range(len(rows)), 2))
               if rows[order[i]] & (1 << order[j]))


@lru_cache(None)
def canonical(rows):
    return len(rows), min(encode(rows, p) for p in permutations(range(len(rows))))


@lru_cache(None)
def rooted_key(rows):
    """Root is vertex zero; every other labeling is enumerated."""
    return len(rows), min(encode(rows, (0,) + p)
                          for p in permutations(range(1, len(rows))))


def decode(key):
    n, code = key
    return graph(n, (edge for k, edge in enumerate(combinations(range(n), 2))
                     if code & (1 << k)))


def cone(rows):
    n = len(rows)
    return graph(n + 1, [(0, v + 1) for v in range(n)] +
                 [(u + 1, v + 1) for u, v in combinations(range(n), 2)
                  if rows[u] & (1 << v)])


def link_key(rows, root):
    return canonical(induced(rows, adjacent(rows, root)))


def ball_key(rows, root, radius):
    ds = distances(rows, root)
    order = [root] + sorted(v for v in ds if v != root and ds[v] <= radius)
    return rooted_key(induced(rows, order))


def histogram(rows, radius=1):
    return Counter(ball_key(rows, root, radius) for root in range(len(rows)))


def catalog(max_vertices, connected=False):
    keys = set()
    for n in range(0 if not connected else 1, max_vertices + 1):
        edges = list(combinations(range(n), 2))
        for mask in range(1 << len(edges)):
            rows = graph(n, (e for i, e in enumerate(edges) if mask & (1 << i)))
            if connected and len(distances(rows, 0)) != n:
                continue
            keys.add(canonical(rows))
    return sorted(keys)


@lru_cache(None)
def realize(link):
    """Triangular construction; the verifier recomputes rooted balls separately."""
    h = decode(link)
    g, n = cone(h), len(h)
    mult = 1 + sum(row.bit_count() == n - 1 for row in h)
    result = defaultdict(Q, {canonical(g): Q(1, mult)})
    for v, row in enumerate(g):
        if row.bit_count() < n:
            for host, coefficient in realize(link_key(g, v)).items():
                result[host] -= coefficient / mult
    return {host: coefficient for host, coefficient in result.items() if coefficient}


def rank(matrix):
    a = [[Q(value) for value in row] for row in matrix]
    pivot_row = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(pivot_row, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[pivot_row], a[pivot] = a[pivot], a[pivot_row]
        scale = a[pivot_row][col]
        a[pivot_row] = [x / scale for x in a[pivot_row]]
        for i in range(pivot_row + 1, len(a)):
            scale = a[i][col]
            a[i] = [x - scale * y for x, y in zip(a[i], a[pivot_row])]
        pivot_row += 1
        if pivot_row == len(a):
            break
    return pivot_row


def cartesian(a, b):
    n, m = len(a), len(b)
    return graph(n * m, [(u * m + v, x * m + v)
                        for u, x in combinations(range(n), 2) if a[u] & (1 << x)
                        for v in range(m)] +
                 [(u * m + v, u * m + y) for u in range(n)
                  for v, y in combinations(range(m), 2) if b[v] & (1 << y)])


def disjoint(a, b):
    return tuple(a) + tuple(row << len(a) for row in b)


def radius_one():
    links = catalog(4)
    targets = [rooted_key(cone(decode(key))) for key in links]
    witnesses = []
    for key, target in zip(links, targets):
        coefficients = realize(key)
        observed = defaultdict(Q)
        for host, coefficient in coefficients.items():
            require(host[0] <= key[0] + 1, "realization exceeds size bound")
            for ball, multiplicity in histogram(decode(host)).items():
                observed[ball] += coefficient * multiplicity
        observed = {ball: c for ball, c in observed.items() if c}
        require(observed == {target: Q(1)}, f"realization failed for {key}")
        witnesses.append({"link_key": key, "target_rooted_ball_key": target,
                          "host_coefficients": [{"graph_key": h, "coefficient": str(c)}
                                                for h, c in sorted(coefficients.items())]})
    hosts = catalog(5, connected=True)
    columns = [histogram(decode(key)) for key in hosts]
    require(set().union(*(set(c) for c in columns)) == set(targets), "type catalog mismatch")
    matrix = [[column.get(target, 0) for column in columns] for target in targets]
    matrix_rank = rank(matrix)
    require(matrix_rank == len(targets), "radius-one host span has a deficit")
    diagonal = [1 + sum(row.bit_count() == key[0] - 1 for row in decode(key))
                for key in links]
    # Recompute the entire cone histogram matrix, including all off-diagonal entries.
    cone_columns = [histogram(cone(decode(key))) for key in links]
    for j, column in enumerate(cone_columns):
        require(column[targets[j]] == diagonal[j], "wrong cone diagonal")
        require(all(column.get(targets[i], 0) == 0 for i in range(j + 1, len(targets))),
                "cone matrix is not triangular")
    small_links = catalog(3)
    for left in small_links:
        for right in small_links:
            a, b = decode(left), decode(right)
            actual = ball_key(cartesian(cone(a), cone(b)), 0, 1)
            expected = rooted_key(cone(disjoint(a, b)))
            require(actual == expected, f"product-link rule failed: {left}, {right}")
    return {"link_max_vertices": 4, "connected_host_max_vertices": 5,
            "link_types": len(links), "connected_host_types": len(hosts),
            "histogram_matrix_rank": matrix_rank,
            "cone_matrix_diagonal": diagonal, "cone_matrix_determinant": prod(diagonal),
            "product_scope": "Every ordered pair of link types through three vertices; full Cartesian graph then BFS ball",
            "realization_witnesses": witnesses}


def positive_span_fixture():
    def cycle(n):
        return graph(n, [(v, (v + 1) % n) for v in range(n)])
    local, global_measure = defaultdict(Q), {}
    for n, sign in ((8, -1), (16, 1)):
        # The distinct vertex counts separate the full graph supports. Within
        # each cycle all roots are isomorphic by rotation.
        global_measure[n] = sum((Q(sign, n) for _ in range(n)), Q(0))
        for ball, count in histogram(cycle(n), 2).items():
            local[ball] += Q(sign * count, n)
    require(all(c == 0 for c in local.values()), "normalized cycles did not cancel locally")
    # Vertex count distinguishes the full rooted graph supports; each normalized
    # positive law has mass one, even though their radius-two pushforwards agree.
    return {"graphs": ["C8/8", "C16/16"], "difference": "C16/16-C8/8",
            "radius": 2, "local_variation": str(sum(map(abs, local.values()), Q(0))),
            "global_measure_by_vertex_count": {n: str(c) for n, c in global_measure.items()},
            "global_jordan_variation": str(sum(map(abs, global_measure.values()), Q(0))),
            "global_support_separation": "Different total vertex counts",
            "scope": "A finite illustration of cancellation under pushforward, not a proof of the measure characterization"}


def neighbors(point, deleted):
    for axis in range(len(point)):
        for sign in (-1, 1):
            other = tuple(x + (sign if i == axis else 0) for i, x in enumerate(point))
            if frozenset((point, other)) not in deleted:
                yield other


def lattice_ball(starts, radius, deleted=frozenset()):
    ds = dict.fromkeys(starts, 0)
    queue = deque(ds)
    while queue:
        u = queue.popleft()
        if ds[u] == radius:
            continue
        for v in neighbors(u, deleted):
            if v not in ds:
                ds[v] = ds[u] + 1
                queue.append(v)
    return ds


def edge_set(vertices, deleted):
    return {frozenset((u, v)) for u in vertices for v in neighbors(u, deleted)
            if v in vertices}


def deletion_fixture(name, edges, radii, expected_regular, expected_count=None):
    deleted = frozenset(frozenset(edge) for edge in edges)
    endpoints = set().union(*deleted)
    dimension = len(next(iter(endpoints)))
    n_regular = sum(expected_regular.values())
    require(n_regular <= len(endpoints) <= 2 * len(deleted), "bad finite-component bound")
    rows = []
    for radius in radii:
        predicted = set(lattice_ball(endpoints, radius - 1))
        # One extra layer tests the boundary of the affected-root region.
        candidates = lattice_ball(endpoints, radius)
        affected, regular = set(), Counter()
        for root in candidates:
            before = lattice_ball([root], radius)
            after = lattice_ball([root], radius, deleted)
            changed = (before.keys() != after.keys()
                       or edge_set(before, frozenset()) != edge_set(after, deleted))
            if not changed:
                continue
            affected.add(root)
            degrees = {u: sum(v in after for v in neighbors(u, deleted)) for u in after}
            interior = [u for u in after if after[u] < radius]
            # This invariant certifies a change of rooted isomorphism type,
            # not merely a change of the coordinate labels.
            require(any(degrees[u] != 2 * dimension for u in interior),
                    "changed ball lacks the interior deficiency witness")
            if all(degrees[u] == degrees[root] for u in interior):
                regular[degrees[root]] += 1
        require(affected == predicted, f"affected roots wrong for {name}, r={radius}")
        if expected_count:
            require(len(affected) == expected_count(radius), "affected-count closed form failed")
        classification_applies = radius >= len(endpoints) + 1
        if classification_applies:
            require(dict(regular) == expected_regular,
                    f"regular-component classification failed for {name}, r={radius}")
        rows.append({"radius": radius, "affected_roots": len(affected),
                     "positive_regular_mass_by_degree": dict(sorted(regular.items())),
                     "uniform_classification_threshold_met": classification_applies,
                     "certified_character_disc_radius": len(affected) - n_regular
                         if classification_applies else None,
                     "sufficient_finite_buffer_distance_from_endpoints": 2 * radius + 1})
    return {"name": name, "deleted_edges": [list(e) for e in edges],
            "endpoint_count": len(endpoints), "regular_component_vertex_mass": n_regular,
            "expected_eventual_q_coefficients": expected_regular, "observations": rows}


def deletions():
    line = lambda a, b: ((a,), (b,))
    square = {(0, 0), (0, 1), (1, 0), (1, 1)}
    square_boundary = sorted({tuple(sorted((u, v))) for u in square
                              for v in neighbors(u, frozenset()) if v not in square})
    isolated_point = [((0, 0), v) for v in neighbors((0, 0), frozenset())]
    return [
        deletion_fixture("single_line_cut", [line(0, 1)], [1, 3, 4], {}, lambda r: 2*r),
        deletion_fixture("isolated_line_vertex", [line(-1, 0), line(0, 1)],
                         [1, 4, 5], {0: 1}, lambda r: 2*r+1),
        deletion_fixture("isolated_line_edge", [line(-1, 0), line(1, 2)],
                         [1, 5, 6], {1: 2}, lambda r: 2*r+2),
        deletion_fixture("isolated_nonregular_path", [line(-1, 0), line(2, 3)],
                         [1, 5, 6], {}),
        deletion_fixture("line_vertex_and_edge_components", [line(-1, 0), line(0, 1), line(2, 3)],
                         [1, 6, 7], {0: 1, 1: 2}),
        deletion_fixture("single_lattice_cut", [((0, 0), (1, 0))],
                         [1, 3, 4], {}, lambda r: 2*r*r),
        deletion_fixture("isolated_lattice_vertex", isolated_point,
                         [1, 6, 7], {0: 1}, lambda r: 1+2*r*(r+1)),
        deletion_fixture("isolated_lattice_square", square_boundary,
                         [1, 3, 13], {2: 4}),
    ]


def verify():
    return {"scope": "Finite constructive fixtures for three written proofs; no assertion-count coverage metric",
            "radius_one": radius_one(), "positive_span": positive_span_fixture(),
            "finite_deletions": deletions()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("foundation_checkpoint_results.json"))
    args = parser.parse_args()
    result = verify()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Verified foundation fixtures; witnesses written to {args.output}")
