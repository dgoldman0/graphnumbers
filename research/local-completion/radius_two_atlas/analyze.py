"""Exact finite analysis of atlas.cpp output; standard library only.

Ranks modulo a prime are lower bounds over Q. A matching primal/dual
dimension bound certifies equality over Q, R and C; unmatched bounds remain
bounds. All graph coordinates and transport matrices use integers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations
import hashlib
import json
from pathlib import Path

PRIME = 1000003


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_table(path):
    return [line.split() for line in path.read_text().splitlines()
            if line and not line.startswith('#')]


def decode(n, mask):
    rows = [0] * n
    for k, (u, v) in enumerate(combinations(range(n), 2)):
        if mask & (1 << k):
            rows[u] |= 1 << v
            rows[v] |= 1 << u
    return tuple(rows)


def graph(n, edges):
    rows = [0] * n
    for u, v in edges:
        rows[u] |= 1 << v
        rows[v] |= 1 << u
    return tuple(rows)


def neighbors(rows, u):
    return [v for v in range(len(rows)) if rows[u] & (1 << v)]


def distances(rows, root=0):
    d, q = {root: 0}, deque([root])
    while q:
        u = q.popleft()
        for v in neighbors(rows, u):
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


def induced(rows, order):
    return graph(len(order), ((i, j) for i, j in combinations(range(len(order)), 2)
                             if rows[order[i]] & (1 << order[j])))


def ball(rows, root, radius=2):
    d = distances(rows, root)
    return induced(rows, [root] + sorted(v for v in d if v != root and d[v] <= radius))


def cartesian(a, b):
    n, m = len(a), len(b)
    return graph(n * m,
                 [(u * m + v, x * m + v) for u, x in combinations(range(n), 2)
                  if a[u] & (1 << x) for v in range(m)] +
                 [(u * m + v, u * m + y) for v, y in combinations(range(m), 2)
                  if b[v] & (1 << y) for u in range(n)])


@lru_cache(None)
def brute_key(rows, fixed=0):
    """Full permutation oracle, used only for small links/edge neighborhoods."""
    n = len(rows)
    return n, min(sum(1 << k for k, (i, j) in enumerate(combinations(range(n), 2))
                      if rows[p[i]] & (1 << p[j]))
                  for tail in permutations(range(fixed, n))
                  for p in [tuple(range(fixed)) + tail])


def components(rows):
    unseen = set(range(len(rows)))
    result = []
    while unseen:
        component = sorted(distances(rows, min(unseen)))
        unseen.difference_update(component)
        result.append(component)
    return result


def component_counts(rows):
    return Counter(brute_key(induced(rows, c)) for c in components(rows))


@lru_cache(None)
def decorated_components(rows):
    """Vertex labels are neighbor degree minus root degree.

    Pair labels are (adjacent, common-neighbor count minus two).

    Distinct product-coordinate neighbors have the zero label. Therefore
    connected components using nonzero labels give additive coordinates.
    """
    vertices = neighbors(rows, 0)
    n = len(vertices)
    vertex_labels = [rows[v].bit_count() - n for v in vertices]
    labels = [[(0, 0)] * n for _ in range(n)]
    support_edges = []
    for i, j in combinations(range(n), 2):
        u, v = vertices[i], vertices[j]
        label = (int(bool(rows[u] & (1 << v))), (rows[u] & rows[v]).bit_count() - 2)
        labels[i][j] = labels[j][i] = label
        if label != (0, 0):
            support_edges.append((i, j))
    support = graph(n, support_edges)
    counts = Counter()
    for c in components(support):
        key = len(c), min((tuple(vertex_labels[v] for v in p),
                           tuple(labels[p[i]][p[j]] for i, j in combinations(range(len(c)), 2)))
                          for p in permutations(c))
        counts[key] += 1
    return counts


def walk_moments(rows, length):
    counts = [int(i == 0) for i in range(len(rows))]
    closed, opened = [1], [1]
    for _ in range(length):
        counts = [sum(counts[v] for v in neighbors(rows, u)) for u in range(len(rows))]
        closed.append(counts[0])
        opened.append(sum(counts))
    return closed, opened


def coordinates(rows, decorated=False):
    adj = neighbors(rows, 0)
    links = component_counts(induced(rows, adj))
    gamma = component_counts(graph(len(adj), ((i, j) for i, j in combinations(range(len(adj)), 2)
                                               if (rows[adj[i]] & rows[adj[j]]).bit_count() == 1)))
    d = len(adj)
    c, o = walk_moments(rows, 5)
    values = {'degree': d, 'sphere_log_2_scaled': 2 * (len(rows) - 1 - d) - d*d,
              'open_cumulant_2': o[2] - d*d, 'closed_cumulant_3': c[3],
              'closed_cumulant_4': c[4] - 3*d*d,
              'closed_cumulant_5': c[5] - 10*d*c[3]}
    for name, counts in [('link', links), ('root_edge', gamma)]:
        for key, value in counts.items():
            values[f'{name}:{key[0]}:{key[1]}'] = value
    if decorated:
        for key, value in decorated_components(rows).items():
            values['decorated:' + json.dumps(key, separators=(',', ':'))] = value
    return {k: v for k, v in values.items() if v}


def bipartite(rows):
    color = {}
    for start in range(len(rows)):
        if start in color:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in neighbors(rows, u):
                if v in color:
                    if color[v] == color[u]:
                        return False
                else:
                    color[v] = 1 - color[u]
                    q.append(v)
    return True


def face_values(rows):
    d = distances(rows)
    return {'interior_regular': all(rows[u].bit_count() == rows[0].bit_count()
                                    for u in d if d[u] < 2),
            'bipartite': bipartite(rows),
            'triangle_free': all(not (rows[u] & rows[v])
                                 for u, v in combinations(range(len(rows)), 2)
                                 if rows[u] & (1 << v))}


def modular_rank(vectors, prime=PRIME):
    """Sparse column elimination; pivot input indices identify an independent set."""
    basis, pivots = {}, []
    for i, source in enumerate(vectors):
        v = {k: value % prime for k, value in source.items() if value % prime}
        while v:
            k = min(v)
            if k not in basis:
                inverse = pow(v[k], -1, prime)
                basis[k] = {j: (value * inverse) % prime for j, value in v.items()}
                pivots.append(i)
                break
            scale = v[k]
            for j, value in basis[k].items():
                updated = (v.get(j, 0) - scale * value) % prime
                if updated:
                    v[j] = updated
                else:
                    v.pop(j, None)
    return len(basis), pivots


def exact_small_rank(vectors):
    basis = {}
    for source in vectors:
        v = {k: Fraction(x) for k, x in source.items() if x}
        while v:
            k = min(v)
            if k not in basis:
                lead = v[k]
                basis[k] = {j: x / lead for j, x in v.items()}
                break
            scale = v[k]
            for j, x in basis[k].items():
                y = v.get(j, 0) - scale*x
                if y:
                    v[j] = y
                else:
                    v.pop(j, None)
    return len(basis)


def transport_column(rows):
    """Antisymmetric edge-neighborhood divergences, determined by B_2.

    The directed-edge signature is the induced union N[u] union N[v]
    including endpoints, with endpoints ordered first. Rooted orientation
    and its reversal get opposite signs in the same matrix row.
    """
    result = Counter()
    for v in neighbors(rows, 0):
        union = {0, v} | set(neighbors(rows, 0)) | set(neighbors(rows, v))
        rest = sorted(union - {0, v})
        forward = brute_key(induced(rows, [0, v] + rest), 2)
        reverse = brute_key(induced(rows, [v, 0] + rest), 2)
        if forward != reverse:
            result[min(forward, reverse)] += 1 if forward < reverse else -1
    return {k: v for k, v in result.items() if v}


@lru_cache(None)
def rooted_injections(source, target):
    """All injective edge-preserving maps taking source vertex zero to zero."""
    n, m = len(source), len(target)
    sd, td = tuple(map(int.bit_count, source)), tuple(map(int.bit_count, target))
    if n > m or sum(sd) > sum(td) or sd[0] > td[0]:
        return 0
    mapping = {0: 0}
    used = 1

    def visit(used):
        if len(mapping) == n:
            return 1
        best = None
        for u in range(1, n):
            if u in mapping:
                continue
            candidates = [v for v in range(1, m) if not used & (1 << v)
                          and sd[u] <= td[v]
                          and all(not source[u] & (1 << x) or target[v] & (1 << y)
                                  for x, y in mapping.items())]
            if not candidates:
                return 0
            # Fewest candidates, then most already-mapped neighbors.
            choice = (len(candidates), -sum(bool(source[u] & (1 << x)) for x in mapping), u, candidates)
            if best is None or choice[:3] < best[:3]:
                best = choice
        _, _, u, candidates = best
        total = 0
        for v in candidates:
            mapping[u] = v
            total += visit(used | (1 << v))
            del mapping[u]
        return total

    return visit(used)


def load(directory):
    entries = [tuple(map(int, row)) for row in read_table(directory / 'balls.tsv')]
    require([r[0] for r in entries] == list(range(len(entries))), 'nonconsecutive ball IDs')
    balls = [decode(n, m) for _, n, m in entries]
    hosts = [tuple(map(int, row)) for row in read_table(directory / 'hosts.tsv')]
    products = [tuple(map(int, row)) for row in read_table(directory / 'products.tsv')]
    histograms = []
    for row in read_table(directory / 'histograms.tsv'):
        require(int(row[0]) == len(histograms), 'nonconsecutive host histogram IDs')
        histograms.append(dict(tuple(map(int, term.split(':'))) for term in row[1:]))
    return entries, balls, hosts, products, histograms


def reconstruct_supported(balls, hosts, histograms, values):
    """Triangular finite realization, returning exact coefficients and residue.

    Zero residue is equivalent to membership in the supported balanced image
    by the finite-slice theorem. Nonzero residue is an obstruction, not an
    approximate realization. Values are unnormalized local counting masses.
    """
    require(all(isinstance(i, int) and 0 <= i < len(balls) for i in values),
            'array contains an unknown ball ID')
    residue = {i: Fraction(value) for i, value in values.items() if value}
    coefficients = {}
    for (hid, n, _), histogram in reversed(list(zip(hosts, histograms))):
        central = sorted(i for i in histogram if len(balls[i]) == n)
        if not central:
            continue
        coefficient = residue.get(central[0], 0) / Fraction(histogram[central[0]])
        if not coefficient:
            continue
        coefficients[hid] = coefficient
        for i, count in histogram.items():
            value = residue.get(i, 0) - coefficient * count
            if value:
                residue[i] = value
            else:
                residue.pop(i, None)
    return coefficients, residue


def analyze_balance(balls, hosts, histograms, cap, max_ball_vertices, max_host_vertices,
                    execute_injections=False):
    selected = [i for i, rows in enumerate(balls)
                if len(rows) <= max_ball_vertices and max(map(int.bit_count, rows)) <= cap]
    positions = {i: j for j, i in enumerate(selected)}
    transports = [transport_column(balls[i]) for i in selected]
    row_keys = sorted(set().union(*(set(t) for t in transports)))
    row_ids = {k: i for i, k in enumerate(row_keys)}
    dcols = [{row_ids[k]: v for k, v in t.items()} for t in transports]
    hcols, host_ids = [], []
    for host, histogram in zip(hosts, histograms):
        hid, n, mask = host
        if n > max_host_vertices or not set(histogram) <= set(positions):
            continue
        # A full histogram is used. Projecting away unsupported balls would
        # destroy balance and would not provide a legitimate lower bound.
        h = {positions[i]: value for i, value in histogram.items()}
        observed = Counter()
        for j, value in h.items():
            for k, coefficient in dcols[j].items():
                observed[k] += value * coefficient
        require(all(v == 0 for v in observed.values()), f'transport fails on host {hid}')
        hcols.append(h)
        host_ids.append(hid)
    dr, dp = modular_rank(dcols)
    hr, hp = modular_rank(hcols)
    require(hr + dr <= len(selected), 'primal/dual ranks contradict exact annihilation')
    saturated = hr + dr == len(selected)
    center_groups = []
    for host, histogram in zip(hosts, histograms):
        hid, n, _ = host
        central = [i for i in histogram if len(balls[i]) == n]
        if n <= max_ball_vertices and central and set(histogram) <= set(positions):
            center_groups.append((hid, sorted(central)))
    require(set().union(*(set(g) for _, g in center_groups)) == set(selected),
            'center groups do not partition the ball slice')
    reroot_pairs = [(ids[0], other) for _, ids in center_groups for other in ids[1:]]
    # The square rooted-injection matrix is triangular by (vertices, edges),
    # with rooted automorphisms on its diagonal. Differences of its rows
    # within a center group are therefore independent. See RADIUS_TWO_ATLAS.
    universal_dimension = len(center_groups)
    require(hr <= universal_dimension, 'host rank exceeds the proved slice dimension')
    injection_check = None
    if execute_injections:
        print(f'  Executing {len(reroot_pairs)} rooted-injection transport rows', flush=True)
        rrows = []
        for a, b in reroot_pairs:
            rrows.append({j: value for j, i in enumerate(selected)
                          if (value := rooted_injections(balls[a], balls[i])
                              - rooted_injections(balls[b], balls[i]))})
        for h in hcols:
            require(all(sum(row.get(j, 0) * value for j, value in h.items()) == 0
                        for row in rrows), 'rooted-injection transport fails on a host')
        rr, rp = modular_rank(rrows)
        require(rr == len(reroot_pairs), 'rooted-injection constraints lost rank modulo the chosen prime')
        require(hr + rr == len(selected), 'rooted-injection upper and host lower bounds do not meet')
        injection_check = {'reroot_row_rank_mod_prime': rr,
                           'host_rank_mod_prime': hr,
                           'exact_annihilation_DH_zero': True,
                           'bounds_meet': True,
                           'independent_row_indices': rp}
    return {'maximum_degree': cap, 'maximum_ball_vertices': max_ball_vertices,
            'maximum_host_vertices': max_host_vertices, 'ball_types': len(selected),
            'host_columns': len(hcols), 'simple_edge_transport_rows': len(row_keys),
            'rank_prime': PRIME, 'host_rank_mod_prime': hr, 'transport_rank_mod_prime': dr,
            'simple_edge_dimension_lower_bound': hr,
            'simple_edge_dimension_upper_bound': len(selected) - dr,
            'simple_edge_bounds_meet': saturated,
            'simple_edge_scope': 'Induced unions of the endpoint one-neighborhoods give only a subset of all rerooting constraints.',
            'independent_host_ids': [host_ids[i] for i in hp],
            'independent_transport_column_ball_ids': [selected[i] for i in dp],
            'full_rerooting': {'exact_slice_dimension_by_triangular_theorem': universal_dimension,
                              'constraint_dimension': len(reroot_pairs),
                              'basis_host_ids': [hid for hid, _ in center_groups],
                              'rooted_injection_constraint_pairs': reroot_pairs,
                              'executed_matrix_verification': injection_check}}


def summarize_coordinates(balls, products, enriched):
    values = [coordinates(rows, decorated=enriched) for rows in balls]
    names = sorted(set().union(*(set(c) for c in values)))
    name_ids = {name: i for i, name in enumerate(names)}
    columns = [{name_ids[k]: value for k, value in c.items()} for c in values]
    for a, b, c in products:
        expected = Counter(values[a])
        expected.update(values[b])
        require(dict(sorted((k, v) for k, v in expected.items() if v)) == values[c],
                f'coordinate additivity fails on {a}*{b}={c}')
    # Exact rank uses a modest coordinate dimension, irrespective of atlas size.
    rank = exact_small_rank(columns)
    groups = defaultdict(list)
    for i, c in enumerate(values):
        groups[tuple(sorted(c.items()))].append(i)
    collisions = [ids for ids in groups.values() if len(ids) > 1]
    smallest = min((ids[:2] for ids in collisions), key=lambda ids: (max(len(balls[i]) for i in ids), ids), default=None)
    by_size = []
    for n in sorted(set(map(len, balls))):
        ids = [i for i, rows in enumerate(balls) if len(rows) <= n]
        signatures = {tuple(sorted(values[i].items())) for i in ids}
        by_size.append({'through_vertices': n, 'types': len(ids),
                        'distinct_coordinate_vectors': len(signatures),
                        'separates_slice': len(signatures) == len(ids)})
    return {'coordinate_names': names, 'exact_rational_rank': rank,
            'distinct_coordinate_vectors': len(groups), 'collision_classes': len(collisions),
            'largest_collision_class': max(map(len, collisions), default=0),
            'smallest_collision_ball_ids': smallest,
            'smallest_collision_vector': values[smallest[0]] if smallest else None,
            'separation_by_size': by_size}, values


def analyze(directory, balance_sizes):
    entries, balls, hosts, products, histograms = load(directory)
    factorizations = [{()}] + [set() for _ in balls[1:]]
    decompositions = defaultdict(list)
    for a, b, c in products:
        require(0 < a <= b < c, 'factors must precede product by strict vertex size')
        decompositions[c].append((a, b))
    for i in range(1, len(balls)):
        if i not in decompositions:
            factorizations[i] = {(i,)}
        else:
            for a, b in decompositions[i]:
                for left in factorizations[a]:
                    for right in factorizations[b]:
                        factorizations[i].add(tuple(sorted(left + right)))
    nonunique = [{'ball_id': i, 'prime_factorizations': sorted(map(list, f))}
                 for i, f in enumerate(factorizations) if len(f) > 1]
    connected_decoration = [i for i in range(1, len(balls))
                            if sum(decorated_components(balls[i]).values()) == 1]
    require(not set(connected_decoration) & set(decompositions),
            'connected decoration contradicts factorization')
    connected_decoration_set = set(connected_decoration)
    unresolved_by_decoration = [i for i in range(1, len(balls))
                               if i not in decompositions and i not in connected_decoration_set]
    faces = [face_values(rows) for rows in balls]
    for a, b, c in products:
        for name in faces[c]:
            require(faces[c][name] == (faces[a][name] and faces[b][name]),
                    f'face identity fails for {name} on {a}*{b}')
    print('Computing familiar coordinates', flush=True)
    basic, bv = summarize_coordinates(balls, products, False)
    print('Computing decorated-neighbor coordinates', flush=True)
    enriched, ev = summarize_coordinates(balls, products, True)
    old_groups = defaultdict(list)
    for i, values in enumerate(bv):
        old_groups[tuple(sorted(values.items()))].append(i)
    improvement = None
    for ids in old_groups.values():
        if len(ids) > 1:
            first = ids[0]
            second = next((j for j in ids[1:] if ev[j] != ev[first]), None)
            if second is not None:
                candidate = [first, second]
                if improvement is None or (max(map(lambda i: len(balls[i]), candidate)), candidate) < (max(len(balls[i]) for i in improvement), improvement):
                    improvement = candidate
    cap = max(max(map(int.bit_count, rows)) for rows in balls)
    balances = []
    for n in balance_sizes:
        print(f'Computing balance cap={cap}, ball/host vertices<={n}', flush=True)
        balances.append(analyze_balance(balls, hosts, histograms, cap, n, n,
                                        execute_injections=(cap == 3 and n == 10) or (cap == 4 and n <= 6)))
    full_center_groups = [(host[0], sorted(i for i in hist if len(balls[i]) == host[1]))
                          for host, hist in zip(hosts, histograms)]
    full_center_groups = [(hid, ids) for hid, ids in full_center_groups if ids]
    result = {'scope': {'radius': 2, 'maximum_vertices': max(map(len, balls)), 'maximum_degree': cap,
                        'catalog': 'Every rooted radius-at-most-two simple graph in the stated size/degree slice.',
                        'products': 'Every unordered pair of nonunit factors whose product is in the slice.',
                        'all_radius_two_types_at_degree_cap': max(map(len, balls)) == 1 + cap*cap},
              'catalog_counts': {'connected_hosts': len(hosts), 'rooted_balls': len(balls),
                                  'hosts_by_vertices': dict(sorted(Counter(h[1] for h in hosts).items())),
                                  'balls_by_vertices': dict(sorted(Counter(map(len, balls)).items()))},
              'factorization': {'nonunit_product_pairs': len(products),
                                'irreducible_types': len(balls)-1-len(decompositions),
                                'decomposable_types': len(decompositions),
                                'nonunique_prime_factorizations': nonunique,
                                'irreducibles_certified_by_connected_decoration': len(connected_decoration),
                                'irreducibles_with_disconnected_decoration': len(unresolved_by_decoration),
                                'smallest_irreducible_with_disconnected_decoration': unresolved_by_decoration[0] if unresolved_by_decoration else None,
                                'decomposable_factorizations': [{'ball_id': i, 'prime_factorizations': sorted(map(list, factorizations[i]))}
                                                               for i in sorted(decompositions)],
                                'scope': 'Factors embed as induced coordinate axes; every factor of a catalogued ball is smaller and in the same degree cap. This certifies irreducibility and complete factorizations of the catalogued balls only.'},
              'coordinates': {'familiar': basic, 'with_decorated_neighbors': enriched,
                              'smallest_collision_separated_by_decoration': improvement,
                              'new_separating_values': {i: ev[i] for i in improvement} if improvement else None},
              'faces': {'members': {name: sum(f[name] for f in faces) for name in faces[0]},
                        'scope': 'These are proved faces, tested on every retained product; no classification of all faces is claimed.'},
              'balance': balances,
              'full_catalog_balance_by_triangular_theorem': {
                  'supported_slice_dimension': len(full_center_groups),
                  'independent_rerooting_constraints': len(balls) - len(full_center_groups),
                  'basis_host_ids': [hid for hid, _ in full_center_groups],
                  'scope': 'Dimension follows from the written general triangular proof; full numerical matrices are executed only in the explicitly recorded smaller slices.'},
              'data_sha256': {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                              for name in ['balls.tsv', 'hosts.tsv', 'histograms.tsv', 'products.tsv']}}
    (directory/'results.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'directory': str(directory), 'balls': len(balls),
                      'nonunique': nonunique, 'coordinate_ranks': [basic['exact_rational_rank'], enriched['exact_rational_rank']],
                      'coordinate_vectors': [basic['distinct_coordinate_vectors'], enriched['distinct_coordinate_vectors']],
                      'balance_dimensions': [(b['ball_types'], b['full_rerooting']['exact_slice_dimension_by_triangular_theorem']) for b in balances]}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--balance-sizes', type=int, nargs='*', default=[])
    args = parser.parse_args()
    analyze(args.directory, args.balance_sizes)
