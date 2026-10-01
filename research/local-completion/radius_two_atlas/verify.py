"""Independent finite oracles for the radius-two atlas (standard library).

Catalogue coverage through six vertices uses exhaustive labelled graphs and
degree-preserving permutations, with no individualization/refinement or
twin pruning. Products use full Cartesian graphs followed by BFS. Injection
counts use direct permutations instead of the analyzer's search algorithm.
Larger catalogue completeness rests on the documented enumeration argument;
this file does not turn finite examples into a proof of general freeness.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations, permutations, product
import json
from pathlib import Path

from analyze import (ball, cartesian, coordinates, decode, distances, graph,
                     induced, load, neighbors, require, rooted_injections,
                     reconstruct_supported, exact_small_rank)


@lru_cache(None)
def permutation_key(rows, rooted=False):
    groups = defaultdict(list)
    for u in range(int(rooted), len(rows)):
        groups[rows[u].bit_count()].append(u)
    cells = [groups[d] for d in sorted(groups)]
    masks = []
    for choices in product(*(permutations(cell) for cell in cells)):
        order = ((0,) if rooted else ()) + tuple(v for choice in choices for v in choice)
        masks.append(sum(1 << k for k, (i, j) in enumerate(combinations(range(len(rows)), 2))
                         if rows[order[i]] & (1 << order[j])))
    return len(rows), min(masks)


def exact_rooted_isomorphism(a, b):
    if len(a) != len(b):
        return False
    da, db = tuple(map(int.bit_count, a)), tuple(map(int.bit_count, b))
    if da[0] != db[0] or sorted(da) != sorted(db):
        return False
    mapping, used = {0: 0}, {0}

    def visit():
        if len(mapping) == len(a):
            return True
        best = None
        for u in range(1, len(a)):
            if u in mapping:
                continue
            candidates = [v for v in range(1, len(b)) if v not in used and da[u] == db[v]
                          and all(bool(a[u] & (1 << x)) == bool(b[v] & (1 << y))
                                  for x, y in mapping.items())]
            if not candidates:
                return False
            if best is None or len(candidates) < len(best[1]):
                best = u, candidates
        u, candidates = best
        for v in candidates:
            mapping[u] = v
            used.add(v)
            if visit():
                return True
            used.remove(v)
            del mapping[u]
        return False

    return visit()


def exhaustive_small_catalogs(maximum=6):
    expected = {cap: {'hosts': set(), 'balls': set()} for cap in (3, 4)}
    for n in range(1, maximum + 1):
        print(f'Independent labelled enumeration n={n}', flush=True)
        for mask in range(1 << (n*(n-1)//2)):
            rows = decode(n, mask)
            cap = max(map(int.bit_count, rows))
            if cap > 4 or len(distances(rows)) != n:
                continue
            host = permutation_key(rows)
            roots = {permutation_key(induced(rows, [u] + [v for v in range(n) if v != u]), True)
                     for u in range(n) if max(distances(rows, u).values()) <= 2}
            for bound in (3, 4):
                if cap <= bound:
                    expected[bound]['hosts'].add(host)
                    expected[bound]['balls'].update(roots)
    return expected


def direct_injections(source, target):
    if len(source) > len(target):
        return 0
    edges = [(u, v) for u, v in combinations(range(len(source)), 2) if source[u] & (1 << v)]
    return sum(all(target[p[u]] & (1 << p[v]) for u, v in edges)
               for tail in permutations(range(1, len(target)), len(source)-1)
               for p in [(0,) + tail])


def radius_three_control():
    # Connected graphs of maximum degree two are paths or cycles. The
    # radius-three Moore bound is seven, so this list is complete.
    hosts = [graph(n, [(i,i+1) for i in range(n-1)]) for n in range(1,8)]
    hosts += [graph(n, [(i,(i+1)%n) for i in range(n)]) for n in range(3,8)]
    histograms = [Counter(permutation_key(ball(g,u,3),True) for u in range(len(g))) for g in hosts]
    keys = sorted(set().union(*map(set,histograms)))
    ids = {key: i for i, key in enumerate(keys)}
    hcols = [{ids[k]: v for k,v in hist.items()} for hist in histograms]
    pairs = []
    for g, hist in zip(hosts,histograms):
        central = sorted(k for k in hist if k[0] == len(g))
        pairs.extend((central[0],k) for k in central[1:])
    rrows = [{j: value for j,k in enumerate(keys)
              if (value := rooted_injections(decode(*a),decode(*k))
                  - rooted_injections(decode(*b),decode(*k)))} for a,b in pairs]
    require(all(sum(row.get(j,0)*v for j,v in h.items()) == 0 for row in rrows for h in hcols),
            'radius-three transports fail')
    hr, dr = exact_small_rank(hcols), exact_small_rank(rrows)
    require((len(keys),hr,dr) == (15,12,3), 'radius-three degree-two ranks differ')
    return {'radius': 3, 'maximum_degree': 2, 'maximum_vertices': 7,
            'rooted_keys': keys, 'host_histograms': hcols,
            'rooted_injection_constraint_pairs': pairs,
            'exact_host_rank': hr, 'exact_constraint_rank': dr,
            'scope': 'Complete degree-two radius-three slice, using path/cycle classification.'}


def verify_directory(directory, expected):
    entries, balls, hosts, products, histograms = load(directory)
    limit, cap = max(map(len, balls)), max(max(map(int.bit_count, r)) for r in balls)
    small_hosts = [permutation_key(decode(n, mask)) for _, n, mask in hosts if n <= 6]
    small_balls = [permutation_key(rows, True) for rows in balls if len(rows) <= 6]
    require(len(small_hosts) == len(set(small_hosts)), 'duplicate small host isomorphism class')
    require(len(small_balls) == len(set(small_balls)), 'duplicate small rooted isomorphism class')
    require(set(small_hosts) == expected[cap]['hosts'], 'small host catalogue incomplete')
    require(set(small_balls) == expected[cap]['balls'], 'small rooted catalogue incomplete')
    recorded = {(a, b): c for a, b, c in products}
    require(len(recorded) == len(products), 'duplicate product pair')
    computed = set()
    for a in range(1, len(balls)):
        for b in range(a, len(balls)):
            # Axes alone give this necessary size bound; the mixed layer is
            # constructed explicitly instead of using the size formula.
            if len(balls[a]) + len(balls[b]) - 1 > limit:
                break
            if balls[a][0].bit_count() + balls[b][0].bit_count() > cap:
                continue
            whole = cartesian(balls[a], balls[b])
            local = ball(whole, 0)
            if len(local) > limit or max(map(int.bit_count, local)) > cap:
                continue
            require((a, b) in recorded, f'missing product {a},{b}')
            require(exact_rooted_isomorphism(local, balls[recorded[a, b]]), 'incorrect product type')
            computed.add((a, b))
    require(computed == set(recorded), 'extra or omitted product pairs')
    # Every degree-three host is checked. Degree-four checking includes all
    # hosts through seven vertices and a deterministic sample of larger ones.
    checked_hosts = []
    for (hid, n, mask), histogram in zip(hosts, histograms):
        if cap == 4 and n > 7 and hid % 97:
            continue
        rows = decode(n, mask)
        observed = Counter()
        for root in range(n):
            local = ball(rows, root)
            matches = [i for i in histogram if exact_rooted_isomorphism(local, balls[i])]
            require(len(matches) == 1, 'histogram has a missing or duplicate rooted type')
            observed[matches[0]] += 1
        require(observed == histogram, 'incorrect rooted histogram multiplicities')
        checked_hosts.append(hid)
    for source in balls:
        if len(source) > 4:
            continue
        for target in balls:
            if len(target) > 5:
                continue
            require(rooted_injections(source, target) == direct_injections(source, target),
                    'rooted injection count differs from exhaustive permutations')
    # Larger-radius hosts exercise the realization map beyond its basis
    # columns. All coefficients and reconstruction residuals are exact.
    realization_witnesses = []
    for hid in checked_hosts:
        if hosts[hid][1] > 7:
            continue
        coefficients, residue = reconstruct_supported(balls, hosts, histograms, histograms[hid])
        require(not residue, 'a finite host histogram failed exact reconstruction')
        if len(coefficients) > 1 and len(realization_witnesses) < 3:
            realization_witnesses.append({'host_id': hid,
                                         'basis_coefficients': {i: str(c) for i, c in coefficients.items()}})
    p3_endpoint = graph(3, [(0,1),(1,2)])
    endpoint_id = next(i for i, rows in enumerate(balls) if exact_rooted_isomorphism(rows, p3_endpoint))
    _, obstruction = reconstruct_supported(balls, hosts, histograms, {endpoint_id: 1})
    require(bool(obstruction), 'unbalanced rooted path endpoint was accepted')
    return {'degree_cap': cap, 'vertex_limit': limit,
            'exhaustive_permutation_catalog_scope': {'maximum_vertices': 6,
                                                   'host_types': len(small_hosts), 'rooted_types': len(small_balls)},
            'full_cartesian_product_oracle': 'Every admissible unordered factor pair, including detection of missing entries.',
            'histogram_oracle_host_ids': checked_hosts,
            'injection_oracle_scope': 'Every source through four vertices and target through five in this catalogue.',
            'finite_reconstruction_scope': 'Every independently checked host through seven vertices; single P3 endpoint is rejected.',
            'finite_reconstruction_witnesses': realization_witnesses,
            'result': 'passed'}


def main():
    base = Path(__file__).parent
    expected = exhaustive_small_catalogs()
    reports = []
    for name in ('degree3_n10', 'degree4_n9'):
        print(f'Checking products and histograms in {name}', flush=True)
        reports.append(verify_directory(base/name, expected))
    # A literal geometry witness: both roots have degree two and two second-
    # sphere vertices, but neighbor degree profiles (2,2) and (1,3).
    straight = graph(5, [(0,1),(0,2),(1,3),(2,4)])
    branched = graph(5, [(0,1),(0,2),(2,3),(2,4)])
    require(coordinates(straight) == coordinates(branched), 'old-coordinate collision lost')
    require(coordinates(straight, True) != coordinates(branched, True), 'decoration did not separate witness')
    cycle6 = graph(6, [(i,(i+1)%6) for i in range(6)])
    path5 = graph(5, [(i,i+1) for i in range(4)])
    path4 = graph(4, [(i,i+1) for i in range(3)])
    hs = [Counter(permutation_key(ball(rows,u),True) for u in range(len(rows)))
          for rows in (cycle6,path5,path4)]
    keys = set().union(*map(set,hs))
    require(all(hs[0].get(k,0) == 6*(hs[1].get(k,0)-hs[2].get(k,0)) for k in keys),
            'literal cycle/path finite realization identity failed')
    # Higher products exercise the new coordinates beyond the stored degree
    # and size caps, while still using independently materialized graphs.
    fixtures = [straight, branched, graph(3, [(0,1),(1,2),(0,2)]), graph(2, [(0,1)])]
    for a in fixtures:
        for b in fixtures:
            expected_values = Counter(coordinates(a, True))
            expected_values.update(coordinates(b, True))
            expected_values = {k: v for k, v in expected_values.items() if v}
            require(coordinates(ball(cartesian(a,b),0), True) == expected_values,
                    'decorated coordinates fail outside the atlas caps')
    output = {'catalogues': reports,
              'radius_three_degree_two_control': radius_three_control(),
              'coordinate_witness': {'graphs': 'center-rooted P5; rooted tree with neighbor degrees 1 and 3',
                                     'familiar_coordinates_equal': True, 'decorated_coordinates_different': True},
              'finite_realization_identity': 'T_2(C6/6) = T_2(P5-P4)',
              'larger_product_fixture_scope': 'All ordered pairs of the two five-vertex trees, the triangle, and the edge.',
              'limits': 'Finite computational verification. General proofs and full-catalogue enumeration correctness await the separately deferred review.'}
    (base/'verification.json').write_text(json.dumps(output, indent=2, sort_keys=True)+'\n')
    print('Independent atlas verification passed; witnesses recorded in verification.json')


if __name__ == '__main__':
    main()
