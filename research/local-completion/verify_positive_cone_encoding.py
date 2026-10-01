"""Exact finite mechanisms for STRICT_POSITIVE_CONES.md.

This does not construct a non-sofic law. All source actions here are finite
and sofic. The decoder receives only unlabelled graph adjacency and d.
No graphlocal or third-party modules are imported.
"""
from collections import Counter, deque
from fractions import Fraction as Q
from itertools import permutations, product
from math import factorial, lcm
from pathlib import Path
import argparse
import hashlib
import json
import random


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def graph(n, edges):
    g = [set() for _ in range(n)]
    for u, v in edges:
        require(0 <= u < n and 0 <= v < n and u != v, "invalid simple edge")
        require(v not in g[u], "duplicate simple edge")
        g[u].add(v)
        g[v].add(u)
    return tuple(frozenset(row) for row in g)


def edge_set(g):
    return {(u, v) for u, row in enumerate(g) for v in row if u < v}


def sizes(d):
    return 2*d*d+7*d+3, 2*d+4


def encode(action):
    d, n = len(action), len(action[0])
    require(d >= 1 and n >= 1, "nonempty action required")
    for permutation in action:
        require(sorted(permutation) == list(range(n)), "generator is not a permutation")
    g, owners, centers, tails, heads = [], [], [], {}, {}

    def vertex(owner):
        result = len(g)
        g.append(set())
        owners.append(owner)
        return result

    def join(u, v):
        require(u != v and v not in g[u], "encoding produced a nonsimple edge")
        g[u].add(v)
        g[v].add(u)

    for v in range(n):
        center = vertex(v)
        centers.append(center)
        for _ in range(2):
            join(center, vertex(v))
        for i in range(d):
            tail, head = vertex(v), vertex(v)
            tails[v, i], heads[v, i] = tail, head
            join(center, tail)
            join(center, head)
            for _ in range(2*i+3):
                join(tail, vertex(v))
            for _ in range(2*i+4):
                join(head, vertex(v))
    for i, permutation in enumerate(action):
        for v, w in enumerate(permutation):
            join(tails[v, i], heads[w, i])
    return (tuple(frozenset(row) for row in g), tuple(centers),
            tails, heads, tuple(owners))


def port_roles(g, d):
    degrees = [len(row) for row in g]
    result = {}
    for v, row in enumerate(g):
        leaves = sum(degrees[u] == 1 for u in row)
        if 3 <= leaves <= 2*d+2 and degrees[v] == leaves+2:
            result[v] = leaves
    return result


def decode_partial(g, d):
    """No encoding annotations or original vertex IDs enter this decoder."""
    roles = port_roles(g, d)
    centers, ports = set(), {}
    for v, row in enumerate(g):
        if len(row) != 2*d+2 or sum(len(g[u]) == 1 for u in row) != 2:
            continue
        nonleaves = [u for u in row if len(g[u]) != 1]
        if sorted(roles.get(u, -1) for u in nonleaves) == list(range(3, 2*d+3)):
            centers.add(v)
            ports[v] = {roles[u]: u for u in nonleaves}
    partial = [{} for _ in range(d)]
    for u in centers:
        for i in range(d):
            tail = ports[u][2*i+3]
            others = [v for v in g[tail] if v != u and len(g[v]) > 1]
            require(len(others) == 1, "accepted tail has wrong nonleaf degree")
            head = others[0]
            if roles.get(head) != 2*i+4:
                continue
            targets = [v for v in g[head] if v != tail and len(g[v]) > 1]
            require(len(targets) == 1, "accepted head has wrong nonleaf degree")
            w = targets[0]
            if w in centers:
                partial[i][u] = w
    for mapping in partial:
        require(len(set(mapping.values())) == len(mapping), "decoder lost injectivity")
    bad = set()
    for mapping in partial:
        bad.update(centers-set(mapping))
        bad.update(centers-set(mapping.values()))
    return centers, partial, bad


def repair(centers, partial):
    full = []
    for mapping in partial:
        outgoing = sorted(centers-set(mapping))
        incoming = sorted(centers-set(mapping.values()), reverse=True)
        require(len(outgoing) == len(incoming), "partial injection has unequal deficits")
        completed = dict(mapping)
        completed.update(zip(outgoing, incoming))
        require(set(completed) == centers == set(completed.values()), "repair is not a permutation")
        full.append(completed)
    return full


def relabel(g, labels):
    require(sorted(labels) == list(range(len(g))), "not a vertex relabeling")
    return graph(len(g), ((labels[u], labels[v]) for u, v in edge_set(g)))


def ball_vertices(g, root, radius):
    distances, queue = {root: 0}, deque([root])
    while queue:
        u = queue.popleft()
        if distances[u] < radius:
            for v in g[u]:
                if v not in distances:
                    distances[v] = distances[u]+1
                    queue.append(v)
    return set(distances)


def induced(g, vertices):
    order = sorted(vertices)
    index = {v: i for i, v in enumerate(order)}
    edges = [(index[u], index[v]) for u in order for v in g[u]
             if v in index and u < v]
    return graph(len(order), edges), order, index


def action_ball(centers, maps, root, radius):
    """Canonical induced labelled ball via ordered generator/inverse BFS."""
    require(root in centers, "action root is missing")
    inverses = [{v: u for u, v in m.items()} for m in maps]
    order, index, distances = [root], {root: 0}, {root: 0}
    for u in order:
        if distances[u] >= radius:
            continue
        for forward, backward in zip(maps, inverses):
            for v in (forward.get(u), backward.get(u)):
                if v is not None and v not in index:
                    index[v] = len(order)
                    order.append(v)
                    distances[v] = distances[u]+1
    rows = tuple(tuple(index.get(mapping.get(u), -1) for mapping in maps) for u in order)
    return tuple(distances[u] for u in order), rows


def action_histogram(centers, maps, radius):
    require(bool(centers), "empty action has no normalized law")
    counts = Counter(action_ball(centers, maps, c, radius) for c in centers)
    return {key: Q(value, len(centers)) for key, value in counts.items()}


def source_maps(action):
    return [dict(enumerate(p)) for p in action]


def check_exact_action(action, rng):
    d, n = len(action), len(action[0])
    g, original, _, _, owners = encode(action)
    b, cap = sizes(d)
    require(len(g) == b*n, "fiber cardinality")
    require(len(edge_set(g)) == (b-1+d)*n, "edge cardinality")
    require(max(map(len, g)) == cap, "maximum degree")
    require(Counter(owners) == {v: b for v in range(n)}, "fiber sizes vary")
    centers, partial, bad = decode_partial(g, d)
    require(centers == set(original) and not bad, "exact center decoding")
    for i, p in enumerate(action):
        require(partial[i] == {original[v]: original[p[v]] for v in range(n)},
                "exact generator decoding")
    for r in (0, 1, 2):
        require(action_histogram(centers, partial, r)
                == action_histogram(set(range(n)), source_maps(action), r),
                "rooted source law changed")
    labels = list(range(len(g)))
    rng.shuffle(labels)
    h = relabel(g, labels)
    other_centers, other_partial, other_bad = decode_partial(h, d)
    require(other_centers == {labels[c] for c in original} and not other_bad,
            "relabelled center recognition")
    for i, p in enumerate(action):
        require(other_partial[i] == {labels[original[v]]: labels[original[p[v]]]
                                    for v in range(n)}, "relabelled generator decoding")


def finite_catalogues():
    rng = random.Random(6102026)
    records = []
    for d, maximum in ((1, 4), (2, 4), (3, 3)):
        for n in range(1, maximum+1):
            count = loops = opposite = coincident = 0
            for action in product(list(permutations(range(n))), repeat=d):
                check_exact_action(action, rng)
                count += 1
                loops += any(p[v] == v for p in action for v in range(n))
                opposite += any(p[v] != v and p[p[v]] == v
                                for p in action for v in range(n))
                coincident += any(action[i][v] == action[j][v]
                                  for i in range(d) for j in range(i+1, d) for v in range(n))
            require(count == factorial(n)**d, "permutation-action enumeration")
            records.append({"generators": d, "source_vertices": n,
                            "enumerated_actions": count, "fiber_size": sizes(d)[0],
                            "maximum_degree": sizes(d)[1], "actions_with_loops": loops,
                            "actions_with_opposite_arcs": opposite,
                            "actions_with_coincident_label_endpoints": coincident,
                            "independent_vertex_relabeling_per_action": True})
    return records


def local_decoder_check(g, d, samples):
    centers, partial, _ = decode_partial(g, d)
    records = []
    for root in sorted(centers)[:samples]:
        for r in (0, 1, 2):
            R = 3*r+6
            neighborhood, order, index = induced(g, ball_vertices(g, root, R))
            local_centers, local_maps, _ = decode_partial(neighborhood, d)
            require(index[root] in local_centers, "local center was lost")
            require(action_ball(centers, partial, root, r)
                    == action_ball(local_centers, local_maps, index[root], r),
                    "decoder exceeds its claimed radius")
            records.append({"root": root, "decoded_radius": r,
                            "encoding_radius": R, "observed_vertices": len(order)})
    return records


def repair_checks(g, d, horizon=2):
    centers, partial, bad = decode_partial(g, d)
    completed = repair(centers, partial)
    inverses = [{v: u for u, v in mapping.items()} for mapping in completed]
    for old, new in zip(partial, completed):
        for u, v in new.items():
            if u not in old:
                require(u in bad and v in bad, "repair edge escaped the bad set")
    records = []
    for r in range(horizon+1):
        affected, frontier = set(bad), set(bad)
        for _ in range(r):
            following = {m[v] for v in frontier for m in completed+inverses}
            frontier = following-affected
            affected.update(following)
        changed = {v for v in centers if action_ball(centers, partial, v, r)
                   != action_ball(centers, completed, v, r)}
        bound = len(bad)*sum((2*d)**j for j in range(r+1))
        require(changed <= affected, "repair changed a remote rooted ball")
        require(len(affected) <= bound, "affected-root count exceeds bound")
        if centers:
            before = action_histogram(centers, partial, r)
            after = action_histogram(centers, completed, r)
            distance = sum(abs(before.get(k, 0)-after.get(k, 0)) for k in before.keys() | after.keys())
            require(distance <= Q(2*bound, len(centers)), "repair distribution error")
        else:
            distance = Q(0)
        records.append({"radius": r, "changed_roots": len(changed),
                        "roots_near_bad_set": len(affected), "root_count_bound": bound,
                        "distribution_l1_change": distance})
    return {"centers": len(centers), "bad_centers": len(bad),
            "missing_outgoing_per_label": [len(centers)-len(m) for m in partial],
            "radii": records}


def malformed_fixtures():
    n, d = 7, 2
    action = (tuple((v+1) % n for v in range(n)), tuple((v+3) % n for v in range(n)))
    g, centers, tails, heads, owners = encode(action)
    base_edges = edge_set(g)
    center_leaf = next(v for v in g[centers[0]] if len(g[v]) == 1)
    tail_leaf = next(v for v in g[tails[0, 0]] if len(g[v]) == 1)
    head_leaf = next(v for v in g[heads[0, 0]] if len(g[v]) == 1)
    wire = tuple(sorted((tails[0, 0], heads[action[0][0], 0])))
    cases = {
        "missing_center_leaf": graph(len(g), base_edges-{tuple(sorted((centers[0], center_leaf)))}),
        "extra_center_leaf": graph(len(g)+1, base_edges | {(centers[0], len(g))}),
        "missing_cross_wire": graph(len(g), base_edges-{wire}),
        "joined_pendant_leaves": graph(len(g), base_edges | {tuple(sorted((tail_leaf, head_leaf)))}),
        "moved_tail_leaf": graph(len(g),
            base_edges-{tuple(sorted((tails[0, 0], tail_leaf)))}
            | {tuple(sorted((heads[0, 0], tail_leaf)))}),
        "no_centers_path": graph(8, [(i, i+1) for i in range(7)]),
    }
    result = []
    for name, h in cases.items():
        report = repair_checks(h, d)
        report.update(fixture=name, vertices=len(h),
                      locality_samples=local_decoder_check(h, d, samples=2))
        result.append(report)

    # Four wrong-label wires preserve every role and center while leaving
    # two missing arrows per generator. Reversed completion changes the action.
    n = 300
    action = (tuple((v+1) % n for v in range(n)), tuple((v+13) % n for v in range(n)))
    g, centers, tails, heads, _ = encode(action)
    wires = [(0, 0), (20, 0), (10, 1), (40, 1)]
    endpoints = [(tails[v, i], heads[action[i][v], i]) for v, i in wires]
    edges = edge_set(g)-{tuple(sorted(pair)) for pair in endpoints}
    for j, destination in enumerate((2, 3, 0, 1)):
        edges.add(tuple(sorted((endpoints[j][0], endpoints[destination][1]))))
    h = graph(len(g), edges)
    actual_centers, _, _ = decode_partial(h, d)
    require(actual_centers == set(centers), "wrong-label wires altered center recognition")
    report = repair_checks(h, d)
    require(report["missing_outgoing_per_label"] == [2, 2], "wrong-label fixture deficits")
    require(report["bad_centers"] == 8, "wrong-label fixture bad set")
    require(report["radii"][2]["root_count_bound"] < len(centers), "repair bound is vacuous")
    report.update(fixture="wrong_label_wires_300_vertex_action", vertices=len(h),
                  locality_samples=local_decoder_check(h, d, samples=4))
    result.append(report)
    return result


def cutoff(g, cap):
    bad = {v for v, row in enumerate(g) if len(row) > cap}
    return graph(len(g), ((u, v) for u, v in edge_set(g) if u not in bad and v not in bad))


def ball_with_original_labels(g, root, radius):
    vertices = ball_vertices(g, root, radius)
    edges = {(u, v) for u in vertices for v in g[u] if v in vertices and u < v}
    return vertices, edges


def cutoff_fixtures():
    rng = random.Random(20261001)
    cases = [
        ("path", graph(9, [(i, i+1) for i in range(8)]), 1),
        ("star", graph(18, [(0, v) for v in range(1, 18)]), 4),
        ("cycle_with_chords", graph(12, [(i, i+1) for i in range(11)]
            + [(0, 11), (0, 5), (0, 8), (3, 9)]), 2),
    ]
    for j in range(4):
        edges = [(u, v) for u in range(14) for v in range(u+1, 14) if rng.randrange(5) == 0]
        cases.append(("seeded_graph_"+str(j), graph(14, edges), 3+j))
    reports = []
    for name, g, cap in cases:
        cut = cutoff(g, cap)
        require(len(cut) == len(g) and max(map(len, cut)) <= cap, "degree cutoff contract")
        bad = {v for v in range(len(g)) if len(g[v]) > cap}
        rows = []
        for r in range(4):
            affected = set().union(*(ball_vertices(g, v, r) for v in bad)) if bad else set()
            changed = set()
            for root in range(len(g)):
                expected_vertices, expected_edges = ball_with_original_labels(cut, root, r)
                local, order, index = induced(g, ball_vertices(g, root, r+1))
                local_cut = cutoff(local, cap)
                found_vertices, found_edges = ball_with_original_labels(local_cut, index[root], r)
                found_vertices = {order[v] for v in found_vertices}
                found_edges = {tuple(sorted((order[u], order[v]))) for u, v in found_edges}
                require((expected_vertices, expected_edges) == (found_vertices, found_edges),
                        "cutoff requires more than radius r+1")
                if (expected_vertices, expected_edges) != ball_with_original_labels(g, root, r):
                    changed.add(root)
            require(changed <= affected, "cutoff changed an unaffected root")
            rows.append({"radius": r, "changed_roots": len(changed),
                         "roots_near_excess_degree": len(affected)})
        reports.append({"fixture": name, "vertices": len(g), "cap": cap,
                        "excess_degree_vertices": len(bad), "radii": rows})
    return reports


def combine_actions(actions, multiplicities):
    d = len(actions[0])
    maps = [[] for _ in range(d)]
    offset = 0
    for action, count in zip(actions, multiplicities):
        for _ in range(count):
            for i, permutation in enumerate(action):
                maps[i].extend(v+offset for v in permutation)
            offset += len(action[0])
    return tuple(tuple(m) for m in maps)


def root_and_mixture_fixtures():
    actions = [((1, 0), (0, 1)), ((1, 2, 0), (0, 2, 1))]
    weights = [Q(2, 7), Q(5, 7)]
    scaled = [w/len(action[0]) for w, action in zip(weights, actions)]
    L = lcm(*(q.denominator for q in scaled))
    copies = [int(L*q) for q in scaled]
    combined = combine_actions(actions, copies)
    n = len(combined[0])
    require(n == L, "rational disjoint union normalization")
    g, centers, _, _, owners = encode(combined)
    b, _ = sizes(2)
    require(Q(len(centers), len(g)) == Q(1, b), "center root density")
    center_set = set(centers)
    outgoing = [int(v not in center_set) for v in range(len(g))]
    incoming = [0] * len(g)
    for v in range(len(g)):
        if v not in center_set:
            incoming[centers[owners[v]]] += 1
    uniform_out = Q(sum(outgoing), len(g))
    uniform_in = Q(sum(incoming), len(g))
    center_out = Q(sum(outgoing[c] for c in centers), len(centers))
    center_in = Q(sum(incoming[c] for c in centers), len(centers))
    require(uniform_out == uniform_in == Q(b-1, b), "averaged-root transport balance")
    require(center_out == 0 and center_in == b-1, "center-only transport obstruction")
    decoded, maps, bad = decode_partial(g, 2)
    require(not bad, "disjoint union decoding")
    rows = []
    for r in range(3):
        expected = Counter()
        for action, weight in zip(actions, weights):
            hist = action_histogram(set(range(len(action[0]))), source_maps(action), r)
            for key, value in hist.items():
                expected[key] += weight*value
        actual = action_histogram(decoded, maps, r)
        require(dict(expected) == actual, "rational mixture and decoded law disagree")
        rows.append({"radius": r, "rooted_action_types": len(actual),
                     "probabilities": sorted(actual.values())})
    return {"weights": weights, "source_sizes": [2, 3], "copy_counts": copies,
            "disjoint_union_vertices": n, "encoded_vertices": len(g),
            "fiber_size": b, "center_probability": Q(1, b),
            "averaged_root_transport": {"outgoing": uniform_out, "incoming": uniform_in},
            "center_only_transport": {"outgoing": center_out, "incoming": center_in},
            "mixture_local_laws": rows}


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
                        default=Path(__file__).with_name("positive_cone_encoding_results.json"))
    args = parser.parse_args()
    results = {
        "status": "passed",
        "arithmetic": "Python standard library; exact combinatorics and rational probabilities",
        "finite_action_catalogues": finite_catalogues(),
        "malformed_encoding_and_repair": malformed_fixtures(),
        "degree_cutoff_locality": cutoff_fixtures(),
        "root_averaging_and_positive_mixtures": root_and_mixture_fixtures(),
        "scope": "Finite encoding/decoding mechanisms only. Every tested source action is finite and sofic.",
        "external_nonsofic_law_constructed": False,
        "numerical_separating_inequality_computed": False,
        "proof_review": "Independent review of the general proofs remains deferred.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(serialize(results), indent=2, sort_keys=True)+"\n")
    print("PASS: finite positive-cone encoding mechanisms:", args.output)


if __name__ == "__main__":
    main()
