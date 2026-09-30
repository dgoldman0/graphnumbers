"""Measure reusable local geometry against reusable conventional representations.

Run from python/: PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 examples/reuse_benchmark.py --output results/reuse_benchmark.json

All methods receive the same labeled graphs and affected root set. Conventional
methods may cancel identically labeled balls, memoize each root computation,
retain all heat moments, and reuse one common Krylov compression. No method is
charged repeatedly for preprocessing which another method is allowed to reuse.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from fractions import Fraction as Q
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import eigh

from graphlocal import (LocalHistogram, PreparedRelativeHeat, SparseEdgeDifference,
                        cycle, graph)
from graphlocal.graphs import ball, isomorphic
from graphlocal.heat import lazy_returns
from defect_baselines import _krylov_projection


TIMES = tuple(Q(j, 16) for j in range(1, 33))
EPSILON = Q("1e-8")


def clear_caches():
    ball.cache_clear()
    isomorphic.cache_clear()
    lazy_returns.cache_clear()


def measured(call, repeats, clear=None):
    samples = []
    for _ in range(repeats):
        if clear:
            clear()
        start = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - start, result))
    samples.sort(key=lambda item: item[0])
    seconds, result = samples[len(samples) // 2]
    return result, {"seconds": seconds, "sample_seconds": [s for s, _ in samples]}


def extract_terms(defect, radius):
    return tuple((g, c) for u in defect.affected_roots(radius)
                 for g, c in ((ball(defect.after, u, radius), defect.scale),
                              (ball(defect.before, u, radius), -defect.scale)))


def label_aggregate(terms):
    grouped = {}
    for g, c in terms:
        grouped[g] = grouped.get(g, Q(0)) + c
    return tuple((g, c) for g, c in grouped.items() if c)


def moments(terms, degree, steps):
    answer = [Q(0)] * (steps + 1)
    for g, c in terms:
        for j, value in enumerate(lazy_returns(g, degree, steps)):
            answer[j] += c * value
    return tuple(answer)


def root_triangles(g):
    neighbors = g.rows[0]
    return sum((g.rows[v] & neighbors).bit_count() for v in g.neighbors(0)) // 2


def root_four_cliques(g):
    return sum(bool(g.rows[u] & (1 << v) and g.rows[u] & (1 << w)
                    and g.rows[v] & (1 << w))
               for u, v, w in combinations(g.neighbors(0), 3))


OBSERVABLES = tuple(
    (f"degree_{degree}_vertices", lambda g, degree=degree: int(g.rows[0].bit_count() == degree))
    for degree in range(1, 5)) + (
        ("triangles", lambda g: Q(root_triangles(g), 3)),
        ("four_cliques", lambda g: Q(root_four_cliques(g), 4)),
        ("wedges", lambda g: Q(g.rows[0].bit_count() * (g.rows[0].bit_count() - 1), 2)),
    )


def evaluate_observable(terms, function):
    # Exact-label memoization is available even to the ungrouped root baseline.
    cached, total = {}, Q(0)
    for g, c in terms:
        if g not in cached:
            cached[g] = Q(function(g))
        total += c * cached[g]
    return total


def prepare_krylov(defect, blocks):
    original, modified, metadata = _krylov_projection(
        defect.before, defect.after, blocks, defect.edits)
    return (eigh(original, eigvals_only=True, check_finite=False),
            eigh(modified, eigvals_only=True, check_finite=False), metadata)


def krylov_query(prepared, at_time):
    original, modified, _ = prepared
    t, values = float(at_time), []
    for a, b in zip(original, modified):
        if b >= a:
            values.append(math.exp(-t * a) * math.expm1(-t * (b - a)))
        else:
            values.append(-math.exp(-t * b) * math.expm1(-t * (a - b)))
    return math.fsum(values)


def irregular(n):
    return graph(n, [(u, (u + 1) % n) for u in range(n)]
                 + [(u, u + 3) for u in range(7, n - 3, 13)])


def clique_chain(n):
    return graph(n, [(u, v) for start in range(0, n, 4)
                     for u, v in combinations(range(start, start + 4), 2)]
                 + [(u, u + 1) for u in range(3, n - 1, 4)])


def cases():
    specs = []
    for n in (128, 512):
        specs.extend([
            (f"cut_cycle_{n}", lambda n=n: cycle(n), [(0, n - 1, -1)]),
            (f"chord_cycle_{n}", lambda n=n: cycle(n), [(0, 4, +1)]),
            (f"irregular_mixed_{n}", lambda n=n: irregular(n), [(0, 1, -1), (0, 4, +1)]),
        ])
    specs.append(("clique_chain_mixed_128", lambda: clique_chain(128),
                  [(0, 1, -1), (0, 5, +1)]))
    for name, builder, edits in specs:
        start = time.perf_counter()
        defect = SparseEdgeDifference(builder(), edits)
        yield name, defect, time.perf_counter() - start


def benchmark(name, defect, setup_seconds, repeats):
    production, production_timing = measured(
        lambda: PreparedRelativeHeat(defect, TIMES[-1], EPSILON), repeats, clear_caches)
    radius, steps = production.radius, production.steps
    raw, extraction_timing = measured(
        lambda: extract_terms(defect, radius), repeats, clear_caches)
    labeled, label_timing = measured(lambda: label_aggregate(raw), repeats)
    histogram, iso_timing = measured(
        lambda: LocalHistogram(radius, raw), repeats, isomorphic.cache_clear)
    isoterms = tuple((key.graph, c) for key, c in histogram.values.items())
    representations = {"rootwise": raw, "label_aggregated": labeled, "isomorphism_aggregated": isoterms}
    moment_timings = {}
    for method, terms in representations.items():
        result, timing = measured(lambda terms=terms: moments(terms, defect.degree_bound, steps),
                                  repeats, lazy_returns.cache_clear)
        if result != tuple(production.returns):
            raise ArithmeticError(f"{name}: {method} moments differ")
        timing["terms"] = len(terms)
        timing["cache_misses_last_sample"] = lazy_returns.cache_info().misses
        moment_timings[method] = timing
    certificates, heat_query_timing = measured(
        lambda: [production.evaluate(t) for t in TIMES], repeats)
    krylov, krylov_prep_timing = measured(
        lambda: prepare_krylov(defect, max(1, (steps + 1) // 2)), repeats)
    numeric, krylov_query_timing = measured(
        lambda: [krylov_query(krylov, t) for t in TIMES], repeats)
    heat_rows = []
    for t, certificate, estimate in zip(TIMES, certificates, numeric):
        if certificate.interval.radius > EPSILON:
            raise ArithmeticError("Prepared certificate exceeds requested tolerance")
        if not certificate.interval.lower <= Q(estimate) <= certificate.interval.upper:
            raise ArithmeticError(f"{name}: Krylov and exact interval disagree")
        heat_rows.append({"time": str(t), "interval": certificate.interval.to_data(),
                          "krylov_estimate": estimate,
                          "exact_arithmetic_krylov_tail_bound": 2 * float(certificate.defect_tail_bound)})
    motif_results, motif_timings = {}, {method: [] for method in representations}
    for title, function in OBSERVABLES:
        results = {}
        for method, terms in representations.items():
            answer, timing = measured(lambda terms=terms: evaluate_observable(terms, function), repeats)
            results[method] = answer
            motif_timings[method].append(timing["seconds"])
        if len(set(results.values())) != 1:
            raise ArithmeticError(f"{name}: representations disagree on {title}")
        motif_results[title] = str(results["rootwise"])
    # Extra geometry preprocessing relative to the strongest simple comparator.
    extra = iso_timing["seconds"] - label_timing["seconds"]
    gains = [a - b for a, b in zip(motif_timings["label_aggregated"],
                                  motif_timings["isomorphism_aggregated"])]
    running, observed = extra, None
    for count, gain in enumerate(gains, 1):
        running -= gain
        if observed is None and running <= 0:
            observed = count
    mean_gain = sum(gains) / len(gains)
    projected = max(0, math.ceil(extra / mean_gain)) if mean_gain > 0 else None
    aggregation_times = {"rootwise": 0.0, "label_aggregated": label_timing["seconds"],
                         "isomorphism_aggregated": iso_timing["seconds"]}
    totals = {method: extraction_timing["seconds"] + aggregation_times[method]
              + moment_timings[method]["seconds"] + heat_query_timing["seconds"]
              for method in representations}
    print(f"{name}: {len(raw)} root terms -> {len(labeled)} labeled -> {len(isoterms)} iso; "
          f"heat totals label={totals['label_aggregated']:.4g}s, "
          f"iso={totals['isomorphism_aggregated']:.4g}s; "
          f"Krylov={krylov_prep_timing['seconds'] + krylov_query_timing['seconds']:.4g}s", flush=True)
    return {
        "case": name, "vertices": defect.before.n, "degree_bound": defect.degree_bound,
        "edits": [list(edit) for edit in defect.edits], "shared_graph_setup_seconds": setup_seconds,
        "radius": radius, "steps": steps, "affected_roots": len(raw) // 2,
        "root_terms": len(raw), "label_terms_after_cancellation": len(labeled),
        "isomorphism_types_after_cancellation": len(isoterms),
        "geometry": {"shared_extraction": extraction_timing,
                     "label_aggregation": label_timing, "isomorphism_aggregation": iso_timing},
        "moment_preprocessing": moment_timings,
        "all_moment_vectors_exactly_equal": True,
        "heat": {
            "query_count": len(TIMES), "shared_rational_query_timing": heat_query_timing,
            "shared_query_explanation": "All three exact representations produce the identical moment vector. The same interval evaluator is timed once and charged equally to every method.",
            "staged_total_seconds": totals,
            "production_PreparedRelativeHeat_preprocessing": production_timing,
            "production_total_seconds": production_timing["seconds"] + heat_query_timing["seconds"],
            "krylov_preprocessing": krylov_prep_timing,
            "krylov_queries": krylov_query_timing,
            "krylov_total_seconds": krylov_prep_timing["seconds"] + krylov_query_timing["seconds"],
            "krylov_subspace_dimension": krylov[2]["subspace_dimension"],
            "additional_heat_queries_amortize_grouping": False,
            "explanation": "After precomputing equal return moments, additional heat times have identical exact query cost. Geometry aggregation can only help or hurt the one-time moment preparation.",
            "queries": heat_rows,
        },
        "motifs": {
            "query_order": [title for title, _ in OBSERVABLES], "exact_values": motif_results,
            "query_seconds": motif_timings,
            "observed_break_even_query_count_against_label_aggregation": observed,
            "projected_break_even_distinct_queries_at_measured_mean_cost": projected,
            "projection_qualification": "Extrapolation to NEW observables with comparable cost; repeated cached answers give no such amortization. Tiny query timings are noisy.",
            "non_spectral_scope": "Degree-bin and four-clique observables are general local information. Triangle counts are spectral for the adjacency matrix and are not advertised as a universal non-spectral invariant.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats % 2 == 0:
        parser.error("--repeats must be a positive odd integer")
    prepare_krylov(SparseEdgeDifference(cycle(8), [(0, 7, -1)]), 2)
    rows = [benchmark(name, defect, seconds, args.repeats)
            for name, defect, seconds in cases()]
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "repeats": args.repeats, "time_queries": [str(t) for t in TIMES],
        "requested_absolute_error": str(EPSILON),
        "timing_policy": "Median stage timings from independent samples. Exact-label memoization is allowed in every representation. Extraction shared across staged comparisons. Whole production preprocessing also measured independently with cold caches. Numerical preparation includes CSR conversion, Krylov compression and eigendecompositions. Graph construction reported separately.",
        "accuracy_policy": "Raw, label-aggregated and isomorphism-aggregated moments and local observables must agree exactly. Rational heat intervals have radius<=1e-8. Krylov estimates are floating diagnostics; dimension-free exact-arithmetic tails exclude SVD deflation and roundoff.",
        "scope": "This tests representation reuse against conventional reuse, including exact-label cancellation and cached moments. It does not establish a generic graphnumbers speed advantage or charge repeated preprocessing to a comparator.",
        "cases": rows,
        "summary": {"cases": len(rows), "heat_queries": len(rows) * len(TIMES),
                    "motif_queries": len(rows) * len(OBSERVABLES),
                    "all_exact_representations_agree": True,
                    "motif_suites_reaching_observed_break_even": sum(
                        row["motifs"]["observed_break_even_query_count_against_label_aggregation"] is not None
                        for row in rows)},
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
