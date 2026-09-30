"""Compare certified local defect heat traces with conventional algorithms.

Run from python/: PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 examples/defect_benchmark.py --output results/defect_benchmark.json

All finite differences are UNNORMALIZED and use the same 1e-8 absolute error
target. Matrix inputs and graphlocal inputs share the identical labeled graphs.
No numerical comparison is presented as a rigorous floating-point certificate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.special import gammainc, i0e

from graphlocal import (CutLineDefect, Interval, Line, SparseEdgeDifference,
                        cycle, graph, heat_return, path, relative_heat)
from graphlocal.graphs import ball, isomorphic
from graphlocal.heat import lazy_returns

from defect_baselines import dense_heat, defect_krylov_heat, sparse_uniformized_heat


EPSILON = Fraction("1e-8")
TIMES = (Fraction(1, 2), Fraction(2))


def clear_caches():
    ball.cache_clear()
    isomorphic.cache_clear()
    lazy_returns.cache_clear()


def measure_certificate(call, repeats):
    samples = []
    for _ in range(repeats):
        clear_caches()
        start = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - start, result))
    samples.sort(key=lambda item: item[0])
    seconds, result = samples[len(samples) // 2]
    return result, {
        "seconds": seconds,
        "sample_seconds": [seconds for seconds, _ in samples],
        "cache_policy": "Graphlocal caches cleared before every sample.",
    }


def measure_numeric(call, repeats):
    samples = [call() for _ in range(repeats)]
    samples.sort(key=lambda item: item[1]["total_seconds"])
    value, metrics = samples[len(samples) // 2]
    metrics = dict(metrics)
    metrics["sample_seconds"] = [details["total_seconds"] for _, details in samples]
    return value, metrics


def dense_difference(before, after, at_time):
    a, timing_a = dense_heat(before, at_time, normalize=False)
    b, timing_b = dense_heat(after, at_time, normalize=False)
    return b - a, {
        "value": b - a,
        "before_value": a,
        "after_value": b,
        "matrix_seconds": timing_a["matrix_seconds"] + timing_b["matrix_seconds"],
        "eigensolve_seconds": (timing_a["eigensolve_seconds"]
                                + timing_b["eigensolve_seconds"]),
        "total_seconds": timing_a["total_seconds"] + timing_b["total_seconds"],
        "error_status": "Independent dense reference; floating error and subtraction are uncertified.",
    }


def sparse_difference(before, after, at_time, steps, degree):
    a, timing_a = sparse_uniformized_heat(before, at_time, steps, degree, normalize=False)
    b, timing_b = sparse_uniformized_heat(after, at_time, steps, degree, normalize=False)
    return b - a, {
        "value": b - a,
        "steps": steps,
        "matrix_seconds": timing_a["matrix_seconds"] + timing_b["matrix_seconds"],
        "powers_seconds": timing_a["powers_seconds"] + timing_b["powers_seconds"],
        "total_seconds": timing_a["total_seconds"] + timing_b["total_seconds"],
        "peak_power_nonzeros": max(timing_a["peak_power_nonzeros"], timing_b["peak_power_nonzeros"]),
        "combined_truncation_bound": timing_a["truncation_bound"] + timing_b["truncation_bound"],
        "error_status": "Poisson truncation bound excludes floating roundoff and subtraction error.",
    }


def irregular_cycle(n):
    # Short chords keep bounded geometry while producing several local types.
    edges = [(u, (u + 1) % n) for u in range(n)]
    edges += [(u, u + 3) for u in range(7, n - 3, 13)]
    return graph(n, edges)


def finite_cases():
    specs = [(f"cut_cycle_{n}", lambda n=n: cycle(n), [(0, n - 1, -1)])
             for n in (32, 128, 512, 4096)]
    specs.append(("chord_insertion_128", lambda: cycle(128), [(0, 4, +1)]))
    specs.extend((f"irregular_mixed_{n}", lambda n=n: irregular_cycle(n),
                  [(0, 1, -1), (0, 4, +1)]) for n in (128, 256))
    for name, builder, edits in specs:
        start = time.perf_counter()
        before = builder()
        built = time.perf_counter()
        defect = SparseEdgeDifference(before, edits)
        prepared = time.perf_counter()
        yield name, defect, {
            "before_graph_seconds": built - start,
            "after_graph_and_defect_metadata_seconds": prepared - built,
            "total_seconds": prepared - start,
            "policy": "Shared graph setup, performed once per graph and reused at both times.",
        }


def assert_reference(certificate, value, label):
    # Compare the represented float exactly to the rational endpoints. This
    # diagnoses agreement; it does not certify the floating algorithm's error.
    if not certificate.interval.lower <= Fraction(value) <= certificate.interval.upper:
        raise ArithmeticError(f"{label}: numerical reference outside certificate")


def finite_record(name, defect, setup, at_time, repeats):
    certificate, timing = measure_certificate(
        lambda: relative_heat(defect, at_time, EPSILON), repeats)
    if certificate.interval.radius > EPSILON:
        raise ArithmeticError("Certified interval exceeds requested absolute error")
    midpoint = float(certificate.interval.midpoint)
    # Match polynomial trace degree. The dimension-free perturbation tail also
    # bounds Krylov error in exact arithmetic; deflation/roundoff remain outside.
    blocks = max(1, (certificate.steps + 1) // 2)
    krylov, krylov_timing = measure_numeric(
        lambda: defect_krylov_heat(defect.before, defect.after, at_time, blocks,
                                  defect.edits, normalize=False), repeats)
    assert_reference(certificate, krylov, name + " Krylov")
    record = {
        "case": name,
        "vertices": defect.before.n,
        "before_edges": defect.before.edges,
        "after_edges": defect.after.edges,
        "edits": [list(edit) for edit in defect.edits],
        "time": str(at_time),
        "normalization": "Unnormalized trace(after)-trace(before).",
        "shared_setup": setup,
        "certified": {
            **timing,
            "value": midpoint,
            "absolute_error_bound": float(certificate.interval.radius),
            "affected_roots": len(defect.affected_roots(certificate.radius)),
            "certificate": certificate.to_data(),
        },
        "krylov": {**krylov_timing, "value": krylov,
                    "exact_arithmetic_krylov_tail_bound": 2 * float(certificate.defect_tail_bound),
                    "tail_bound_qualification": "Assumes the exact Krylov space and polynomial trace exactness; excludes floating-point roundoff and SVD rank truncation.",
                    "difference_from_certified_midpoint": abs(krylov - midpoint)},
    }
    if defect.before.n <= 512:
        dense, dense_timing = measure_numeric(
            lambda: dense_difference(defect.before, defect.after, at_time), repeats)
        assert_reference(certificate, dense, name + " dense")
        record["dense"] = dense_timing
        record["krylov"]["absolute_difference_from_dense"] = abs(krylov - dense)
        record["certified"]["absolute_difference_from_dense"] = abs(midpoint - dense)
        sparse_steps = 0
        intensity = defect.degree_bound * float(at_time)
        while 2 * defect.before.n * gammainc(sparse_steps + 1, intensity) > float(EPSILON) / 4:
            sparse_steps += 1
        sparse, sparse_timing = measure_numeric(
            lambda: sparse_difference(defect.before, defect.after, at_time,
                                      sparse_steps, defect.degree_bound), repeats)
        assert_reference(certificate, sparse, name + " sparse")
        sparse_timing["absolute_difference_from_dense"] = abs(sparse - dense)
        record["sparse_uniformization"] = sparse_timing
    if name.startswith("cut_cycle"):
        limit = -math.expm1(-4 * float(at_time)) / 2
        record["analytic_infinite_cut"] = {
            "value": limit,
            "difference_from_certified_midpoint": abs(limit - midpoint),
            "status": "Floating evaluation of the infinite-size limit, not a finite-graph certificate.",
        }
    print(f"{name}, t={at_time}: {midpoint:.12g} +/- "
          f"{float(certificate.interval.radius):.2g}; local {timing['seconds']:.4g}s, "
          f"Krylov {krylov_timing['total_seconds']:.4g}s; "
          f"r={certificate.radius}, affected={record['certified']['affected_roots']}", flush=True)
    return record


def factorized_infinite(at_time):
    left = relative_heat(CutLineDefect(), at_time, EPSILON / 4)
    right = heat_return(Line(), at_time, EPSILON / 4)
    endpoints = [a * b for a in (left.interval.lower, left.interval.upper)
                 for b in (right.interval.lower, right.interval.upper)]
    return Interval(min(endpoints), max(endpoints)), left, right


def infinite_record(product, at_time, repeats):
    value = CutLineDefect() * Line() if product else CutLineDefect()
    certificate, timing = measure_certificate(
        lambda: relative_heat(value, at_time, EPSILON), repeats)
    analytic = -math.expm1(-4 * float(at_time)) / 2
    if product:
        analytic *= float(i0e(2 * float(at_time)))
    assert_reference(certificate, analytic, "infinite cut product" if product else "infinite cut")
    record = {
        "case": "cut_line_times_line" if product else "cut_line",
        "time": str(at_time),
        "finite_graph_materialized": False,
        "global_variation_bound": None,
        "certified": {**timing, "certificate": certificate.to_data(),
                       "value": float(certificate.interval.midpoint),
                       "absolute_error_bound": float(certificate.interval.radius)},
        "analytic": {
            "value": analytic,
            "formula": "(1-exp(-4t))*exp(-2t)*I_0(2t)/2" if product else "(1-exp(-4t))/2",
            "status": "Independent floating evaluation of the analytic formula.",
        },
    }
    if product:
        (interval, left, right), factor_timing = measure_certificate(
            lambda: factorized_infinite(at_time), repeats)
        if interval.radius > EPSILON:
            raise ArithmeticError("Factorized interval exceeds requested absolute error")
        if interval.upper < certificate.interval.lower or interval.lower > certificate.interval.upper:
            raise ArithmeticError("Independent exact intervals do not overlap")
        record["factorized_certified"] = {
            **factor_timing,
            "interval": interval.to_data(),
            "value": float(interval.midpoint),
            "absolute_error_bound": float(interval.radius),
            "left_steps": left.steps,
            "right_steps": right.steps,
            "basis": "Cartesian heat trace multiplicativity, with exact interval endpoint products.",
        }
    print(f"{record['case']}, t={at_time}: {record['certified']['value']:.12g}; "
          f"local {timing['seconds']:.4g}s", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats % 2 == 0:
        parser.error("--repeats must be a positive odd integer")
    # Warm the shared numerical backend before timing either numerical method.
    dense_heat(path(3), 0.5)
    defect_krylov_heat(cycle(5), path(5), 0.5, 2)
    sparse_uniformized_heat(path(3), 0.5, 2)
    records = [finite_record(name, defect, setup, at_time, args.repeats)
               for name, defect, setup in finite_cases() for at_time in TIMES]
    limits = [infinite_record(product, at_time, args.repeats)
              for product in (False, True) for at_time in TIMES]
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "requested_absolute_error": str(EPSILON),
        "normalization": "All finite differences use unnormalized heat traces; no shrinking 1/n signal.",
        "repeats": args.repeats,
        "timing_policy": "Median of independent samples; graphlocal caches cleared per sample. Shared Graph construction/defect metadata reported separately. Matrix construction included in numerical totals. Backend warmed once. Both times reuse the same input Graph objects.",
        "accuracy_policy": "Rational interval radius <= requested error for certified outputs. Krylov uses m=ceil(M/2), matching the certificate's polynomial moment degree. Its dimension-free exact-arithmetic tail bound is conservatively twice the certificate defect tail; numerical SVD deflation and floating roundoff are excluded. Dense comparisons and analytic values are floating diagnostics. Sparse powers use a combined theoretical Poisson tail <=epsilon/4; roundoff remains outside that bound.",
        "scope": "Synthetic fixed edge changes and their cut-line limits. Classical Krylov, finite-rank trace identities, uniformization, and Cartesian heat factorization are established techniques. Measurements establish neither a new algorithm nor a general advantage over existing approaches.",
        "finite_cases": records,
        "infinite_cases": limits,
        "summary": {
            "finite_comparisons": len(records),
            "infinite_comparisons": len(limits),
            "dense_comparisons": sum("dense" in row for row in records),
            "all_numerical_references_inside_exact_rational_intervals": True,
            "max_krylov_dense_absolute_difference": max(
                row["krylov"].get("absolute_difference_from_dense", 0) for row in records),
            "median_local_over_krylov_time": statistics.median(
                row["certified"]["seconds"] / row["krylov"]["total_seconds"] for row in records),
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
