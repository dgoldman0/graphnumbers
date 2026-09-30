"""Compare certified local heat return with conventional dense diagonalization.

Run from python/: PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 examples/heat_benchmark.py --output results/heat_benchmark.json
NumPy/SciPy are needed only for this comparison, not for graphlocal itself.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import eigh

from graphlocal import Finite, Line, cartesian, cycle, graph, heat_return, path
from graphlocal.graphs import ball, isomorphic
from graphlocal.heat import lazy_returns


def clear_caches():
    ball.cache_clear()
    isomorphic.cache_clear()
    lazy_returns.cache_clear()


def dense_heat(g, t):
    start = time.perf_counter()
    laplacian = np.diag([row.bit_count() for row in g.rows]).astype(float)
    for u in range(g.n):
        for v in g.neighbors(u):
            laplacian[u, v] = -1
    built = time.perf_counter()
    eigenvalues = eigh(laplacian, eigvals_only=True, check_finite=False)
    value = np.exp(-t * eigenvalues).mean()
    done = time.perf_counter()
    return float(value), {"matrix_seconds": built - start, "eigensolve_seconds": done - built,
                          "total_seconds": done - start, "matrix_bytes": int(laplacian.nbytes)}


def irregular(n):
    rng = random.Random(20260930)
    edges = {(u, (u + 1) % n) for u in range(n)}
    vertices = list(range(n))
    rng.shuffle(vertices)
    for u, v in zip(vertices[:n // 2:2], vertices[1:n // 2:2]):
        if (u - v) % n not in (1, n - 1):
            edges.add((u, v))
    return graph(n, edges)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    t, tolerance = "1/2", "1e-8"
    # Initialize the numerical backend before measuring cases.
    dense_heat(path(3), 0.5)
    cases = [(f"cycle_{n}", cycle(n), Line(), None) for n in (64, 256, 768)]
    cases += [("prism_256", cartesian(cycle(128), path(2)),
               Line() * Finite.from_graph(path(2), True),
               Finite.from_graph(cycle(128), True) * Finite.from_graph(path(2), True)),
              ("torus_576", cartesian(cycle(24), cycle(24)), Line() * Line(),
               Finite.from_graph(cycle(24), True) ** 2),
              ("irregular_128", irregular(128), None, None)]
    records = []
    for name, g, limit, factorized in cases:
        value = Finite.from_graph(g, normalize=True)
        clear_caches()
        start = time.perf_counter()
        certificate = heat_return(value, t, tolerance)
        local_seconds = time.perf_counter() - start
        baseline, timing = dense_heat(g, 0.5)
        # Floating comparison is diagnostic; it is not the certificate proof.
        lower, upper = float(certificate.interval.lower), float(certificate.interval.upper)
        if not lower - 1e-12 <= baseline <= upper + 1e-12:
            raise ArithmeticError(f"{name}: numerical baseline outside certified interval")
        record = {
            "case": name, "vertices": g.n, "edges": g.edges,
            "local_seconds": local_seconds, "dense": {"value": baseline, **timing},
            "midpoint": float(certificate.interval.midpoint),
            "absolute_error_bound": float(certificate.interval.radius),
            "observed_difference": abs(float(certificate.interval.midpoint) - baseline),
            "certificate": certificate.to_data(),
        }
        if name.startswith("cycle_"):
            # A strong family-specific conventional baseline is also cheap.
            start = time.perf_counter()
            eigenvalues = 2 - 2 * np.cos(2 * np.pi * np.arange(g.n) / g.n)
            analytic = float(np.exp(-0.5 * eigenvalues).mean())
            record["cycle_formula"] = {"value": analytic, "seconds": time.perf_counter() - start}
        if limit is not None:
            clear_caches()
            start = time.perf_counter()
            limiting = heat_return(limit, t, tolerance)
            record["limit"] = {
                "seconds": time.perf_counter() - start,
                "midpoint": float(limiting.interval.midpoint),
                "error_bound": float(limiting.interval.radius),
                "same_local_data": certificate.local.histogram == limiting.local.histogram,
            }
        if factorized is not None:
            clear_caches()
            start = time.perf_counter()
            factored = heat_return(factorized, t, tolerance)
            record["factorized"] = {
                "seconds": time.perf_counter() - start,
                "same_local_data": certificate.local.histogram == factored.local.histogram,
                "same_interval": certificate.interval == factored.interval,
            }
            if not record["factorized"]["same_interval"]:
                raise ArithmeticError("Factorized and materialized certificates disagree")
        records.append(record)
        print(f"{name}: {record['midpoint']:.10f} +/- {record['absolute_error_bound']:.2g}; "
              f"local {local_seconds:.4f}s; dense {timing['total_seconds']:.4f}s; "
              f"{certificate.to_data()['local_types']} rooted types", flush=True)
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "threads": {key: os.environ.get(key) for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]},
        "time": t, "requested_error": tolerance, "cases": records,
        "method": "One cold graphlocal-cache run per case; graph construction and Finite construction excluded. Dense matrix construction included. Numerical backend warmed once.",
        "interpretation": "Synthetic examples, not a broad performance study. Dense eigensolvers, family-specific formulas, and exact certified local calculations provide different guarantees. No general speedup claim.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
