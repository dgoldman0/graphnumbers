"""Run standard-library tests and optionally record a machine-readable result."""
import argparse
import json
import platform
import time
import unittest
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path)
args = parser.parse_args()
start = time.perf_counter()
suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {
    "python": platform.python_version(), "tests_run": result.testsRun,
    "failures": len(result.failures), "errors": len(result.errors),
    "skipped": len(result.skipped), "success": result.wasSuccessful(),
    "seconds": time.perf_counter() - start,
    "coverage": ["160 independent brute-force isomorphism comparisons",
                 "300 exact local/full Cartesian histogram comparisons",
                 "rational full-matrix checks of the 2R locality boundary",
                 "70-digit independent heat and exponential closed forms",
                 "deliberately inexact oracle error propagation",
                 "reconstruction duals and infeasibility witnesses",
                 "258 sparse-edit local/full histogram comparisons",
                 "cut-line limit, variation growth, and exact lazy-return moments",
                 "relative heat, signed Cartesian propagation, separated/nearby defects",
                 "optional NumPy/SciPy block Krylov polynomial exactness and dense comparisons"],
}
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(0 if result.wasSuccessful() else 1)
