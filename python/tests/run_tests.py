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
                 "prepared geometry/moment reuse, including inexact source oracles",
                 "line-cut inclusion-exclusion versus finite subset sums through five cuts",
                 "interaction support, binomial moments, and positive image series",
                 "942 tree cut-set identities, 3763 local identities, and exact branching/cycle moments",
                 "buffered square-lattice pair geometry and leading coupling formula",
                 "crossing-cut products, compositional profiles and inexact controlled heat certificates",
                 "cyclic incidence moments versus full integer subset matrices, with repeated defects",
                 "249 exhaustive tree cut leading terms and exact order-five/six cancellation fixtures",
                 "rationally certified time-dependent interaction sign reversal",
                 "support-distance decay, geometric profiles, larger degree caps and Cartesian products",
                 "674 exact cospectral geometry checks and independent motif comparators",
                 "1671 exact arithmetic geometry checks, joint correlations and link characters",
                 "Neumann inverse coefficients, weighted tails and inexact source stability",
                 "joint local laws, higher moment jets, cumulants and formal reciprocal identities",
                 "certified jet extraction and incompatible custom statistic identity rejection",
                 "3780 exact cut-line coordinate, phase, power norm and entire-calculus checks",
                 "unbounded-variation exponential units, weighted Poisson tails and inverse identity",
                 "local inverse residuals, inexact refinement and local/global invertibility separation",
                 "optional NumPy/SciPy block Krylov polynomial exactness and dense comparisons"],
}
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(0 if result.wasSuccessful() else 1)
