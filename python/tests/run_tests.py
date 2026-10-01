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
def test_ids(collection):
    for item in collection:
        if isinstance(item, unittest.TestSuite):
            yield from test_ids(item)
        else:
            yield item.id()


executed_ids = list(test_ids(suite))
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {
    "python": platform.python_version(), "tests_run": result.testsRun,
    "failures": len(result.failures), "errors": len(result.errors),
    "skipped": len(result.skipped), "success": result.wasSuccessful(),
    "seconds": time.perf_counter() - start,
    "discovered_test_ids": executed_ids,
    "skipped_tests": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
    "scope": "Execution report for the listed tests; mathematical proofs and independent-oracle scope are documented separately.",
}
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(0 if result.wasSuccessful() else 1)
