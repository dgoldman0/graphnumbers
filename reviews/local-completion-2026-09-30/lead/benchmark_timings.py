"""Timing fields recorded in python/results/heat_benchmark.json, in milliseconds.

Run from the repository root. Supports the benchmark-interpretation finding:
the factorized prism path versus a dense eigensolve, and explicit
neighborhood extraction versus dense on the torus and irregular cases.
"""
import json

with open("python/results/heat_benchmark.json") as f:
    cases = json.load(f)["cases"]
for i, case in enumerate(cases):
    name = case.get("name") or case.get("case") or f"case {i}"
    dense = case.get("dense", {}).get("total_seconds")
    local = case.get("local_seconds")
    factorized = case.get("factorized", {}).get("seconds")
    row = f"{i}: {str(name)[:38]:38s} local={local * 1e3:9.2f}"
    if dense is not None:
        row += f"  dense={dense * 1e3:7.2f}  local/dense={local / dense:7.1f}x"
    if factorized is not None:
        row += f"  factorized={factorized * 1e3:6.2f}"
    print(row)
