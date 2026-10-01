"""List the exact-check counts quoted in the READMEs and the totals recorded in JSON.

Run from the repository root. This compares numbers only; whether each
counted check is substantive is assessed in the referee reports.
"""
import glob
import json
import os
import re

SOURCES = ["README.md", "research/local-completion/README.md", "python/README.md",
           "research/local-completion/local_graph_completion.tex"]
PATTERN = re.compile(r"(\d[\d,]*)\s+(?:[a-z-]+\s+){0,3}checks")
quoted = {}
for source in SOURCES:
    with open(source) as f:
        for match in PATTERN.finditer(f.read()):
            quoted.setdefault(match.group(1), set()).add(source)
print("Counts quoted in documentation:")
for number, sources in sorted(quoted.items(), key=lambda kv: int(kv[0].replace(",", ""))):
    print(f"  {number:>8}  {sorted(sources)}")

print("Totals recorded in JSON results:")
KEYS = ("total_checks", "total_recorded_checks", "exact_checks", "checks")
for path in sorted(glob.glob("research/local-completion/*.json") + glob.glob("python/results/*.json")):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        found = [(k, data[k]) for k in KEYS if isinstance(data.get(k), int) and not isinstance(data.get(k), bool)]
        if found:
            print(f"  {os.path.basename(path):45s} {found}")
