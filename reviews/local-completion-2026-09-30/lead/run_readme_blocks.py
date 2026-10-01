"""Execute every ```python block in python/README.md as a separate script.

Run from the repository root. Each block runs with PYTHONPATH=python/src in
a fresh interpreter; the script reports OK, FAIL or TIMEOUT per block.
"""
import os
import re
import subprocess
import sys
import tempfile

with open("python/README.md") as f:
    blocks = re.findall(r"```python\n(.*?)```", f.read(), re.S)
print(f"python blocks: {len(blocks)}")
env = dict(os.environ, PYTHONPATH=os.path.abspath("python/src"), PYTHONDONTWRITEBYTECODE="1")
failures = 0
with tempfile.TemporaryDirectory() as scratch:
    for i, block in enumerate(blocks):
        path = os.path.join(scratch, f"block{i:02d}.py")
        with open(path, "w") as f:
            f.write(block)
        try:
            result = subprocess.run([sys.executable, path], capture_output=True, text=True,
                                    timeout=600, env=env)
            status = "OK" if result.returncode == 0 else f"FAIL rc={result.returncode}"
            detail = "" if result.returncode == 0 else (result.stderr.strip().splitlines() or [""])[-1][:200]
        except subprocess.TimeoutExpired:
            status, detail = "TIMEOUT", ""
        failures += status != "OK"
        first = block.strip().splitlines()[0][:70] if block.strip() else ""
        print(f"block {i:02d} {status:10s} | {first} {detail}")
raise SystemExit(1 if failures else 0)
