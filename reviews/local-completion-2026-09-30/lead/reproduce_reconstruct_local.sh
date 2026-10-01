#!/usr/bin/env bash
# Input-validation behavior of research/local-completion/reconstruct_local.py.
# Run from the repository root.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)
cd research/local-completion
echo "== disconnected rooted target (rows [0,0]), normal interpreter"
python3 -B reconstruct_local.py --input "$HERE/inputs/disconnected_target.json" --output "$OUT/a.json" 2>&1 | tail -1
echo "== asymmetric adjacency rows [2,0], normal interpreter"
python3 -B reconstruct_local.py --input "$HERE/inputs/asymmetric_rows.json" --output "$OUT/b.json" 2>&1 | tail -1
echo "== asymmetric adjacency rows [2,0], python -O (asserts disabled)"
python3 -B -O reconstruct_local.py --input "$HERE/inputs/asymmetric_rows.json" --output "$OUT/c.json" 2>&1 | tail -1
python3 -c "import json,sys; print('   reported status:', json.load(open(sys.argv[1])).get('status'))" "$OUT/c.json"
rm -rf "$OUT"
