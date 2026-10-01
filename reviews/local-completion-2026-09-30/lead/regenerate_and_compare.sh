#!/usr/bin/env bash
# Regenerate every deterministic recorded result and compare it with the
# committed JSON (timing/platform keys ignored; see compare_json.py).
# Run from the repository root. Benchmarks that record wall-clock timings
# (heat_benchmark, defect_benchmark, reuse_benchmark) are not regenerated.
set -u
ROOT=$(pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)
status=0
cd "$ROOT/research/local-completion"
for v in verify_local_algebra:verification_results \
         verify_representation:representation_results \
         verify_multiplication:multiplication_results \
         verify_spectral_approximation:spectral_approximation_results \
         verify_intrinsic_structure:intrinsic_structure_results \
         verify_reflection_extension:reflection_extension_results; do
    script=${v%%:*}; result=${v##*:}
    python3 -B "$script.py" --output "$OUT/$result.json" > "$OUT/$script.log" 2>&1 \
        || { echo "$script.py exited with an error"; status=1; }
    python3 "$HERE/compare_json.py" "$result.json" "$OUT/$result.json" || status=1
done
python3 -B reconstruct_local.py --input reconstruction_example.json \
    --output "$OUT/reconstruction_example_result.json" > "$OUT/reconstruct.log" 2>&1 \
    || { echo "reconstruct_local.py exited with an error"; status=1; }
python3 "$HERE/compare_json.py" reconstruction_example_result.json \
    "$OUT/reconstruction_example_result.json" || status=1
cd "$ROOT/python"
for n in arithmetic_geometry_verification branch_planar_verification certified_arithmetic \
         controlled_heat_examples defect_breadth_verification geometry_bound_examples \
         higher_interaction_verification interaction_analysis unbounded_inverse_examples \
         unbounded_inverse_verification; do
    PYTHONPATH=src python3 -B "examples/$n.py" --output "$OUT/$n.json" > "$OUT/$n.log" 2>&1 \
        || { echo "examples/$n.py exited with an error"; status=1; }
    python3 "$HERE/compare_json.py" "results/$n.json" "$OUT/$n.json" || status=1
done
PYTHONPATH=src python3 -B examples/cospectral_geometry.py > "$OUT/cospectral_geometry.json" 2> "$OUT/cospectral_geometry.log" \
    || { echo "examples/cospectral_geometry.py exited with an error"; status=1; }
python3 "$HERE/compare_json.py" results/cospectral_geometry.json "$OUT/cospectral_geometry.json" || status=1
rm -rf "$OUT"
exit $status
