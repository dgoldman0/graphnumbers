"""Reproduce separation-sensitive heat bounds without floating point."""
import argparse
from fractions import Fraction as Q
import json
from pathlib import Path

from graphlocal import (EdgeInteraction, GeometricInteraction, controlled_heat,
                        cycle, interaction_geometry, interaction_heat_bound)


def build_report():
    rows = []
    for size in (18, 30, 48):
        spacing = size // 3
        value = EdgeInteraction(cycle(size), [(i, i + 1, -1) for i in (0, spacing, 2 * spacing)])
        geometry = interaction_geometry(value)
        bound = interaction_heat_bound(geometry, "1/2", "1e-10")
        geometric = controlled_heat(GeometricInteraction(value), "1/2", "1e-10")
        generic = controlled_heat(value, "1/2", "1e-10")
        assert geometry.vanishing_order == size
        assert bound.magnitude_bound < Q("1e-10")
        assert geometric.interval.radius < Q("1e-10")
        assert geometric.steps == 0
        rows.append({"cycle_size": size, "cut_spacing": spacing,
                     "time": "1/2", "generic_profile_magnitude_bound": "1",
                     "geometry_bound": bound.to_data(),
                     "geometric_profile_heat": geometric.to_data(),
                     "generic_profile_heat": generic.to_data()})
    return {"arithmetic": "Exact rational bounds and certificates; no floating point used.",
            "interpretation": "Same three edits and degree cap two. Separation improves analytic bounds and reduces required moment order. No timing or algorithmic superiority claim.",
            "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(build_report(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
