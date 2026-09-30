"""Reproduce rational certificates for branching and crossing-cut heat."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as Q
import json
from pathlib import Path

from graphlocal import CutLineDefect, EdgeInteraction, controlled_heat, cycle, star


def decimal(value):
    value = Q(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    crossing = CutLineDefect() ** 2
    cases = [("crossing cuts E squared", crossing, Q(1, 10), "crossing", 2),
             ("crossing cuts E squared", crossing, Q(1, 2), "crossing", 2)]
    for k in (3, 4):
        value = EdgeInteraction(star(k), [(0, v, -1) for v in range(1, k + 1)])
        cases.append((f"all {k} star edges", value, Q(1, 2), "star", k))
    triangle = EdgeInteraction(cycle(3), [(0, 1, -1), (1, 2, -1), (0, 2, -1)])
    cases.append(("all three triangle edges", triangle, Q(1, 2), "triangle", 3))
    rows = []
    with localcontext() as ctx:
        ctx.prec = 80
        for name, value, t, kind, k in cases:
            certificate = controlled_heat(value, t, "1e-10")
            z = (-decimal(t)).exp()
            expected = (((1 - z ** 4) / 2) ** k if kind == "crossing" else
                        z * (1 - z) ** k if kind == "star" else -(1 - z) ** 3)
            assert decimal(certificate.interval.lower) <= expected <= decimal(certificate.interval.upper)
            assert certificate.interval.radius <= Q("1e-10")
            rows.append({"case": name, "independent_80_digit_reference": str(expected),
                         "reference_contained": True, "certificate": certificate.to_data()})
    result = {"method": "exact rational local-moment certificates",
              "interpretation": "Closed forms check the implementation; no runtime comparison is claimed.",
              "requested_radius": "1/10000000000", "all_references_contained": True,
              "cases": rows}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
