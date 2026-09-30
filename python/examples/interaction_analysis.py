"""Certified line-cut heat and resolvent interactions from one graph element.

The resolvent routine is specialized to the proved CutInteraction moment
formula. It is not a claimed resolvent evaluator for arbitrary elements.
"""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as Q
import json
from math import comb
from pathlib import Path

from graphlocal import (BudgetExceeded, CutInteraction, Interval,
                        connected_cut_interaction, relative_heat)
from graphlocal.graphs import integer, rational


def interaction_moment(ell, order):
    integer(ell, "segment vertices", 1)
    integer(order, "moment order")
    if order % 2:
        return Q(0)
    return Q(2 * ell, 2 ** order) * sum(
        comb(order, order // 2 + ell * k)
        for k in range(1, order // (2 * ell) + 1))


def interaction_resolvent(ell, shift, epsilon="1e-12", max_steps=4096):
    """Rational enclosure using 0<=d_j<=1 and a geometric remainder.

    R_s=(s+2)^(-1) sum_j [2/(s+2)]^j d_j. The remainder after M is
    nonnegative and at most [2/(s+2)]^(M+1)/s. This is a family-specific
    consequence of the proof, stronger than its generic edit certificate.
    """
    integer(ell, "segment vertices", 1)
    integer(max_steps, "max_steps")
    shift, epsilon = rational(shift), rational(epsilon)
    if shift <= 0 or epsilon <= 0:
        raise ValueError("Shift and tolerance must be positive")
    rho = 2 / (shift + 2)
    weight, retained = Q(1), Q(0)
    for order in range(max_steps + 1):
        retained += weight * interaction_moment(ell, order) / (shift + 2)
        weight *= rho
        tail = weight / shift
        if tail <= 2 * epsilon:
            return Interval(retained, retained + tail), order
    raise BudgetExceeded("Resolvent series exceeds max_steps")


def decimal(x):
    x = Q(x)
    return Decimal(x.numerator) / Decimal(x.denominator)


def build_report():
    heat, resolvent = [], []
    for ell in (1, 2, 4, 8):
        for t in (Q(1, 4), Q(1), Q(3)):
            result = relative_heat(CutInteraction(ell), t, "1e-10")
            heat.append({"segment_vertices": ell, "time": str(t),
                         "interval": result.interval.to_data(),
                         "midpoint": float(result.interval.midpoint),
                         "moment_order": result.steps, "radius": result.radius,
                         "exact_first_nonzero_order": 2 * ell,
                         "positive_by_proof": True})
        for shift in (Q(1), Q(2), Q(4)):
            interval, steps = interaction_resolvent(ell, shift)
            with localcontext() as ctx:
                ctx.prec = 80
                s = decimal(shift)
                radical = (s * (s + 4)).sqrt()
                root = (s + 2 - radical) / 2
                power = root ** (2 * ell)
                exact_formula = 2 * ell / radical * power / (1 - power)
                if not decimal(interval.lower) <= exact_formula <= decimal(interval.upper):
                    raise ArithmeticError("Resolvent series and radical formula disagree")
            resolvent.append({"segment_vertices": ell, "shift": str(shift),
                              "interval": interval.to_data(), "moment_order": steps,
                              "closed_form_80_digit": str(exact_formula)})
    many = connected_cut_interaction(range(32))
    radius = 16
    local = many.local(radius)
    return {"heat_interactions": heat, "resolvent_interactions": resolvent,
            "higher_interaction": {"cut_positions": list(range(32)),
                                   "formal_subset_terms": 2 ** 32,
                                   "reduced_interaction_terms": 1,
                                   "span": 31, "sign": 1,
                                   "first_nonzero_radius": radius,
                                   "computed_local_types": len(local.values),
                                   "local_variation": str(local.norm(0)),
                                   "scope": "Exact line-cut identity; no subset enumeration or claim against an optimized conventional derivation."},
            "interpretation": "Heat and resolvent evaluate the same signed interaction element. Resolvent's specialized moment bound comes from the exact line-cut proof."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))
