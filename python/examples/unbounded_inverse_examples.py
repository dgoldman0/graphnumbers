"""Reproduce local certificates for units with unbounded local variation.

The exponential's global unit property and the resolvent obstruction use
the accompanying proofs.  Finite rational calculations below check their
implemented local consequences; local residual success alone is never
promoted to a global inverse.
"""
import argparse
from fractions import Fraction as Q
import json
from pathlib import Path

from graphlocal import CutLineDefect, Finite
from graphlocal.defect_exponential import CutLineExponential
from graphlocal.local_inverse import (
    UncertifiedLocalInverse, local_inverse_certificate, refine_local_inverse)


def run_examples():
    radius, k, tolerance = 2, 1, Q("1e-4")
    parameter = Q(1, 1000)
    value = CutLineExponential(parameter)
    inverse = value.inverse()
    forward = value.approximation_certificate(radius, k, tolerance)
    backward = inverse.approximation_certificate(radius, k, tolerance)
    product = forward.approximation.multiply(backward.approximation)
    identity = Finite.scalar(1).local(radius)
    product_defect = (product.histogram - identity).norm(k)
    assert product_defect <= product.error

    initial = local_inverse_certificate(value, identity, k, source_epsilon=tolerance)
    refined = refine_local_inverse(value, identity, k, epsilon=tolerance,
                                   source_epsilon=tolerance)
    comparison = (refined.approximation.histogram - backward.approximation.histogram).norm(k)
    comparison_bound = refined.approximation.error + backward.approximation.error
    assert refined.approximation.error <= tolerance
    assert comparison <= comparison_bound
    assert refined.truncation_degree > 0

    defect = CutLineDefect()
    resolvent_parameter = Q(1, 16)
    resolvent_source = 1 - defect * resolvent_parameter
    one_ball_identity = Finite.scalar(1).local(1)
    local_certificate = local_inverse_certificate(resolvent_source, one_ball_identity, k)
    local_refined = refine_local_inverse(resolvent_source, one_ball_identity, k,
                                        epsilon=tolerance)
    assert local_certificate.residual_bound == Q(5, 8)
    assert local_refined.approximation.error <= tolerance

    obstruction_radius = 4
    large_identity = Finite.scalar(1).local(obstruction_radius)
    rejection = None
    try:
        local_inverse_certificate(resolvent_source, large_identity, k)
    except UncertifiedLocalInverse as error:
        rejection = str(error)
    assert rejection is not None
    # The proved additive-coordinate character gives value +1 on every
    # half-line atom and -1 on the line atom of T_r(E). Its extension to
    # the local product monoid is a theorem, not inferred from this sum.
    atom_evaluation = 2 * obstruction_radius - 2 * obstruction_radius * (-1)
    source_character_value = 1 - resolvent_parameter * atom_evaluation
    assert atom_evaluation == 16
    assert source_character_value == 0

    return {
        "method": "exact rational certificates and finite rooted-histogram comparisons",
        "proof_scope": (
            "The entire cut-line calculus proves exp(tE)*exp(-tE)=1 globally and "
            "unbounded variation for nonzero t. A proved local character supplies "
            "the larger-radius resolvent obstruction. The computations check "
            "specified local consequences, not those universal assertions."),
        "requested_weighted_error": str(tolerance),
        "exponential_unit": {
            "parameter": str(parameter),
            "exact_variation_formula_for_radius_at_least_two": "exp(4*r*abs(parameter))",
            "global_finite_variation_certificate": None,
            "forward": forward.to_data(),
            "inverse": backward.to_data(),
            "inverse_identity": {
                "observed_weighted_defect": str(product_defect),
                "certified_error_bound": str(product.error),
                "contained": True,
            },
            "initial_local_candidate": initial.to_data(),
            "refined_local_inverse": refined.to_data(),
            "agreement_with_specialized_inverse": {
                "observed_weighted_difference": str(comparison),
                "combined_error_bound": str(comparison_bound),
                "contained": True,
            },
        },
        "local_success_without_global_invertibility": {
            "source": "1-E/16",
            "radius_one": {
                "certificate": local_certificate.to_data(),
                "refinement": local_refined.to_data(),
                "scope": "Only the radius-one weighted local inverse is certified.",
            },
            "radius_four": {
                "unweighted_defect_variation": str(defect.local(4).norm(0) / 16),
                "weighted_candidate_residual": str(defect.local(4).norm(k) / 16),
                "residual_test_rejection": rejection,
                "proved_character_atom_values": {"half_line": "1", "line": "-1"},
                "character_value_of_E": str(atom_evaluation),
                "character_value_of_source": str(source_character_value),
                "conclusion_from_character_theorem": (
                    "The radius-four marginal is a nonunit, hence the source "
                    "has no inverse in the full graph algebra."),
                "distinction": "Residual-test failure alone is inconclusive; the character proves the obstruction.",
            },
        },
        "all_checks_passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(run_examples(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
