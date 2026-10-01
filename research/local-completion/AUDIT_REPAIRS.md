# Audit repair register

1 October 2026. Applies to the local Cartesian graph completion and
`graphnumbers-local` 0.8.1. The historical
[findings](../../reviews/local-completion-2026-09-30/FINDINGS.md) and referee
reports remain unchanged. Their line numbers refer to `65f08bb`.

This is a first repair checkpoint, not a claim that the entire audit backlog
is closed. The priority is sound certificate handling, precise statements,
and accurate descriptions of the evidence before further theory is added.

## Library corrections

| Finding | Disposition | Evidence |
| --- | --- | --- |
| C1 | Sum, scale, product and exponential consumers check source radius, weight and error allowance. Stronger weight certificates are weakened safely. Product/exponential paths also check their used norm bounds. | Malformed custom sources are rejected; valid scalar sources are compared with exact coefficients. |
| C2 | Degree projection rejects above-cap weighted mass exceeding the input error; inverse checks a declared mass against the source certificate. Controlled heat uses the same projection guard. | A misdeclared scaled line is rejected by both consumers; exactly budgeted above-cap noise is accepted, a smaller error budget rejected. |
| C3 | Reconstruction rejects empty or disconnected catalog entries before solving. | The audited disconnected-catalog shape now raises `ValueError`. |
| C4 | Built-in statistic factories normalize arguments before their private caches. Arbitrary custom axes retain identity semantics. | Equivalent positional/keyword calls give the same axis and law; convolution is checked against exact scalar coefficients. |
| C5 | Partial. Exponential majorants honor an increased `max_terms`, removing the hidden 512-term ceiling. Term/search budgets still do not bound elapsed time or total arithmetic work. | `exp(600, max_terms=2000)` encloses an independent high-precision decimal value. Hard isomorphism and generic inverse/exponential work accounting remain open. |
| C6 | Research graph and rooted-type validation raises explicit errors, including under `python -O`. Disconnected types fail before distance signatures are formed. | An optimized-Python subprocess exercises invalid rows and disconnected targets. The reconstruction dual-feasibility and primal checks remain the soundness checks; the cost-equality check is not additional evidence. |
| C7 | `graph()` rejects repeated undirected edges, including reversed duplicates. | Both duplicate forms are rejected. Shrikhande fixtures now supply each undirected edge once. |
| C8 | Scalar controlled heat, exact zero/scalar inverse locals, zero cut exponential locals, and geometric interaction wrappers are accepted. | Exact scalar results and wrapper/raw equivalence are checked. Wrapper equivalence is an API regression, not an independent oracle for the underlying moment algorithm. |

The regressions are in
[test_audit_regressions.py](../../python/tests/test_audit_regressions.py).
The full library suite passes in this environment, including optional numerical
baselines. The new regression module also passes with `python -S`.
[verification.json](../../python/results/verification.json) records the
discovered test IDs, failures, errors and skips; the previous hard-coded
"coverage" list has been removed. A test-method count is not a theorem-coverage
metric. Custom mathematical promises cannot be verified in general: the new
guards reject observable contradictions rather than certify arbitrary oracles.

Reproduce from the repository root:

```sh
PYTHONPATH=python/src python python/tests/run_tests.py --output python/results/verification.json
PYTHONPATH=python/src python -S -m unittest discover -s python/tests -p test_audit_regressions.py
```

## Statements and proofs

| Findings | Disposition |
| --- | --- |
| D1–D9 | Active summaries now state d>=3 for the mixed tree/plane theorem; qualified heat domains and benchmark comparisons; the sequence-space measure criterion; accurate lattice-heat precision; small-catalog reconstruction scope; arrangement sensitivity; no proved strict profile-domain inclusion; reduced divisibility and radius qualifications. |
| G1 | Prime-rescaling rigidity now gets its common nonzero cycle factor from continuity of vertex mass composed with the extension, without assuming injectivity of that extension. |
| G2 | The binomial moment identity now passes to completed elements through continuity of fixed-order local observables and multiplication. |
| G3–G4 | The edit-budget Laplacian-moment bound supplies locally uniform absolute convergence of the heat Taylor series and justifies coefficient comparison. |
| G5 | Trace-norm convergence of the line's relative exponential series identifies the operator trace with stabilized local moments. Cartesian trace factorization agrees with the moment-profile heat functional. |
| G6 | The tree-contour argument now proves that each internal successor permutation is a single cycle, using the bridge property. |
| M1–M10, M12–M14 | Added radius and cut-count restrictions, one-way balance implication, missing weighted-tail factor, distinct retractions, prescribed-link hypothesis, subtraction sign, named character family, factor-two threshold and finite-marginal wording. |
| M15 | The mixed note records the nonzero E²+2P obstruction for regular-face characters at r>=2. The radius-one face is the whole monoid and does not obey that relation; its explicit marginal is 4 times the degree-two star minus 4 times the degree-three star. |
| M11, M17, M18 | Informational: existing realization and buffer bounds remain sufficient and deliberately unoptimized; the characteristic-zero restriction is unnecessary for the noted domain argument. No sharper bound is claimed here. |
| M16 | Still open in this repository. Strict finite-cone inclusion needs a checked encoding of the cited labelled-network result into simple graphs. The plan records the route, not a completed proof. |

These are written mathematical repairs, not machine-checked proofs. The
new radius-one marginal for E²+2P was also compared with exact library
coefficients; the displayed expression itself gives a direct independent
calculation from T1(E)=2z-2z² and T1(P)=2z³-2z⁴ on stars.

## Verification scope and remaining work

| Finding | Correction and remaining task |
| --- | --- |
| E1 | Branching verification separates executed component reductions from proposed stabilization, eigenvalue and cyclic-quotient tests. Removed the unsupported torus claim; original crossing-product radii are 1..3. Planar weighted/table/determinant checks remain explicitly pending. The intrinsic verifier does not execute the sign substitution. The higher-interaction verifier uses a different four-defect fixture from the earlier K4 example. |
| E2 | The inverse example's larger-radius obstruction is identified as a scalar character calculation. Literal breadth-report constants are identified as summary constants, not new measurements. Replace these with executed claim-specific witnesses before counting them as evidence. |
| E3–E4 | Removed headline assertion counts from active summaries and stated dependence/duplication where material. Historical JSON and referee output are retained as records. The original TeX fixture statement still needs revision: pairwise-distinct histograms of 19 graphs do not establish their linear independence. |
| E5 | The runner now derives its inventory from discovered tests. New valid scalar cases use exact coefficients or a decimal exponential reference. Older self-comparisons and interval-overlap checks remain; they are consistency tests, not independent certificates. |
| E6 | Broader independent tests of representation sufficiency, nontrivial tree-factor recovery and cumulant construction remain necessary. The new validation regressions do not close these mechanism-coverage gaps. |

Next evidence work should target specific missing mechanisms, preserve explicit
fixtures and expected outputs, and explain the independent oracle used. Do not
turn repeated assertions, identities implied by earlier identities, or literal
report fields into headline check counts.

The subsequent [foundation checkpoint](RADIUS_ONE_STRUCTURE.md) adds a direct
cone-induction realization proof at radius one, with explicit rational
witnesses and an independently enumerated host-matrix rank. Its
[standalone verifier](verify_foundation_checkpoint.py) imports no library
code and also checks the fragile finite-component cases of the
[deletion theorem](FINITE_DELETION_ARITHMETIC.md). This strengthens the
radius-one and deletion evidence; it does not close the all-radius
representation or tree-factor/cumulant coverage gaps above. The
[positive-span note](POSITIVE_SPAN.md) gives a measure-theoretic proof;
its cycle fixture illustrates local cancellation rather than verifying
the general Jordan argument computationally.

## Literature

| Finding | Disposition |
| --- | --- |
| L1 | Credited Aldous–Lyons' "Product Networks" proposition and rooted independent product in the literature review. The required Cartesian mass-transport argument can be proved directly by two applications of mass transport and Tonelli. |
| L2 | Pending: reconcile the original TeX bibliography and any regenerated paper with the later graph-ring and moment-control references. |
| L3 | Verified, no wording retraction needed. Knill's author PDF explicitly has Section 1.14 on weighted Wiener algebras; the review now points there. The abstract alone did not settle the question. |
| L4–L6 | Pending: published-version metadata, remaining theorem pinpoints, and targeted additional reading. No blanket claim of full-text verification is made. |

Primary passages checked in this pass: Aldous–Lyons, Proposition 2.2
(involution invariance and unimodularity), their product-network definition
and proposition, and Knill, Section 1.14. The older Aldous–Lyons author
manuscript and the arXiv rendering use different numbering for the product
proposition; the review records both rather than silently conflating them.
