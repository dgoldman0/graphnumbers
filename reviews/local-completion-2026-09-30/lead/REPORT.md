# Lead reviewer's checks

The lead reviewer coordinated the 14 referees, did the checks below directly, and independently reproduced or confirmed the referee findings this review relies on. Every command here runs from the repository root. Outputs saved in [`outputs/`](outputs/) were produced at commit `65f08bb` (the head of `main` when this folder was committed); each output file records the commit it ran on.

## 1. Core construction, checked by hand

`research/local-completion/local_graph_completion.tex` (Candidate v0.1) defines the algebra that every later note builds on, so it was checked line by line rather than delegated. All of the following are correct as stated:

- **Product-ball identity (Lemma 1).** Distances add in a Cartesian product, and induced balls preserve distances from their root, so truncating the factors first loses no eligible vertex or edge. Associativity, commutativity, the unit and the size bound |B ⋆_r D| ≤ |B||D| follow.
- **Proposition 2.** Convolution of histograms, submultiplicativity of p_{r,k}, and the monotonicity p_{s,j} ≤ p_{r,k} for s ≤ r, j ≤ k (truncation aggregates signed coefficients before the triangle inequality is applied).
- **Separation (Lemma 3).** At a radius exceeding every diameter, each rooted ball is a whole component.
- **Theorem 4 (completion).** The weighted ℓ¹ convolution algebras E_{r,k} are unital Banach algebras; the closure of T(A_0) in their countable product is a commutative unital Fréchet (locally m-convex) algebra, with the rational span dense.
- **Metric (2).** It is complete and translation-invariant, and |a−b| ≤ d(a·1, b·1) ≤ 2|a−b| on scalars.
- **Observables (Propositions 5–6).** The bounds |e| ≤ p_{1,1}/2 and |i| ≤ p_{1,1}; multiplicativity of v and i; the Leibniz rule for e; motif counts bounded by p_{s,q−1} through the induced s-ball of the root image.
- **Theorem 7 and the table.** Cycle balls are centred paths exactly when n > 2r+1 (at n = 2r+1 the ball is the cycle itself). The lattice ball sizes N_d(r) = Σ_j 2^j C(d,j) C(r,j) give 3, 5, 5, 13, 7, 25 as tabulated.
- **Calculus (Proposition 8 and the directional derivative).** p(exp(x+h) − exp(x) − exp(x)h) ≤ e^{p(x)}(e^{p(h)} − 1 − p(h)).
- **Section 6 counterexamples.** K_n/n³ gives p_{0,1} = n^{-2}, p_{r,1} = n^{-1}, p_{1,2} = 1 and triangle count → 1/6. C_{2n} − 2C_n → 0 while component count stays −1. Non-normability follows from directedness of the seminorms. A convergent sequence of finite graphs is eventually constant.

## 2. Reproducibility

| Check | Script | Output | Result |
|---|---|---|---|
| Regenerate every deterministic recorded result and compare with the committed JSON. Timing and platform keys are ignored; parameters such as `time` are compared. | `regenerate_and_compare.sh` (uses `compare_json.py`) | [regenerate_and_compare.txt](outputs/regenerate_and_compare.txt) | All 18 results identical: 7 research verifiers and 11 example scripts, including `cospectral_geometry` and the new `defect_breadth_verification` |
| Library test suite, with and without NumPy/SciPy | `python/tests/run_tests.py` | [test_suite.txt](outputs/test_suite.txt) | 127/127 pass at `65f08bb` (9 skipped without the optional packages, none with them); 115/115 at `d6aba96` |
| Every ```python block in `python/README.md` | `run_readme_blocks.py` | [readme_blocks.txt](outputs/readme_blocks.txt) | 9/9 run |
| Check counts quoted in READMEs against JSON totals | `check_counts.py` | [check_counts.txt](outputs/check_counts.txt) | Every quoted count matches a recorded total. Whether the counted checks are substantive is a separate question; see [FINDINGS.md](../FINDINGS.md#e-verification-evidence) |
| Committed PDF against its TeX source | `pdf_vs_tex.sh` | [pdf_vs_tex.txt](outputs/pdf_vs_tex.txt) | Consistent: same distinctive passages and the same 8 numbered results. Built 16:12 UTC, before the first commit |

Benchmarks that record wall-clock timings (`heat_benchmark`, `defect_benchmark`, `reuse_benchmark`) were not regenerated; their recorded numbers are discussed in finding D3.

Repository hygiene at `65f08bb`: no caches or build products are tracked, the ignore files cover them, and there are no active git hooks or CI configuration.

## 3. Library code read directly

- **`elements.py`, `Exponential.approximate`.** The series tail z^{n+1}/(n+1)!·1/(1 − z/(n+2)) is a valid bound. The input-error term δ·e^{‖T_rX‖+1} is a valid Lipschitz bound, because ‖exp a − exp b‖ ≤ ‖a − b‖·e^{max(‖a‖,‖b‖)} and δ ≤ 1. Soundness therefore depends only on the inner approximation honouring the requested weight, which is the gap in finding C1.
- **`inverse.py`, `_power_majorant`.** It includes the (r+1)^k factor, so the certified Neumann tail is the correct bound (30) of GEOMETRIC_ARITHMETIC.md. The note's sentence equating it with (32) drops that factor, which is a documentation slip only (finding M6).
- **`medium_defects.py` (new in `65f08bb`).**
  - The regular-tree histogram assigns 2(d−1)^j roots to the type cut at distance j, which is correct by symmetry, and the regular type has coefficient −2Σ_{j<r}(d−1)^j.
  - The finite witnesses have adequate buffers: depth 2r+1 for trees, and a (4r+4)×(4r+3) grid with margin 2r+1.
  - The variations 4Σ_{j<r}(d−1)^j and 4r² follow by counting affected roots; in the lattice |B_{r−1}(a) ∪ B_{r−1}(b)| = 2r².

## 4. Referee findings reproduced or confirmed

| Finding | How | Evidence |
|---|---|---|
| C1 `exp()` trusts a weaker-weight inner certificate | `reproduce_code_findings.py` (c) | claimed error 6.4e-4, actual 0.13 at k = 3 |
| C2 degree-cap projection drops mass silently | `reproduce_code_findings.py` (d) | certified mass 1 and error 0, against `.mass` = 4/3; `heat_return` rejects the same input |
| C3 disconnected catalog graphs in `reconstruct()` | `reproduce_code_findings.py` (a) | cost 4 and negative mass 1, but the returned element is K₂ |
| C4 `lru_cache` axis identity in `nonspectral.py` | `reproduce_code_findings.py` (b) | identical laws compare unequal |
| C6 `reconstruct_local.py` input validation | `reproduce_reconstruct_local.sh` | `KeyError: 1` on a disconnected target; under `python -O`, invalid rows are reported as `outside_catalog_span` |
| D1 README independence sentence (`README.md:194-195`) | read against `MIXED_MEDIUM_ARITHMETIC.md:8,17,369-374` and `BRANCHING_ARITHMETIC.md:248` | the mixed theorem assumes d ≥ 3, and the note itself says E and P are not shown independent |
| D3 benchmark interpretation | `benchmark_timings.py` | prism: factorized 6.8 ms vs dense 2.3 ms; torus: factorized 6.9 ms vs dense 14.3 ms; explicit extraction 135× dense on the torus, 87× on the irregular graph |
| D5 lattice heat value quoted as a midpoint | `lattice_heat_value.py` | closed form 0.216932012066; interval valid; midpoint 1.9e-9 high |
| E1 verification lists that describe absent checks | read against `python/examples/branch_planar_verification.py` | no infinite-tree stabilization, eigenvalue or formula-(9) test; the only cyclic-quotient call expects an exception |
| E2 hard-coded radius-4 "obstruction" | read `python/examples/unbounded_inverse_examples.py:50-64` | `2*4 - 2*4*(-1) == 16`, with a comment conceding the character extension is "not inferred from this sum" |
| E3 tautological transport-balance check | read `research/local-completion/verify_representation.py:245-254` | sums a divergence that is zero on every finite graph |
| D4 "closed family beyond" wording | read `README.md:32-33` and `REPRESENTATION_THEOREM.md:351` | wording as reported |

## 5. Limits of this review

- Universal statements were checked by reading proofs. Finite computations supplement them; they do not replace them.
- Full texts of the cited literature were unavailable (network policy), so section and theorem pinpoints in citations remain unverified.
- No timing benchmark was rerun, and no performance claim was independently measured beyond the cases the code reviewers timed.
