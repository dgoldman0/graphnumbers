# Consolidated findings

Every finding from the 15 reports (14 referees plus the lead reviewer), deduplicated and ordered by severity within each category.

- **Source** names the referee folder that reported the finding; the full evidence is in its `REPORT.md`.
- **✓** in the Lead column means the lead reviewer reproduced the finding or checked it against the repository text ([lead/REPORT.md](lead/REPORT.md#4-referee-findings-reproduced-or-confirmed)).
- Line numbers refer to commit `65f08bb`.

## Summary

| Category | Count | Character |
|---|---|---|
| Mathematical errors (false theorem or invalid proof) | **0** | None found in the TeX note or any of the 26 later notes |
| [D. Documentation overclaims](#d-documentation-overclaims) | 9 | Summaries claiming more than the notes prove |
| [C. Library code](#c-library-code) | 8 | Validation holes and bugs; no unsound certificate on built-in inputs |
| [E. Verification evidence](#e-verification-evidence) | 6 | Checks described but absent, tautological, or overcounted |
| [G. Proof gaps](#g-proof-gaps-conclusions-true) | 6 | Missing steps; every conclusion is true |
| [M. Statement-level corrections](#m-statement-level-corrections) | 18 | Missing qualifiers, slips, understatements |
| [L. Literature](#l-literature) | 6 | No fabricated citations |

## D. Documentation overclaims

| ID | Location | Finding | Source | Lead |
|---|---|---|---|---|
| D1 | `README.md:194-195` (also the `65f08bb` commit message) | Says "distinct branching degrees and the planar cut give independent entire coordinates." The branching note counts d = 2 (the line) as a branching degree, but the mixed theorem is proved only for d ≥ 3, and `MIXED_MEDIUM_ARITHMETIC.md:369-374` says E and P are not shown independent. Every face character the method uses satisfies χ(P) = −χ(E)²/2. Fix: "distinct degrees d ≥ 3 and the planar cut". | n1, n3 | ✓ |
| D2 | `README.md:84-86` | "shows why heat return requires degree and global variation control." The proof shows only that a degree bound alone does not give continuity, and the sparse-defect note certifies heat on elements of unbounded variation using an edit budget. Fix: "requires control beyond the topology, such as global variation or an edit budget." | m4 | |
| D3 | `README.md:88-89`; `research/local-completion/EFFECTIVE_ANALYSIS_AND_HEAT.md:248-251` | "Measurements demonstrate the value of retaining Cartesian factorizations." The recorded gain is against the library's own explicit-graph route. Against a dense eigensolve, the factorized path is slower on the prism (6.8 vs 2.3 ms) and faster on the torus (6.9 vs 14.3 ms); no conventional baseline that exploits the product structure was run. "Especially in the irregular example" is contradicted by the data: the torus is the worst case (135× dense, against 87×). | m4 | ✓ |
| D4 | `README.md:32-33`; `research/local-completion/REPRESENTATION_THEOREM.md:351` | "A closed sequence-space family beyond [finite signed measures]." The copy J(ℝ^ℕ) is closed and complemented but contains measure-representable elements J(ℓ¹); the part beyond the measures is not closed. Fix: "a closed complemented copy of ℝ^ℕ whose elements have finite signed measures exactly when c ∈ ℓ¹." | m1 | ✓ |
| D5 | `python/README.md:54,760`; `python/examples/quickstart.py` | Square-lattice heat return at t = 1/2 quoted as "approximately 0.2169320140." That is the midpoint of the certified interval; the true value (e^{−1}I₀(1))² = 0.2169320121, which the interval does contain. Quote the interval, or 0.21693201. | m4 | ✓ |
| D6 | `README.md:154-156`; `research/local-completion/QUANTITATIVE_APPROXIMATION.md:150-154` | Reconstruction tool "with exact optimality certificates." The certificates are sound, but the exhaustive basis search exhausts its default budget on simple targets (for example, the radius-1 line target on the ≤5-vertex catalog after about 3 minutes). All four recorded certificates come from full-rank or tiny catalogs. State the practical scope. | m3 | |
| D7 | `README.md:115` | "orientation-sensitive finite-edge interactions." The note shows the interaction does not depend on orientation; it depends on arrangement (perpendicular versus collinear adjacency, I₄ = 226 vs 218). | m5 | |
| D8 | `research/local-completion/README.md:43-44` | "certified heat on that larger controlled domain." The profile class contains the edit-budget domain and is closed under products, but it is not shown to be strictly larger (`PLANAR_DEFECTS.md:402-404`). | m5 | |
| D9 | `research/local-completion/README.md:58, 65, 310-311`; `README.md:141-142` | Three smaller wording issues:<br>• "Determines … divisibility" holds only in the reduced sense the note gives.<br>• "Local variation exactly 1/(1−2\|t\|)" holds for r ≥ 1; at r = 0 it is 1.<br>• "Not asserted at arbitrary radii" reads as disclaiming tree-product results that are asserted for every r ≥ 2; it should say it concerns products containing P. | m7, n1 | |

## C. Library code

On every path built from the library's own elements, all certified bounds held: thousands of cases against networkx, mpmath, SciPy/HiGHS and closed forms, including adversarial inexact inputs.

| ID | Location (`python/src/graphlocal/`) | Finding | Source | Lead |
|---|---|---|---|---|
| C1 | `elements.py:332-347` | `Exponential.approximate` does not check the weight or radius of the inner approximation. A custom element returning a weight-1 certificate when weight 3 is requested yields a claimed error of 6.4e-4 against a true error of 0.13. `heat_return` and `relative_heat` do check. Needs a custom element that breaks the documented contract. | code-core | ✓ |
| C2 | `inverse.py:109-113`, `controlled.py:83-88` | The degree-cap projection drops mass above the declared cap without checking it against the source error. A custom element that misstates `degree_bound` gets a zero-error certificate (mass 1, against the object's own `.mass` of 4/3) and a wrong `controlled_heat` interval. `heat_return` rejects the same input. | code-advanced | ✓ |
| C3 | `reconstruction.py:105-164` | `reconstruct()` accepts disconnected catalog graphs. The reported cost and negative mass describe catalog coordinates, but the returned element merges components (cost 4 and negative mass 1, element K₂). Reject disconnected entries. | code-core | ✓ |
| C4 | `nonspectral.py:68,84` (with 179-181, 271-273) | `lru_cache` keys on the call form, so `rooted_cliques(3)` and `rooted_cliques(size=3)` are different axis objects. Identical joint laws then compare unequal and `+` raises. Normalize arguments or compare axes by value. | code-advanced | ✓ |
| C5 | `graphs.py:207-230`; `defect_exponential.py:128-135`; `inverse.py:128-131`; `local_inverse.py:142-145`; `elements.py:66,330,335` | Work budgets do not bound work:<br>• The isomorphism budget counts search nodes that each cost Θ(n²m), so about 100 minutes pass before `BudgetExceeded` on hard 160-vertex pairs.<br>• `CutLineExponential` at r = 2, ε = 1e-3 runs 64 s without any budget triggering.<br>• `exp_bracket`'s fixed 512-term cap makes `exp` of inputs with norm above about 510 impossible at any `max_terms`. | code-core, code-advanced | |
| C6 | `research/local-completion/reconstruct_local.py` (validation in `verify_local_algebra.py:27-32,192`) | Input validation is by `assert`:<br>• Under `python -O`, invalid adjacency rows are reported as `outside_catalog_span`.<br>• A disconnected target raises `KeyError` before the intended `ValueError`.<br>• The `cost != dual_value` check is vacuous; soundness rests on the dual-feasibility and primal checks, which are correct. | m3 | ✓ |
| C7 | `graphs.py:68-78` | `graph()` silently merges repeated or reversed edges, whereas `from_networkx` rejects multigraphs. | code-core | |
| C8 | `interaction_moments.py:39-40,105-106`; `interaction_bounds.py:74-75`; `controlled.py:50` | API holes:<br>• `GeometricInteraction` is rejected where it is documented as equivalent.<br>• `controlled_heat(3, 1)` raises `AttributeError`.<br>• `NeumannInverse(0)` and `CutLineExponential(0)` have no exact `local()`. | code-advanced | |

## E. Verification evidence

This is the main systematic weakness. The referees' independent computations covered every gap listed here, and the mathematical claims held, so the issue is how the evidence is described rather than whether the results are true.

| ID | Finding | Source | Lead |
|---|---|---|---|
| E1 | **Notes describe checks the code does not perform.**<br>• `BRANCHING_DEFECTS.md:259-275`: infinite-tree stabilization, eigenvalue checks, and tests of (3) and (9) are absent. Only (2) is compared, and the only cyclic-quotient call expects an exception.<br>• `PLANAR_DEFECTS.md:72-76`: the torus checks are absent; `:142-144` claims r = 1..4 but the tests cover r = 1..3.<br>• `PLANAR_ARITHMETIC.md:292-295`: the weighted formula (5), the invariant table, the determinant 24 and the coefficients (−2r²)^n are not checked.<br>• `INTRINSIC_GRAPH_STRUCTURE.md:312-316` and `research/local-completion/README.md:291-295`: the "exact sign-change witness" never applies θ.<br>• `HIGHER_INTERACTION_GEOMETRY.md:340-343`: the verifier uses a different fixture from the stated K₄ example, though their moments agree. | m5, n2, m3, m6 | ✓ (BRANCHING) |
| E2 | **Scripts described as checking what they hard-code.**<br>• `research/local-completion/README.md:266-267` claims "a local success with a larger-radius obstruction"; `python/examples/unbounded_inverse_examples.py:50-64` only asserts `2*4 - 2*4*(-1) == 16`.<br>• `python/examples/defect_breadth_verification.py:339-345` writes literal values that `test_defect_breadth_verification.py:26` then asserts. | m8, n2 | ✓ (README) |
| E3 | **Headline check counts include tautological, implied or duplicated checks.**<br>• `verify_spectral_approximation.py:83-84` (1−2u ≠ 0 for u = ±1, ±i; 1 − 2·½ = 0), `:130-131`, and `:142` ((2r+1)+2r == 4r+1).<br>• `unbounded_inverse_verification.py:186` compares two copies of one recipe; 882 of its 3,780 checks (`:172`) are implied by `:169`; and `:258` is 39 scalar-identity checks.<br>• `arithmetic_geometry_verification.py`: 626 of its 1,671 checks re-run the cospectral verifier, and 76 rook–Shrikhande checks run in a formal ring without building graphs. Its tail and character checks are tautologies, and `certified_arithmetic.py` restates a constructor formula.<br>• `verify_representation.py:245-254`: transport balance holds on every finite graph by construction.<br>• `BRANCHING_DEFECTS.md:245-249`: the 3,763 histogram identities follow from the 942 component identities, and the "independent" verifier imports library internals. | m2, m3, m8, m7, m1, m5 | ✓ (representation) |
| E4 | **Counts attributed or scoped inaccurately.**<br>• `README.md:53-56` credits all 335 checks to the quantitative note; only 140 belong to it.<br>• The `.tex` note's "separation was checked for 19 graph types" means pairwise-distinct histograms, not linear independence.<br>• "441 rooted-product fixtures" / "arbitrary rooted fixtures": 882 of 1,061 checks come from 7 radius-2 graphs whose balls are the whole graph. | m3, m1, n3 | |
| E5 | **Tests that compare the implementation with itself, or only check that two library intervals overlap.**<br>• `test_prepared_interactions.py:63-71`, `test_defects.py:110-119`, `test_interaction_bounds.py:126-128`, `test_controlled_branching.py:153-156`.<br>• `test_local_inverse.py:50,99,111`, `test_defect_exponential.py:71,96`, `test_medium_defects.py:71,98-99`.<br>• `unbounded_inverse_verification.py:123-135` re-implements the library's tail formula.<br>• The "coverage" list in `run_tests.py:20-47` is hard-coded text. | code-core, code-advanced, n1, n2 | |
| E6 | **Thin coverage of key mechanisms.**<br>• Isomorphism is tested on 4-vertex pairs, and the grid test checks vertex count only.<br>• `NeumannInverse` is tested at r = 1 only, and the locality test uses ≤4-vertex graphs.<br>• The multiplication separation regression stops at 4 vertices.<br>• Representation formula (1) is not tested as written, and sufficiency is untested.<br>• The branching tree-factor lemma and the K_F construction are untested. | code-core, code-advanced, m4, m2, m1, n1 | |

## G. Proof gaps (conclusions true)

| ID | Location (`research/local-completion/`) | Missing step | Source |
|---|---|---|---|
| G1 | `INTRINSIC_GRAPH_STRUCTURE.md:171-177` | Cites the diagonal theorem, which assumes injectivity on the completion. Continuity alone gives the needed constant, because U(C_m) − U(C_n) ∈ N_R. | m3 |
| G2 | `SPARSE_DEFECTS_AND_RELATIVE_HEAT.md:313-315` | Extending the binomial identity to completed elements needs continuity of the fixed-D moment functionals and of multiplication. | m4 |
| G3 | `DEFECT_INTERACTIONS.md:226` | Needs H_t to equal its Laplacian-moment series on the class; this follows from \|δ_m\| ≤ 2qm(2D)^{m−1}, which is not stated. | m4 |
| G4 | `BRANCHING_ARITHMETIC.md`, (33)→(34) | Uses H_t(X) = Σ(−t)^mΔ_m/m!, which is valid by absolute convergence but not stated. | n1 |
| G5 | `PLANAR_DEFECTS.md:209,220-223` | Identifies the operator trace on ℓ²(ℤ^k) with the local heat functional without the diagonal-sum argument. Section 4's multiplicativity gives the result directly. | m5 |
| G6 | `HIGHER_INTERACTION_GEOMETRY.md:286-289` | Surjectivity onto rotation systems needs one more line: the successor permutation at each internal vertex is a single cycle. | m6 |

## M. Statement-level corrections

| ID | Location (`research/local-completion/`) | Correction | Source |
|---|---|---|---|
| M1 | `BRANCHING_ARITHMETIC.md:144-146` (16), `:196-198` (22), `:332-333` (29) | Add r ≥ 2; each is false at r = 1. | n1 |
| M2 | `UNBOUNDED_VARIATION_INVERSION.md:196` (21) | Add r ≥ 2 to the display; it is false at r = 1. | m8 |
| M3 | `MULTIPLICATION_AND_UNITS.md:423-429` (22) | Add r ≥ 1. | m2 |
| M4 | `INTERACTION_DECAY.md:152-153` | Add k ≥ 2. | m6 |
| M5 | `REPRESENTATION_THEOREM.md:242` | "Exactly" should be "implies"; K_{1,5} is a counterexample to the converse. | m1 |
| M6 | `GEOMETRIC_ARITHMETIC.md:386-392` | (32) omits the (r+1)^k factor of (30). The library uses the correct bound (✓ lead). | m7 |
| M7 | `GEOMETRIC_ARITHMETIC.md:127-129` | Conflates two different continuous retractions onto the closure of ℝ[H]. | m7 |
| M8 | `GEOMETRIC_ARITHMETIC.md:71-72` | Vertex transitivity is not sufficient: C₄ and the Petersen graph are counterexamples. | m7 |
| M9 | `GEOMETRIC_ARITHMETIC.md:249` | The stated difference has the opposite sign. | m7 |
| M10 | `CHARACTERS_AND_INVERSION.md:223-225` | Name the semicharacter family; the phase family does not detect units. | m2 |
| M11 | `REPRESENTATION_THEOREM.md:112-115,163-165,186,297-300` | The bound M(D,r) = (D+1)b(D,r) is loose by at least a factor of D+1, and Step B assumes more than it uses. | m1 |
| M12 | `INTERACTION_DECAY.md:288-289` | Wording is off by a factor of 2: the threshold is twice the number of internal edges. | m6 |
| M13 | `SPARSE_DEFECTS_AND_RELATIVE_HEAT.md:215`; `EFFECTIVE_ANALYSIS_AND_HEAT.md:139` | Each marginal variation is finite (only the supremum is infinite), and "M terms" should be M+1. | m4 |
| M14 | `PLANAR_ARITHMETIC.md:9-10, 180` | Quadratic growth comes from two-dimensionality, not cycles; A should be A_ℂ. | n2 |
| M15 | `MIXED_MEDIUM_ARITHMETIC.md:369-374` | Understatement: no face-character method can separate E and P, since E² + 2P ≠ 0 is annihilated by all of them. | n3 |
| M16 | `INTRINSIC_GRAPH_STRUCTURE.md:37-38` | Understatement: for bounded degree the finite cone corresponds to sofic laws, so the 2024 Bowen–Chapman–Lubotzky–Vidick disproof of the Aldous–Lyons conjecture likely gives P_fin ⊊ P_loc. This depends on encoding labelled networks as simple graphs, which was not verified. | m3 |
| M17 | `MULTIPLICATION_AND_UNITS.md:297` | The characteristic-0 restriction is unnecessary (informational). | m2 |
| M18 | `BRANCHING_ARITHMETIC.md` §1; `PLANAR_ARITHMETIC.md`; `BRANCHING_DEFECTS.md` §4 | Several buffer and stabilization thresholds are sufficient but not sharp (informational). | n1, n2, m5 |

## L. Literature

All 59 cited works exist with correct authors, titles, years and identifiers. Full texts could not be fetched from the review environment.

| ID | Finding | Source |
|---|---|---|
| L1 | `LITERATURE_REVIEW.md:150-166` should credit Aldous–Lyons for the rooted product of unimodular laws and its compatibility with local limits (exact section unverified). | lit-review |
| L2 | The bibliography at `local_graph_completion.tex:300-303` predates the literature review; it omits the finite Cartesian graph ring sources and Kurauskas. | lit-review |
| L3 | `LITERATURE_REVIEW.md:101-103`: "weighted Wiener algebras" is not visible in Knill (2021)'s abstract, which says "Wiener algebra". This may be an overstatement. | lit-review |
| L4 | Nine entries cite only the preprint of a published paper; the list is in the report. | lit-review |
| L5 | Section and theorem pinpoints are unverified because full texts were unreachable. | lit-review, m1, m2, m4, m8 |
| L6 | Suggested additional references: Hammack–Imrich–Klavžar; Lovász; Hora–Obata; Michael (1952); van der Hofstad vol. 2; Backhausz–Szegedy. | lit-review |
