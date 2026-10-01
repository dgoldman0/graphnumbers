# Referee report: unbounded-variation inversion and effective local inversion

- **Assigned:** `research/local-completion/UNBOUNDED_VARIATION_INVERSION.md`, `EFFECTIVE_LOCAL_INVERSION.md`, cross-checked against `python/src/graphlocal/local_inverse.py`, `defect_exponential.py`, `python/examples/unbounded_inverse_verification.py`, `unbounded_inverse_examples.py` and the related tests.
- **Revision reviewed:** `d6aba96`.
- **Evidence:** `indep.py` (independent canonical form and products), `powers.py`, `additivity.py`, `chars.py`, `chars2.py`, `residual.py`, `residual2.py`, `tail.py`, `r1.py`, `r4.py`, `test_canon.py` and the run logs `run_r2_m7.txt`, `run_r2_n4.txt`, `run_r3_n3.txt`, all in this folder.
- **Brief:** [BRIEFS.md](../BRIEFS.md#m8-unbounded-inversion).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder. See the [folder README](../README.md#provenance) for what was not preserved.

---

**Referee report: UNBOUNDED_VARIATION_INVERSION.md and EFFECTIVE_LOCAL_INVERSION.md**

**Verdict:** I found no ERROR and no GAP in either note. Claims (a)–(f) all hold: I checked every proof line by line and recomputed the key numbers with my own code, which shares nothing with the repo. What I found is minor: verifier scripts that test less than described, one README sentence that overstates a script, and one equation that is false at r=1 if read out of context. The repo is unmodified (`git status` is clean). Scratch work is in this folder.

**How I checked.** I wrote my own graph builder, Cartesian products and induced r-balls. I also wrote an exact rooted canonical form by individualization–refinement. It agreed with networkx VF2 on 32,131 random pairs and was invariant under 300 random relabellings. I computed T_r(E^n) by expanding (P_m − C_m)^{□n} into products of explicit paths and cycles, P_m^k □ C_m^{n−k}, with m ≥ 2r+2.

**Coverage: results checked and found correct**

*UNBOUNDED_VARIATION_INVERSION.md*
- **(1), T_r(E) and ‖·‖₁ = 4r:** recomputed from P_m − C_m at r = 1, 2, 3, 4. At m = 2r+1 = 5 the formula fails (a C₅ type appears), so the m ≥ 2r+2 threshold is right.
- **(2)–(8), additivity of d, Q, N, sphere-log and c_j on all of M_r (r ≥ 2):** the proof is correct. I independently checked it on 3,000 random non-path ball pairs at r = 2, 3 and 400 at r = 4, with zero failures. These include triangles and chorded 4-cycles; N < 0 does occur, as the note says.
- **(9)–(10), atom coordinates are unit vectors:** correct.
- **Claim (a), (17): ‖T_r Σ aₙEⁿ‖₁ = Σ|aₙ|(4r)ⁿ for r ≥ 2.** Confirmed from explicit graphs:
  - r = 2, n ≤ 4: 1, 3, 6, 10, 15 types; ℓ¹ = 8ⁿ.
  - r = 3, n ≤ 3: 1, 4, 10, 20 types; ℓ¹ = 12ⁿ.
  - r = 4, n ≤ 2: 15 types at n = 2; ℓ¹ = 256.
  - Every coefficient equals the multinomial × 2ⁿ(−r)^{e_L}, and my coordinates return the exponent tuple for each type.
  - The supports of different powers are pairwise disjoint.
  - Also correct for odd m = 7.
- **(11)–(14), phase characters:** χ(E) = 2r(u−v). Evaluated on the explicit histograms, the prescribed λ is hit exactly, including on the boundary |λ| = 4r, and χ(Eⁿ) = χ(E)ⁿ. Multiplicativity also holds on products of random non-path graphs (relative error about 1e−13).
- **(15)–(16) and (18):** local spectrum is the closed disk of radius 4r; σ(E) = ℂ; 1 − tE is a nonunit for every t ≠ 0; the resolvent norm is 1/(1−4r|t|).
- **(19)–(24), closure of ℂ[E] ≅ 𝒪(ℂ) as topological algebras:** the upper and lower estimates, closedness, injectivity and (23) are all correct.
- **Claim (c), (25):** f(E) has bounded variation only if f is constant. The scalar-intersection statement correctly uses the criterion in REPRESENTATION_THEOREM §4 (a finite signed measure exists iff sup_r ‖T_r‖₁ < ∞).
- **Claim (d), (26)–(28):** the obstruction comes from continuous characters χ_{r,θ} defined on all of A_ℂ, with |χ(X)| ≤ ‖T_rX‖₁ ≤ p_{r,1}(X). This works because the coordinates are additive on the whole monoid M_r, not only on products of atoms. No retraction is used, and §9 explicitly disclaims one. Real versus complex is handled correctly: zeros are taken in ℂ, so for example 1 + E² is a nonunit of the real algebra.
- **Claim (e), (29)–(30), divisibility:** correct. The bound |g/f| ≤ ‖T_rY‖₁ on |λ| ≤ 4r makes every singularity of g/f removable, and the domain theorem gives uniqueness of the quotient.
- **Claim (b), (31)–(32):** ‖T_r exp(±tE)‖₁ = e^{4r|t|} for r ≥ 2; σ(exp tE) = ℂ∖{0}.
- **(33)–(35), factorial tail certificate:** checked against high-precision sums on 400 random (r, k, t, N) cases. No violations; the worst ratio of true tail to bound was 0.99986.

*EFFECTIVE_LOCAL_INVERSION.md*
- **(4) residual certificate, (5)–(10) refinement and its measured-residual variant, (11) comparison bound:** the algebra is correct. I tested all three on 2,556 random instances in ℓ¹(ℕ, (n+1)^k), with no violations; the worst observed ratio was 0.84 of the bound.
- **§4, effective inverse theorem (claim (f)):** the termination proof via (15) is correct; q_{s,j} converges to the true residual.
  - The algorithm needs no modulus beyond the error bounds the input name already supplies.
  - Approximants need not agree across radii and weights. The exact targets do, by uniqueness and because truncation is a homomorphism, exactly as §5 says.
  - It is a plain Banach-algebra computability fact with no complexity bound, which the note itself acknowledges.
- **§6, (16) and the X_N example:** correct.
- **Example numbers, recomputed from explicit graphs:**

  | Quantity | Value |
  |---|---|
  | r = 1, residual of 1 − E/16 | 5/8 (inverse-norm bound 8/3) |
  | r = 2, t = 1/1000, residual | 17169/500000 (from p_{2,1}(E) = 34, p_{2,1}(E²) = 676) |
  | r = 4, p_{4,1}(E)/16 | 31/4 |
  | r = 4, character with u = 1, v = −1 on T₄(E) | χ(E) = 16, so χ(1 − E/16) = 0 |

- **Implementation:** `local_inverse.py` matches (3), (4), (6) and the δ₁MM₁ + Bq₁^{N+1}/(1−q₁) variant. The `defect_exponential.py` tail matches (34)–(35), and `norm_bound` is a valid upper bound (Stirling/Touchard formula and the `exp_bracket` upper end both checked).
- **Repo checks:** the verifier (3,780 checks, count confirmed), the examples script and the 12 unit tests all pass.
- **READMEs:** the top-level, research and python READMEs do not overstate the mathematics. They say zeros in ℂ, r ≥ 2 for exact variation, and a promise domain for computability.

**Findings (most severe first)**

1. **OVERCLAIM (minor)** — `research/local-completion/README.md:266-267` says the examples script checks "a local success with a larger-radius obstruction". In `python/examples/unbounded_inverse_examples.py:50-64`, the radius-4 part only records two things:
   - the failure of the δ_e candidate, which the script itself calls inconclusive;
   - hard-coded arithmetic, `2*4 - 2*4*(-1) == 16` and `1 - 16/16 == 0`.

   No coordinates or character are evaluated on radius-4 data. The obstruction is true (my computation above), but the script does not test it.

2. **MINOR (verifier tests less than described)** — `python/examples/unbounded_inverse_verification.py`:
   - Line 186, "explicit path atoms equal cut-line marginal", compares two copies of the same path recipe (`defects.py:93-99` and `path_atoms`). Neither derives T_r(E) from P_n − C_n, so the check is tautological.
   - Line 172 contributes 882 of the 3,780 checks, and each one is implied by the additivity check on line 169.
   - Line 258 (39 checks) tests the scalar identity Σⱼ tʲ(−t)^{n−j}/(j!(n−j)!) = δ_{n0}, which has no graph content.
   - Power checks stop at n = 3 (r = 2) and n = 2 (r = 3), and are built in the truncated local monoid, never from finite graphs.

   My explicit-graph computation covers these gaps.

3. **MINOR (presentation)** — `UNBOUNDED_VARIATION_INVERSION.md:196`: equation (21) is displayed without "r ≥ 2". The "Fix r ≥ 2" at line 30 covers it, but §8 (line 317) switches to "every r ≥ 1", so a reader could misapply it. At r = 1, (21) is false:
   - T₁(E) = 2(X − X²) in the root-degree variable X, and the supports of Eⁿ and E^{n+1} overlap (computed overlaps of 1, 2, 3 types for n = 1, 2, 3).
   - For f(z) = z + z²/2, the left side is 8 and the right side is 12.
   - ‖T₁ exp(E)‖₁ = 11.98, versus e⁴ = 54.60. For t < 0 the value does equal e^{4|t|}, because the signs then alternate with degree.
   - The r = 1 resolvent threshold is t ∈ (−1/4, 1/2), not |t| < 1/4.

   None of these r = 1 statements is claimed anywhere; the restriction to r ≥ 2 is necessary and is stated everywhere else.

4. **Not checked** — the Bhatt–Patel "Example 1.4, pp. 137–138" citation: repository.ias.ac.in is blocked by the network proxy.
