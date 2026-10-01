# Referee report: quantitative approximation, intrinsic structure, reflection, reconstruction tool

- **Assigned:** `research/local-completion/QUANTITATIVE_APPROXIMATION.md`, `INTRINSIC_GRAPH_STRUCTURE.md`, `REFLECTION_EXTENSION.md`, the tool `reconstruct_local.py`, and what `verify_spectral_approximation.py`, `verify_intrinsic_structure.py` and `verify_reflection_extension.py` test.
- **Revision reviewed:** `d6aba96`.
- **Evidence:** `t1_line.py` … `t11_bounds_small.py`, `indep.py`, `refl.py`, the adversarial inputs `adv1.json` … `adv7.json`, `line_r1_n5.json`, and the output `t3_out.txt`, all in this folder.
- **Brief:** [BRIEFS.md](../BRIEFS.md#m3-approx-intrinsic).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder. The report was delivered twice with identical content. See the [folder README](../README.md#provenance) for what was not preserved.

---

## Referee report: QUANTITATIVE_APPROXIMATION.md, INTRINSIC_GRAPH_STRUCTURE.md, REFLECTION_EXTENSION.md, reconstruct_local.py

**Bottom line:** I found no ERROR-level problem. Every theorem and every quantitative claim in the three notes checked out, both by reading the proofs line by line and by recomputing with my own code. The findings below are one small proof gap, some overstated descriptions of the verifiers and the tool, and robustness/practicality limits in the tool.

How I checked: I wrote my own networkx-based rooted-ball, histogram and parity-filter code and used scipy/HiGHS for linear programs. None of the repo package or verifier code was used as an oracle. Scratch work is in this folder. The repo was not modified.

### Coverage (checked and found correct)

**QUANTITATIVE_APPROXIMATION.md**
- **§1:** tail estimate (1), cutoff D in (2), realization size M=(D+1)b, and the r=0 case.
- **§2:** cofactor/Hadamard bound (3), mass bound (4), p_{s,k}(x) ≤ M^k C(x), and the Q-bound.
- **§3:** primal (5) and dual (6), the basis-enumeration completeness argument (complementary slackness, extension to an active basis), and (7).
- **§4 identity (8):** recomputed for R=1..7. C_{2R+2}/(2R+2) gives the same histogram; C_{2R+1} does not.
- **§4 table, uniqueness:** radius-R histograms of all connected graphs on ≤2R+1 vertices have full column rank for R=1,2,3 (4, 31, 996 graphs).
- **§4 table, costs:** independent LP minima are 5, 9, 13 with negative mass 2, 4, 6, i.e. 4R+1 and 2R. With 2R+2 vertices allowed the minimum is 1, including on all 143 connected graphs of ≤6 vertices at R=2. The "sharp" tradeoffs are genuinely sharp in the stated sense.
- **§5:** (9), (10), divergence of both positive and negative mass when c∉ℓ¹, and the general unbounded-variation statement. ‖T_{N_m}J(c)‖₁ = 2Σ|c_j| recomputed with tail terms included.
- **Recorded certificates:** the example (cost 9, negative mass 4) and all four certificates in `spectral_approximation_results.json` re-verified independently: primal equality, dual feasibility on every catalog graph, equal objectives, and catalog completeness against the graph atlas.

**reconstruct_local.py**
- The certificate logic is sound and in exact Fractions: Hc=b is checked on all rows, |Hᵀf| ≤ n_j is checked on every column, and weak duality then gives optimality.
- The out-of-span separator f = e_i − H[i,J]A⁻¹ correctly annihilates every column.
- `catalog()` matches the networkx atlas for n ≤ 6 under every degree cap (bijection checked).
- On 49 random rational targets the tool returned "optimal" 39 times, each time equal to the HiGHS optimum; the other 10 hit the search budget. I found no case of a false "optimal" or a false "infeasible" on valid input.

**INTRINSIC_GRAPH_STRUCTURE.md**
- Filtration theorem, (1), and ker p_{r,k}=N_r.
- Strictness example, recomputed for r=1..5.
- Bounds (2) and (3): 720 random cases, no failures.
- Diagonal rigidity. Injectivity is genuinely needed: X↦I(X)·1 is continuous, unital, diagonal and not the identity.
- Bridge lemma, primality of cycles, and the finite-prime theorem.
- Sign-change discontinuity: (6) recomputed for n=5..12 and k ≤ 4; V(θ(Z_n))=2 confirmed.
- Unit-based translation obstruction, and the §6 positivity counterexamples.

**REFLECTION_EXTENSION.md**
- F is well defined using degree parity only (no factorization needed); (2) and (3) hold. Multiplicativity checked on 150 random pairs.
- Continuity (4): the (r+1)-ball determines the sign and the r-ball of QG (checked for r=0..3).
- R and O: FR=id, FO=−id, and locality of both for r ≥ 1.
- Bounds (4), (6), (8): 360 random signed checks with no failures. The O bound 3^{k+1} is attained.
- Positivity of R and O, kernel P₃+K₁, and the decomposition ker F ⊕ A.
- A_par: F restricts to a continuous involutive automorphism, and A_par is proper.
- §4 theorem: formula (10) holds for n ≥ 3 and the star coefficients for n ≥ 4.
- Lifting criterion (11)–(13) and its inverse formula; the witness j for U.
- D_△: multiplicativity checked on 200 random pairs.
- θ, (14) and (15): the cube-ball coefficient −2(2n−1)/(2n+1) recomputed for n=5..9. It fails at n=4, where the prism is Q₃, which is consistent with the stated n ≥ 6.

### Findings (most severe first)

1. **GAP (minor) — INTRINSIC_GRAPH_STRUCTURE.md:171-177.**
   - *Claim:* "such an automorphism extends continuously to A only when every λ_P=1."
   - *Problem:* the diagonal theorem it cites assumes Φ is injective on A. Injectivity on A₀ does not pass to a continuous extension.
   - *Fix:* continuity alone gives c_{C_m}=c_{C_n} for all m,n > 2R+1, because U(C_m)−U(C_n) ∈ N_R. So c = λ_{C_n} ≠ 0 and the proof goes through. The conclusion is true; the stated justification is incomplete.

2. **OVERCLAIM (practicality) — README.md:154-156 ("rational reconstruction tool with exact optimality certificates"); QUANTITATIVE:150-154.**
   - The search runs over all C(N, rank) bases with an exact inverse for each, which is exponential.
   - With the default budget, the simplest radius-1 line target ([6,1,1]) on the complete ≤5-vertex catalog (31 graphs) returns `budget_exhausted` after about 3 minutes. The optimum is 1, and the dual certificate f = δ(centered P₃) is trivial.
   - With a budget of 20,000, every rank-deficient radius-1 trial I ran exhausted the budget (≤5 vertices: 4/4; ≤6 vertices, degree ≤3: 3/5; all ≤6 vertices: 1/1).
   - All four recorded certificates come from full-rank or tiny catalogs (search_work 2, 16, 2, 136).
   - Correctness is unaffected, and budget exhaustion is reported honestly.

3. **MINOR — tool robustness.**
   - **Validation by `assert` only:** input validation relies on asserts (verify_local_algebra.py:27-32). Under `python -O`, malformed rows are accepted: asymmetric [2,0] or a self-loop [3,1] is reported as `outside_catalog_span` with a "separator". I reproduced this.
   - **Disconnected target:** a disconnected target ([0,0]) crashes with an uncaught KeyError (verify_local_algebra.py:192) before the intended ValueError check (reconstruct_local.py:117-121) is reached.
   - **Vacuous check:** the `cost != dual_value` check (line 184) holds by construction for any basis and sign choice. Soundness comes from the dual-feasibility check (lines 173-175) and primal-equality check (line 184), which are correct.

4. **OVERCLAIM (minor) — descriptions of what the verifiers check.**
   - **Sign-change witness:** INTRINSIC:312-316 and README:291-295 say the verifier checks the "exact sign-change witness". The image check (verify_intrinsic_structure.py:86-89) only computes n/n ± m/m; it never applies θ. Only the input error (6) is actually computed.
   - **"Independently evaluated" certificates:** README:283-285 uses this phrase, but the re-check reuses the same `catalog()`, `hist()` and isomorphism code. My own independent check does confirm them.
   - **Check count:** the top-level README.md:53-56 credits all 335 checks to the quantitative results. Only 140 belong to that note; 195 are Fourier and power-growth checks for CHARACTERS_AND_INVERSION.

5. **MINOR — tautological checks.**
   - verify_spectral_approximation.py:142 checks the arithmetic identity (2r+1)+2r == 4r+1.
   - Lines 130-131 re-derive the solver's own negative-mass formula.
   - Lines 83-84 are trivially true: 1−2ζ ≠ 0 for a fourth root of unity ζ, and 1−2·½ = 0.

6. **MINOR (understatement / inconsistency) — INTRINSIC:37-38.**
   - *Claim:* equality P_fin = P_loc "has not been established."
   - *What follows:* normalizing positive approximants shows that mass-one elements of P_fin are sofic laws, and for bounded degree, membership in P_fin is equivalent to soficity.
   - *Consequence:* the negative Aldous–Lyons resolution cited in the project's own LITERATURE_REVIEW.md:224-227 (Bowen–Chapman–Lubotzky–Vidick) then gives P_fin ⊊ P_loc. This holds modulo the standard encoding of labelled networks as graphs, which I have not verified.
   - The note and the README ("either positive cone") should say so. No proof depends on this.

No off-by-one, sign, quantifier, A₀-versus-completion, or real-versus-complex errors were found. The stated thresholds (n ≥ 5, n ≥ 6, r ≥ 1) are all valid, and several are conservative.
