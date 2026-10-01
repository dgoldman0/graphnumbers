# Referee report: geometric arithmetic and nonspectral calculus

- **Assigned:** `research/local-completion/GEOMETRIC_ARITHMETIC.md`, `NONSPECTRAL_CALCULUS.md`, `NONSPECTRAL_INTRINSIC.md`, cross-checked against `python/src/graphlocal/nonspectral.py`, `inverse.py`, `python/examples/arithmetic_geometry_verification.py` and `certified_arithmetic.py`.
- **Revision reviewed:** `d6aba96`.
- **Evidence:** `gtools.py`, `rs1.py`, `rs2.py`, `rs3.py`, `rs4.py` (with `rs4.out`), `nc1.py`, `sym1.py`, `flat.py` and `add1.py`, all in this folder.
- **Brief:** [BRIEFS.md](../BRIEFS.md#m7-arith-nonspectral).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder. See the [folder README](../README.md#provenance) for what was not preserved.

---

## Referee report: GEOMETRIC_ARITHMETIC.md, NONSPECTRAL_CALCULUS.md, NONSPECTRAL_INTRINSIC.md

**Bottom line:** I found no ERROR and no substantive GAP in the three notes. Every theorem I checked is correct and every number I recomputed matches. The findings below are all MINOR: wording, a missing constant, summary phrasing, and verifier quality. All my scratch code is in this folder (gtools.py, rs1.py, rs4.py, nc1.py, sym1.py, flat.py, add1.py). The repo was not modified; I ran its verifiers from a scratch copy with `-B`.

### Coverage (checked and correct)

**GEOMETRIC_ARITHMETIC**
- **Link facts (1)–(2), (5):** Link additivity, Σc_i ≤ deg, and continuity of L_J are correct. A randomized check over 643 product roots found 0 failures.
- **Retract theorem (7)–(10), key claim (a):** Correct in the multi-generator version. Link-component counts add under ⋆₁, L_J(U_i)=z_i, and P_G=S_G∘L_J is a continuous unital homomorphism that is the identity on B_G. The seminorm bounds give the S_m topology.
- **A^∞(D̄^m) identification:** Standard and correct.
- **Units, spectrum, divisibility (11)–(14):** Correct.
- **Relative algebraic closure, key claim (b):** Correct. The proof genuinely needs the domain theorem; without it, ε in B[ε]/(ε²) retracting ε↦0 is a counterexample. The retraction's kernel plays no role; only multiplicativity and P_G|_B = id are used.
- **No nth roots of H or 1+H in A_C:** Correct. The 1+H obstruction really is the weighted/boundary one: the radius-1 formal square root Σ binom(1/2,j)δ_{cone(jK₁)} is in unweighted ℓ¹ (Σ|·| = 2) but not in the weighted spaces (k=1 partial sums grow like √J).
- **Principal kernel (15)–(17), key claim (c):** Correct. Closedness holds because the ideal equals ker D|_B. The integral quotient lies in A^∞ and keeps real coefficients, and division is continuous.
- **(18)–(19):** Correct. y vanishes at (1,−2/3), and a symbolic three-generator decomposition reconstructs exactly.
- **Section 5 rook–Shrikhande, key claim (d):** All correct, recomputed independently.
  - Exact characteristic polynomials: (λ−6)(λ−2)⁶(λ+2)⁹ for adjacency and λ(λ−4)⁶(λ−8)⁹ for the Laplacian, for both graphs; A²=4I+2J for both.
  - Links are 2K₃ (rook) and C₆ (Shrikhande); root K₄ counts are 2 and 0. The radius-1 balls are cone(2K₃) and cone(C₆); the radius-2 (and radius-3) balls are the whole rooted graphs.
  - With T₁ powers formed literally from product balls (not assuming link additivity), T₁(X)^{*n} for n ≤ 7 is Σ C(n,a)(−1)^b δ_{cone(2aK₃⊔bC₆)}, with ℓ¹ norm 2ⁿ and ball size 6n+1.
  - For t ∈ {1/4, −1/3, 9/20, −49/100}, (1+tX)·(partial inverse) = 1 − (−t)^{N+1}X^{N+1} exactly. Partial ℓ¹, p_{1,1} and p_{1,2} match Σ(2|t|)ⁿ, Σ(2|t|)ⁿ(6n+1) and Σ(2|t|)ⁿ(6n+1)².
  - Link characters vanish on 1+tX at t = ±1/2.
  - At radius 2, the five products R, S, R², RS, S² each have a single 2-ball WL class (they are vertex-transitive), and their radius-1 truncations are pairwise distinct.
  - Conclusion: σ(X) = closed disk of radius 2, invertible ⇔ |t|<1/2, and variation 1/(1−2|t|) for r≥1.
- **Section 6:** (27)–(30) are correct, given the CHARACTERS unit criterion that another reviewer is checking. (31), (33) and (34) are correct; the Stirling formula and the tail sum were checked symbolically.

**NONSPECTRAL_CALCULUS**
- **(1)–(5):** Clique additivity and bounds, the joint-distribution homomorphism, and bound (4) are correct.
- **(6)–(9):** Transforms, stability bounds and the reciprocal bound are correct.
- **(10)–(13):** Higher Leibniz rules and the jet homomorphism (the α! factor is needed) are correct.
- **(14)–(20):** Reciprocal moments, exponentials and cumulants are correct, checked with sympy, including κ_α(e^{sX}) = sM_α(X) and κ₁₁.
- **Seven-vertex example (21)–(22):** Recomputed. The table matches, F_{G−J} = z²(w−1)(1−z), M₁₁ = 6 vs 7, and the covariance difference is −1/7. M₁₁((U(G)−U(J))Y) = −1/7 under four product backgrounds.
- **(23)–(26):** Correct; the series for μ_{c₄}(U_t⁻¹) agrees with the actual multinomial inverse, and M₁–M₃ match.
- **Flat-function element (27):** Correct. An FFT check gives c₀ = e⁻¹, imaginary parts ~1e−17, and Σc_n, Σnc_n, Σn²c_n ≈ 0. Higher moments are lost to float noise, but the analytic argument is sound.
- **Finite reconstruction (28)–(29):** Correct.
- **Paw witness:** Weights 0 and 1, so the weighted array is unbalanced. Correct.

**NONSPECTRAL_INTRINSIC**
- **Point derivations:** δ_F are continuous point derivations with |δ_F| ≤ p_{1,1}. Correct.
- **Independence theorem:** Correct. The 31×31 matrix δ_F(cone F′) over connected F with ≤5 vertices has full rank, is block-triangular, and has diagonal 1+u(F).
- **Cotangent space:** Infinite-dimensional, as claimed. Correct.
- **Diagonal-derivation theorem:** Correct.
- **Balance lemma and both classifications:** Correct, including the gluing construction for n > r.
- **Witnesses:** P₃ gives c_{K₁} = (1,2,1), and cone(F) plus a leaf gives apex 1 vs leaf 0. Correct.

### Findings, most severe first (all MINOR)

1. **GEOMETRIC_ARITHMETIC.md:127–129** says "The one-variable case with G₁=K₂ is the earlier normalized-edge retract." The subalgebra and S_G do coincide, but the retraction P_G = S∘L_{K₁}, which counts isolated vertices of the link, differs from the earlier S∘D, which uses degree.
   - Evidence: for T = K₃/3, D(T) = z², so S∘D(T) = H²; but L_{K₁}(T) = 1, so P_G(T) = 1.
   - So there are at least two distinct continuous retractions onto the closure of R[H]. This is harmless for the theorems, since any retraction works, but the sentence conflates them.

2. **GEOMETRIC_ARITHMETIC.md:386–392:** "With m=rk, the tail in (30) is exactly (32)." Formula (32) equals Σ_{n>N}(1+nD)^{rk}qⁿ, which I verified symbolically. The tail in (30) carries an extra factor (r+1)^k, which (32) omits.

3. **research/local-completion/README.md:65 and README.md:141–142:** "The inverse's local variation is exactly 1/(1−2|t|)." This holds for r ≥ 1, which the note states at line 319. At r=0 the variation is |V(Y_t⁻¹)| = 1.

4. **GEOMETRIC_ARITHMETIC.md:71–72:** "Vertex transitivity is sufficient." It is sufficient only if the common link is connected and the links are distinct across i. C₄ = K₂□K₂ and the Petersen graph are vertex-transitive with disconnected links 2K₁ and 3K₁.

5. **GEOMETRIC_ARITHMETIC.md:249:** "Their difference is 3(T−H²)/4." That is y−x; x−y = −3(T−H²)/4. Trivial sign/ordering point.

6. **Verifier evidence is thinner than the "1,671 exact checks" headline suggests** (README lines 85–88 and 252; python/examples/arithmetic_geometry_verification.py). The count reproduces exactly, and the README correctly says the universal claims rest on proofs. Breakdown:
   - 626 checks are a re-run of `cospectral_geometry.verify_geometry`, the earlier note's fixture.
   - 882 are link checks on 441 product roots from 7 small fixtures.
   - The 76 rook–Shrikhande inverse checks run in a formal polynomial ring in commuting symbols R, S and never build a graph. So "exact binomial total variation" does not test the disjoint-support claim it stands for.
   - The tail check (q^{N+1}/(1−q) = q^{N+1} + q·tail) and the "bounded link character annihilates endpoint family" check (1 + t(z²−w) = 0 with hard-coded scalars) are tautologies.
   - The degree-kernel checks only exercise polynomial division, not the analytic quotient or closedness.
   - In certified_arithmetic.py, `inverse.variation_bound == 1/(1-q)` restates the constructor's own formula.
   - The only genuine graph-level inverse test is certified_arithmetic.py at r=1, t=1/1024, truncation degree 2; it passes (36 checks).
   - My independent computations above cover what these checks do not.

7. **README "determines … divisibility" (research README:58):** The note reduces ambient divisibility to divisibility in A^∞(D̄^m) and itself says zeros are not a general test. "Determines" is acceptable only in that reduced sense.

**Calibration:** I am confident in the correctness of key claims (a)–(f). The only results that rest on other notes are the domain theorem (MULTIPLICATION_AND_UNITS), used by (b) and its complex version, and the unweighted-ℓ¹ unit criterion (CHARACTERS_AND_INVERSION), used by GEOMETRIC_ARITHMETIC (27). Their correctness should be confirmed by the reviewer of those notes.

---

*Lead reviewer's note:* finding 2 is a documentation slip only. The library's `inverse.py` (`_power_majorant`) includes the (r+1)^k factor, so its certified tail is the correct bound (30). The two dependencies named under "Calibration" were refereed separately ([m2-units](../m2-units/REPORT.md), [m1-representation](../m1-representation/REPORT.md)) and found correct.
