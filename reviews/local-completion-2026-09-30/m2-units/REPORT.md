# Referee report: multiplication, units, characters and identification

- **Assigned:** `research/local-completion/MULTIPLICATION_AND_UNITS.md`, `CHARACTERS_AND_INVERSION.md`, and the mathematical content of `FOLLOWUP_IDENTIFICATION.md` (bibliography accuracy went to the literature referee). Also: what `verify_multiplication.py` tests.
- **Revision reviewed:** `d6aba96`.
- **Evidence:** `gl.py` (independent toolkit) and `t1_coproduct.py` … `t5c.py` in this folder.
- **Brief:** [BRIEFS.md](../BRIEFS.md#m2-units).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder. See the [folder README](../README.md#provenance) for what was not preserved.

---

## Referee report: MULTIPLICATION_AND_UNITS.md, CHARACTERS_AND_INVERSION.md, FOLLOWUP_IDENTIFICATION.md (mathematical content only)

**Bottom line:** I found no ERROR and no GAP. Every theorem in the three notes is correctly proved, and the prose summaries do not overstate what is proved. All findings are MINOR or informational.

**How I checked:** I read every proof line by line. I then wrote my own implementation from scratch, using networkx only for isomorphism matching. It computes balls, Cartesian products, ⋆_r, rooted hom counts, the coloring coproduct, K_F, walk cumulants and formal reciprocals. The code is in this folder (`gl.py`, `t1`–`t5*.py`). I also re-ran the repo's verifiers from a scratch copy: the counts 21,697 and 335 reproduce. The repo is untouched (`git status` is clean).

### Coverage: checked and found correct

**MULTIPLICATION_AND_UNITS.md**
- **(1)–(3), finite factorization, formal associativity:** correct. Independently, |C⋆D| ≥ |C|+|D|−1, root-degree additivity and |C⋆D| ≤ |C||D| held for every product of ball types with at most 5 vertices, at r = 1, 2, 3.
- **§2.1 coalgebra (loop convention, coassociativity, grading, counit) and the hom identity (5):** correct. The bijection between product homomorphisms and (coloring, factor maps) is valid. My own check: 19,415 instances of (5) at r = 1 and 2 (balls up to 5 vertices; patterns up to 4 edges, including parallel edges) and coassociativity on 35 patterns, with 0 failures.
- **§2.2, K_F = log_*χ is additive (7):** 0 failures on 761 products.
- **§2.2, separation:** the Möbius-inversion plus mutual-injection argument is correct.
- **§3 (well-order), part (a):** correct. Walk-cumulant additivity (8)–(10), the well-order argument, strict compatibility (13) and the unit being least all hold. The well-order is exactly what handles infinite support at a fixed radius. Root degree alone would not suffice, since for r ≥ 2 there are infinitely many balls of each root degree.
  - Independent consequence check: ⋆_r is cancellative on all 19, 62 and 73 ball types (up to 5 vertices) at r = 1, 2, 3.
- **Local domain theorem and the completion step:** correct. The completion step (pick a common radius, use compatibility) is valid. The argument works over any integral-domain coefficient ring.
- **§4 unit theorem, part (b):** correct.
  - Necessity holds.
  - Compatibility follows because truncation is a continuous homomorphism; b_0 = 1/V(X) comes from (18).
  - Balance: F_M is local with observation radius R+1, its divergence vanishes at roots of degree above M, and both divergences are dominated by 2C|B_{2R+2}|^{k+1}.
  - Recursion (15) agrees with the geometric series (16) in my code.
  - The sufficiency direction relies on REPRESENTATION_THEOREM.md. I spot-checked its Steps A–C and found nothing wrong, but did not referee it fully.
- **§5 and part (c):** correct.
  - The hypercube ball formula (21) checks out for n ≤ 8, r ≤ 4, and b_r(n) ≤ (n+1)^r for n < 400, r ≤ 11.
  - The induced topology is that of S₊, and the closure of ℝ[H] is the real form of A^∞(D̄).
  - The "units tested in the full completion" claim is properly justified. D is a continuous homomorphism on all of A_loc with D∘S = id, so Φ_z = ev_z∘D (|z| ≤ 1) are characters of the full completion. That gives the "only if" direction; S(1/f) gives the "if" direction.
  - (26) and the K1+K2 example are right. I computed the reciprocal of K1+K2: coefficient (−2)ⁿ on stars at r = 1 and on B₂(Qₙ) at r = 2.
- **§6 and part (d):** correct.
  - q_j is local and additive for j ≤ 2r+1. The threshold is sharp: at j = 2r+2 additivity fails on 3,397 of 3,844 products at r = 2.
  - q_N(C_N) = 2, q_N(C_2N) = 0, and χ_N(X_N) = 0 for N = 3, 5, …, 25. T_R(X_N) = T_R(1) for R < (N−1)/2, and D(X_N) ≡ 1.
  - The radius-1 reciprocal of X_3 has ℓ¹ mass exactly 1 at every even degree, so it diverges, consistent with X_3 being a nonunit.
  - Nonunits are dense and the unit group is not open.
  - Inversion is continuous: (31) is valid because each p_{r,k} is the norm of a Banach-algebra homomorphism T_r: A_loc → E_{r,k}. No general Fréchet theorem is needed.

**CHARACTERS_AND_INVERSION.md (part (e))**
- **Phase characters:** continuous and multiplicative. The sinc-average separation (Fubini plus two dominated-convergence steps) is correct.
- **Semisimplicity:** correct, and it also holds for the real algebra. A complex character restricted to A_ℝ has image ℝ or ℂ, so its kernel is maximal.
- **§2 local spectral results:** correct. Growth lemma (4) (spot-checked on 40 balls), local character theorem, compactness of Σ_r, spectral invariance (6) and inverse-closedness all hold.
- **§3:** correct. The equivalence 1⇔2⇔3⇔4, spectral formula (7) and monotonicity in r all hold. The 1−2H example is right: phase characters separate elements but do not detect units.

**FOLLOWUP_IDENTIFICATION.md, mathematics only (parts (f) and (g))**
- **Generalized-power-series identification:** correct, because every subset of a well-ordered set is artinian and narrow.
- **Weighted Fréchet pieces:** the weights are submultiplicative, each L_r is a Beurling–Fréchet algebra, and A_loc is Arens–Michael. It is not the Arens–Michael envelope of A_0, because component count is discontinuous.
- **Obstructions:** all four are correct.
  - C_n/n → 0 coefficientwise in the series algebra.
  - The full formal ring is local, whereas ker I ≠ ker V.
  - No continuous norm: any continuous seminorm satisfies q ≤ C·p_{R,K}, and z_j is invisible at those radii.
  - The 𝒪(ℂ) comparison is accurate.

**Summaries:** the top-level README (lines 37–56) and research README (lines 108–112, 142–153, 318–329) match what is proved.

### Findings (none above MINOR)

1. **MINOR: two cited verifier checks are tautologies.** CHARACTERS_AND_INVERSION.md:227–231 says the verifier checks "the distinction between separating phases and interior spectral zeros". The corresponding checks (verify_spectral_approximation.py:79–84) never touch graph data:
   - `phase_family_misses_interior_zero` (64 checks) asserts 1−2u ≠ 0 for u ∈ {±1, ±i}.
   - `interior_character_detects_nonunit` asserts 1 − 2·½ = 0.

   Also, `exact_fourier_coefficient_recovery` only tests character orthogonality on (ℤ/4)³, using K_F mod 4 on 5 fixtures; it is not the ℝ^m sinc average used in the proof. Only `exact_phase_character_products` (64) and `cartesian_power_ball_growth` (60) exercise actual algebra. The note does correctly disclaim proof by fixtures.

2. **MINOR: ambiguous family.** CHARACTERS_AND_INVERSION.md:223–225 says "The displayed local family does separate elements and detect all units." The phase family (1) provably does not detect units (§3's 1−2H example). Only the semicharacter family {φ_s∘T_r : s ∈ Σ_r} does both. The sentence should name the family.

3. **MINOR: missing r ≥ 1 in (22).** MULTIPLICATION_AND_UNITS.md:423–429 states (22) "for every finite coefficient sequence" without restricting to r ≥ 1. At r = 0 the left side is |Σc_n|, not Σ|c_n|. The topology conclusion is unaffected, since p_{0,k} ≤ p_{1,1}.

4. **MINOR: regression scope.** `small_ball_separation` in verify_multiplication.py only covers balls of at most 4 vertices with patterns of at most 4 vertices. In my run, walk cumulants plus K_F over patterns with at most 4 edges fail to separate 5-vertex balls: for example, centered P₅ versus rooted C₅ at r = 2 gives only 60 distinct keys for 62 types. This is expected, since the proof uses patterns as large as the ball; using the balls themselves as hom patterns separates all 62. So the regression does not probe the separation mechanism beyond 4 vertices.

5. **Informational: characteristic-zero restriction is unnecessary.** MULTIPLICATION_AND_UNITS.md:297 limits the domain argument to fields of characteristic 0. The order is a property of the monoid, so the leading-term argument works over any integral domain.

6. **Informational: IKS definition not checked.** I could not check FOLLOWUP's "exactly ℝ((M_r)) in IKS Def. 3.4" against the source, because egress to arxiv.org and smertnig.at is blocked. Mathematically the statement is right for Ribenboim's ring ℝ[[M_r, ≤]] with the order (11). The bibliography reviewer should confirm IKS's definition.

**Dependency caveat:** the sufficiency direction of the unit criteria in both notes inherits REPRESENTATION_THEOREM.md, which I spot-checked but did not fully referee.
