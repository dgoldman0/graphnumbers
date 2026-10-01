# Referee report: planar arithmetic (commit 65f08bb)

- **Assigned:** `research/local-completion/PLANAR_ARITHMETIC.md`, cross-checked against `python/src/graphlocal/medium_defects.py` (`SquareLatticeEdgeCut`), `python/examples/defect_breadth_verification.py` and `python/tests/test_medium_defects.py`.
- **Revision reviewed:** a snapshot of `65f08bb`.
- **Evidence:** `rb.py` and `check1_marginals.py` … `check5_multisets.py` in this folder, with `check3_cube.out`.
- **Brief:** [BRIEFS.md](../BRIEFS.md#n2-planar).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder. Two pickled histogram caches (about 0.7 MB, regenerable by the scripts) and a binary profiler dump were not preserved. See also the [folder README](../README.md#provenance).

---

I found no ERROR or GAP in PLANAR_ARITHMETIC.md: every theorem, lemma and number checks out, both line by line and against my own recomputation. The only findings are four MINOR ones about verification coverage and wording. Nothing was modified in SNAP or the repo. The repo's verifier passed (1,061 checks) and `tests.test_medium_defects` passed (6 tests), with no `__pycache__` written. My scripts are in this folder (`rb.py`, `check1_marginals.py` … `check5_multisets.py`).

**How I recomputed.** I built explicit buffered grids (margins 2r−1 to 2r+3) and took the radius-r ball at every root of G and G∖e, with no shortcut to the roots near the edge. Rooted balls were classified exactly: invariants plus a WL hash to group them, then a VF2++ isomorphism test with distance-from-root labels.

**Results checked and found correct**
1. (1)–(2), the affected-root criterion and N_r = 2r²: the proof is complete, and the histogram is stable at all those margins for r = 1..4.
2. (3): the histogram equals Σ w_b δ_{A_{r,h,b}} − 2r²δ_{B_r} exactly for r ≤ 4. No positive atoms merge there (1 + r(r+1)/2 types).
3. (4): ‖T_rP‖₁ = 4, 16, 36, 64 for r = 1..4.
4. (5), the weighted seminorms: the detour argument is correct, and the formula matches the computation for k = 1, 2, 3 and r = 1..4.
5. D(P) = 2(z³−z⁴), and the seam element EL has variation 4r. I recomputed EL from (P_n□C_m − C_n□C_m)/m at r = 2, 3.
6. Face lemma (6), question (a): yes, it is a genuine face. Interior product degree is deg_C(u)+deg_D(v); both directions are complete; the unit K1 is regular. 600 random pairs at r = 2, 3 pass.
7. (7), the semicharacter: the induced character χ_{r,z} = s_{r,z}∘T_r is a continuous character on all of A_loc, since |χ| ≤ p_{r,0} and T_r is multiplicative.
8. (8): every positive atom is irregular for r ≥ 2 (checked computationally for r = 2..4), so χ_{r,z}(P) = −2r²z⁴.
9. (9)–(10), question (e): the closed disk of radius 2r² lies inside the local spectrum, which lies inside the disk of radius 4r². σ(P) = ℂ is valid because local spectra sit inside the ambient spectrum (T_r is a unital homomorphism).
10. (11), question (b): no cancellation between powers is needed, because the regular part of T_r(P^n) is exactly (−2r²)^n δ_{B_r(Z^{2n})}, and these balls have distinct root degrees 4n. This holds for every r ≥ 2, and r = 0, 1 need only upper bounds. Confirmed for (r,n) = (2, ≤3), (3,2), (3,3), (4,2).
11. (12)–(13): the upper bound, and the topological isomorphism from real entire functions onto the closure of ℝ[P].
12. (14), question (c): the "only if" direction uses χ_{r,z} for r ≥ 2, |z| ≤ 1. Its values −2r²z⁴ cover all of ℂ, and σ(f(P)) = f(ℂ).
13. (15)–(17): the exponential bounds and the tail certificate. Ratio monotonicity and (17) were spot-checked in exact arithmetic.
14. Section 4, question (d):
    - The invariant table and determinant 24 are correct, and (d, N, L₂, J) are additive on all of M₂ (300 random checks).
    - σ_{ℓ¹(M₂)}(T₂P) is exactly the closed disk of radius 16, and (18) holds.
    - T₂(P^n) for n ≤ 3 has C(n+3,3) types, norm 16^n, and supports disjoint across n.
    - T₂(P²) and T₃(P²), recomputed from the explicit four-term grid products, equal my atom convolutions (10 types / 256 and 28 types / 1296).
15. Radius-3 collision, question (f): it is genuine.
    - The balls at (−1,−1) and (0,−2) are not rooted-isomorphic to each other.
    - Both have 25 vertices, 35 edges (intact ball: 36), root degree 4, Q = 4 and sphere vector (1,4,8,12).
    - My own implementation of the cut-line coordinates (cut-line note (2)–(8)) reproduces that note's (10) on line atoms and gives (0,0,0,2) for both balls and the intact ball.
    - Exactly these two of the six radius-3 positive atoms, (h,b) = (1,1) and (0,2), collide with the intact ball.
16. README claims: the planar statements are accurate. The relative Laplacian moments (0,−2,−16,−116,−848) for the lattice and −832 for the 4-tree are confirmed on two box sizes, so the heat difference does start at 2t⁴/3.

**Findings (all MINOR)**

1. **Verification scope (PLANAR_ARITHMETIC.md:292–295).** The note says finite calculations "can check the buffered stabilization, weighted formulas, and invariant table." The shipped code never checks:
   - the weighted formula (5);
   - the Section 4 table (d, N, L₂, J) or the determinant 24;
   - the coefficient (−2r²)^n of planar powers.

   The face-lemma fixtures are radius 2 only (defect_breadth_verification.py:253–270). My independent runs confirm all of these, so it is a coverage gap, not a mathematical one. Adding these checks would close it.

2. **Tautological or literal checks.**
   - test_medium_defects.py:98 asserts `norm_bound(1,2) == local(1).norm(2)`, but `Element.norm_bound` is literally `self.local(radius).norm(k)` (elements.py:102–105).
   - test_medium_defects.py:99 asserts `approximate(...).error == 0`, which follows from the default `approximate` (elements.py:107–109).
   - In the verifier's report, `planar_coordinate_obstruction` (defect_breadth_verification.py:339–345) hard-codes `sphere_sizes`, `root_squares` and `equal_coordinates` as literals. test_defect_breadth_verification.py:26 then asserts against that literal.
   - The real checks (verifier lines 241–250) only compare the three balls' signatures with each other, not against the specific values (1,4,8,12) and 4.

3. **Wording (PLANAR_ARITHMETIC.md:9–10).** "The cyclic ambient geometry changes the growth … to quadratic" misattributes the cause. The quadratic growth comes from the 2-dimensional diamond of affected roots, not from cycles; a ladder graph has cycles and linear growth.

4. **Wording (PLANAR_ARITHMETIC.md:180).** "Extends to a character of A" should read A_ℂ for non-real λ. Equation (14) itself is correctly stated in A_ℂ.

**Observations, not defects**
- At r = 3 (n ≤ 3) and r = 4 (n = 2), distinct multisets of atoms always gave distinct product types. Radius-two-style freeness may therefore hold more generally. The note correctly claims the exact disk only at radius two.
- The stabilization threshold n ≥ 2r+1 is sufficient but not sharp: n ≥ 2r already works for [−n,n]².
- I did not referee the branching or mixed-medium claims in the READMEs (e.g. exp(B₃P) is a unit, 1 − B₃P is not). Their use of the planar face lemma and of the irregularity of the planar atoms matches what this note proves.
