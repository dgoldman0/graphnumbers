# Referee report: branching arithmetic (commit 65f08bb)

- **Assigned:** `research/local-completion/BRANCHING_ARITHMETIC.md`, cross-checked against `python/src/graphlocal/medium_defects.py` (`InfiniteRegularTreeCut`), `python/examples/defect_breadth_verification.py` and `python/tests/test_medium_defects.py`.
- **Revision reviewed:** a snapshot of `65f08bb`.
- **Evidence:** `canon.py`, `coords.py`, `atoms.py` and `check1_marginals.py` … `check7_general.py` in this folder, with their outputs `check*.out`. `check3.out` ends in the traceback of a stalled run, superseded by `check3b_multisets.py`.
- **Brief:** [BRIEFS.md](../BRIEFS.md#n1-branching).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder. See the [folder README](../README.md#provenance) for what was not preserved. The lead reviewer confirmed finding 1 against the repository text.

---

**Referee report: BRANCHING_ARITHMETIC.md (commit 65f08bb)**

**Bottom line.** I found no ERROR and no substantive GAP in the note. I reproduced every quantitative claim with code written from scratch: AHU canonical forms for trees, and WL colour refinement plus VF2 for everything else. The problems I did find are one README overclaim, some missing r≥2 qualifiers, and thin verifier coverage of the note's key new lemma.

**Coverage (each checked and found correct)**
- **§1 (2)–(6):** I built finite buffered trees for d=2..5, r=1..4, at buffers L=2r−1, 2r and 2r+1; all three give identical histograms. Each case has exactly r+1 types, coefficients +2q^j and −2S_r, one deficient interior vertex at distance j, ball sizes b−S_{r−j}, variation 4S_r, and p_{r,k} as in (6) for k=1,2,3. B_2 matches E (P_n−C_n) at r=1..4.
- **§2 (7)–(12):** The atom values (12) hold for d=2..5, r=2..4. The Q-identity and additivity of N_d and c_j hold on 378 pairs of *arbitrary* balls at each of r=2,3. These include triangles, chorded 4-cycles, K4, wheels and random graphs, so the claim is checked on the whole monoid, not just atom products. Values like 2/3 occur.
- **§3 (13)–(18):** The character construction, the r≥2 restriction and the threshold algebra are right. At r=1 the local spectrum is {2w^{d−1}(1−w): |w|≤1}, which is not a disk, so the restriction is necessary.
- **§3–4 (16)/(20) no cancellation:** I checked that atom products of all multisets are pairwise non-isomorphic, using an isomorphism-invariant certificate. Single degree: r=2 up to power 7 (d=2), 6 (d=3), 5 (d=4), 4 (d=5); r=3 up to 6/4/3; r=4, d=3 up to power 2.
- **Direct finite products:** (G₃∖e−G₃)□(P_n−C_n) at r=2 and r=3, and (G₃∖e−G₃)² at r=2, equal the local convolution, with norms 96, 336 and 144.
- **(19), (21), divisibility, (22):** The logic is correct for r≥2.
- **§5 tree-factor lemma:** The proof is valid line by line, and the depth-k+1≤r square visibility argument holds. I also implemented the proof's recovery procedure. It recovers the exact factor multiset, and products are pairwise distinct, for every multiset of nontrivial rooted-tree balls tested: size ≤2 from trees with ≤7 vertices and size ≤3 from trees with ≤5 vertices, at r=2 and r=3 (3,471 multisets in total).
- **§6 (23):** Supports are disjoint and coefficients do not cancel. Checked for degrees (2,3), (2,4), (3,4), (3,5), (2,3,4), (2,3,4,5) at r=2 and for (2,3), (3,4), (2,3,4) at r=3 (up to 1,000 multisets per case).
- **§6 characters, (25)–(28):** The K_F rank argument is sound. Separation plus the lemma give ℚ-independence, finitely many columns reach full rank, and the resulting h_C are additive on all of M_r. So the phase characters really are continuous characters of all of A_ℂ. The polydisk topology argument, the units and spectra, and the real-scalar versions all hold.
- **§7 (30)–(34):** I computed exact integer traces on large buffered graphs:
  - Tree d=4 (L=6 full trace; L=6,7 local): Δ₀..₈ = 0, −2, −16, −116, **−832**, −5972, −42976, …
  - Lattice (boxes n=8,10,12 and a 25-torus, all agreeing): 0, −2, −16, −116, **−848**, −6332, −48136, …
  - a_k is (2,10,56,334) for the tree and (2,10,56,338) for the lattice.
  - So −832 is the tree and −848 the lattice.
  - Formula (32) also holds on ℤ³ (c_e=4), and on trees with d=3 and d=5.
  - Heat difference: H_t(B₄)−H_t(P) = (2/3)t⁴ − 3t⁵ + (43/6)t⁶ − … Finite-graph eigenvalue heat traces agree; at t=0.02 I get 9.75105e−8 against 9.75104e−8 from the series.
- **README numbers:** The variations 4Σ(d−1)^j and 4r² (P at r=1..4), and the moments −832/−848, are correct.
- **Repo checks:** The repo's verifier passes (1,061 checks) and test_medium_defects passes. Nothing was written into SNAP.

**Findings (most severe first)**

1. **OVERCLAIM — README.md:194–195 (same wording in the commit message).**
   - Claim: "More generally, distinct branching degrees and the planar cut give independent entire coordinates."
   - Problem: BRANCHING §6 uses "branching degrees" to include d=2 ("They may include d₁=2"), and the commit says "distinct tree degrees, including the cut line". Read that way, the sentence asserts that E and P are independent.
   - MIXED_MEDIUM_ARITHMETIC.md:8, 17 and 370–374 proves only d_i≥3 and states that independence of E and P is *not* established.
   - Concretely, R₂⋆_rR₂ is the ℤ² ball Z_r, so every regular-face character satisfies χ(P) = −χ(E)²/2. The method provably cannot separate E from P.
   - Fix: say d≥3, as research/local-completion/README.md:187 already does.

2. **MINOR — missing r≥2 qualifiers.**
   - Boxed (22) at BRANCHING:196–198 and (29) at :332–333 carry no r≥2. (16) at :144–146 relies only on the "Fix r≥2" in §2.
   - All three are false at r=1. For example, ‖T₁exp(B₂)‖₁ ≈ 11.98 while e⁴ ≈ 54.6, and ‖T₁(2E+E²)‖₁ = 16 while (16) gives 24.
   - The cut-line note does state (r≥2) at its (31). Add the qualifier here too.

3. **MINOR — verifier coverage and near-tautologies.**
   - The new code does not test the note's key new ingredients:
     - the §5 tree-factor lemma;
     - the K_F construction;
     - exactness of (23) for polynomials with several monomials in two or more tree degrees;
     - (16) beyond power 2;
     - additivity of N_d and c_j away from atom products.
   - The multi-degree polynomial check at defect_breadth_verification.py:288–317 includes P. It certifies only the regular-face lower bound, which is the MIXED note's method, not equality (23).
   - The (12) check at :192–200 runs only at r=2 with at most two atoms.
   - test_medium_defects.py:71 (`norm(0)==4*branch_sum`) is guaranteed by how `local()` builds its coefficients once the length check at :70 passes. The genuinely independent finite-witness comparison (:48–58) covers only r≤2 plus (d,r)=(3,3).

4. **MINOR — research/local-completion/README.md:310–311.**
   - "The observed low-radius product norm equalities are not asserted at arbitrary radii" reads as disclaiming what BRANCHING (16)/(23) *do* assert for tree products at every r≥2.
   - Clarify that the disclaimer concerns products containing P.

5. **MINOR, harmless.**
   - §1's buffer condition L≥2r is sufficient but not sharp: L≥2r−1 already suffices. That is what the verifier uses, and I confirmed it gives identical histograms.
   - The step from (33) to (34) silently uses H_t(X) = Σ(−t)^mΔ_m/m!. That identity is valid for the uniformized definition in SPARSE_DEFECTS_AND_RELATIVE_HEAT.md, by absolute convergence since |Δ_m| ≤ 2m(2D)^{m−1}. One sentence saying so would help.

**Scratch files** are in this folder. The scripts are canon.py, coords.py, atoms.py, check1_marginals.py, check2_coords.py, check3_products.py, check3b_multisets.py, check4_laplacian.py, check5_treefactor.py, check6_planar_var.py and check7_general.py, with outputs in the matching `*.out` files. check3_products.py stalled in VF2 after its first checks (direct products and d=2 powers passed); check3b_multisets.py replaces it.
