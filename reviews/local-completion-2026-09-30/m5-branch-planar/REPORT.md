# Referee report: branching and planar defects

- **Assigned:** `research/local-completion/BRANCHING_DEFECTS.md`, `PLANAR_DEFECTS.md`, cross-checked against `python/src/graphlocal/controlled.py`, `edge_interactions.py`, `python/examples/branch_planar_verification.py`, `controlled_heat_examples.py` and `python/tests/test_controlled_branching.py`.
- **Revision reviewed:** `d6aba96`.
- **Evidence:** `gcore.py`, `bcheck.py` and the `test_*.py` scripts in this folder.
- **Brief:** [BRIEFS.md](../BRIEFS.md#m5-branch-planar).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder. See the [folder README](../README.md#provenance) for what was not preserved.

---

## Referee report: BRANCHING_DEFECTS.md and PLANAR_DEFECTS.md

**Verdict.** I found no ERROR and no substantive GAP in either note. Every theorem, formula and table number I recomputed with my own code is correct. All the findings are about how the verification is described, or are wording overclaims in the READMEs.

The scratch scripts are in this folder. They use networkx with exact isomorphism and exact integer arithmetic, and no repo code except the tests that deliberately exercise the implementation. The repo is unmodified: `git status` is clean apart from `__pycache__` directories that were already there.

### Coverage: checked and correct

**BRANCHING_DEFECTS.md**
- **Theorem (2), C_F = (-1)^(k-ℓ) C_L, and its proof.** The component-tracking coefficient, the "independent complement omits only leaves" lemma and the star contraction are all sound. The hypotheses are: G connected; every *selected* edge is a bridge of G; unselected bridges and cycles may sit inside blocks. It is an identity in A_0, so it holds for every observable. Brute force:
  - All 3,863 cut sets of all trees with ≤8 vertices, in the component basis. Rooted histograms at r=1,2 were also checked through n=7.
  - 400 random bridge selections in graphs with cyclic blocks and unselected bridges (component basis plus histograms, r≤3).
  - 150 random trees with 9–13 vertices.
  - 360 direct heat-trace evaluations.
  - My counts for n≤7 match the note: 942 cut sets and 370 proper reductions.
- **Leaf-subset formula (3):** checked directly on 3,885 cases. **Budgets (4) and (8):** correct.
- **Star formulas (5)–(7):** component identity holds for k=2..6 and fails at k=1, which the note correctly excludes. Heat e^{-t}(1-e^{-t})^k confirmed numerically.
- **Section 4, infinite trees.** The stabilization argument is sound; in fact every connected exhaustion containing F already has the same quotient tree. The three-ray radius-1 marginal −K13 + 3P3 − 3K2 + K1, with variation 8, is reproduced. Stabilization actually happens by arm length 2r−1, so the stated 2r-neighbourhood condition is a valid sufficient condition.
- **General expansion (9):** checked on 600 random graphs, 463 of them with non-tree quotients (loops and parallel edges). The triangle formula (10) is exact.
- **Sign claims.** The 3-cut line interaction is negative (also checked on P_81), the 3-edge star is positive, and the triangle is negative. So the conclusion that no sign rule based only on the cut count can hold is valid.

**PLANAR_DEFECTS.md**
- **(1) and (2):** confirmed as general identities on random symmetric L with |b|²=|c|²=2.
- **The table:** all of I_2–I_6, plus I_7 and I_8, recomputed by dense integer matrix powers on a 23×23 open grid and on 12×12 and 10×10 tori; all agree. Also confirmed: α=β=10, su=6, sv=38 and 37, and the t⁴/3 difference.
- **Word argument (3)/(4):** correct. The (m, b^T L^m c) pairs and the leading coefficients 1, 1, 2/3, 1/30, 1/120 are reproduced.
- **(6)–(10).** I computed T_r(E²) directly from the four finite products, not by convolution:

  | Element | Radius | Variation | Rooted types |
  |---|---|---|---|
  | E² | r=1..4 | 16, 64, 144, 256 | 3, 6, 10, 15 |
  | E³ | r=1, 2 | 64, 512 | 4, 10 |

  Every coefficient matches (9) exactly, and (10) is exact.
- **(11)–(13):** correct. From my own r=4 histogram, d_j(E²;4) = 0, 0, ½, 0, ½, 0, ½, 0, ½. Formulas (20) and (21) checked for k=2 (j≤8) and k=3 (j≤4).
- **Section 4, the controlled domain.** The class is closed under sums (take the larger cap, add profiles), scalars and products (add caps, convolve profiles), and contains the unit K1 with profile (1). Certified heat is valid on all of it. I checked line by line:
  - the binomial argument for raising the cap, and the factorial-moment identity;
  - the interpolation proof of (16), including mixed insertions and deletions;
  - the tail bound (17);
  - the quantities C_M and B_M, the reciprocal-exponential enclosure, and the radius bound (19);
  - that heat does not depend on the cap and is multiplicative;
  - continuity on classes with a fixed profile.

  Stress test of (16): 831 random interactions, including mixed edits, at caps D, D+1 and D+3. The maximum of |d_j|/bound was 1.000: the bound is tight and never exceeded.
- **Implementation.** `controlled_heat` implements (18)–(19) faithfully. 105 random `EdgeInteraction` enclosures (including mixed edits, at t = 0.1, 0.5, 2) all contained 40-digit eigenvalue values; E² at t=1/20 and 1/3 and E³ at t=1/20 contained the closed forms. `EdgeInteraction.finite()` matched brute force on 300 cases.

### Findings, most severe first
1. **OVERCLAIM — BRANCHING_DEFECTS.md:260–275** ("The verification covers the following distinctions"). Several listed checks do not exist in the referenced code:
   - `verify_trees` compares only against (2), never (3).
   - There is no infinite-tree exhaustion or stabilization test.
   - The star and triangle heats are checked against closed forms, not "full Laplacian eigenvalues".
   - Nothing tests (9). `quotient_leaf_cuts` simply raises on cyclic or loop quotients, and grep finds no fixture with loops or parallel edges.

   The mathematics is unaffected: I verified (3) and (9) independently.

2. **MINOR — BRANCHING:245–249.**
   - The "independent" verifier imports the library's IsoGraph, LocalHistogram, components and apply_edge_edits.
   - The 3,763 rooted-histogram identities follow logically from the 942 component-basis identities, because T_r is linear in the component basis. So they test the histogram code, not the theorem. The r=0 cases are just mass 0 = 0.

3. **MINOR — PLANAR:72–76.** The claimed checks on a 12×12 torus (sparse) and a 10×10 torus (dense) are not in the repo. The verifier uses open 17×17 and 21×21 grids, and the only dense test is on C_4. The values are nevertheless correct; I reproduced them on exactly those tori.

4. **MINOR — PLANAR:142–144.** The note says E² was "checked at r=1,2,3,4", but the repo tests cover only r=1..3 (`range(1, 4)`). The r=4 values (256 and 15) are correct by my computation.

5. **MINOR (wording) — README.md:115, "orientation-sensitive finite-edge interactions".** The note itself states the interaction is unchanged if either edge's orientation is reversed (PLANAR:58–59). What is actually shown is dependence on how the two edges are arranged (perpendicular versus collinear adjacency), first visible at order t⁴ (I_4 = 226 versus 218). Something like "arrangement-sensitive" would be accurate.

6. **MINOR (overclaim) — research/local-completion/README.md:43–44, "certified heat on that larger controlled domain".** The profile class does contain the old edit-budget domain and is closed under products. But it is not shown to be strictly larger: PLANAR:402–404 itself concedes that E^k might have some other bounded edit representation.

7. **MINOR (presentation) — PLANAR:209 and 220–223.** The "hence" that gives H_t(E^k) from Tr K_t^{⊗k} silently identifies the operator trace on ℓ²(Z^k) with the local heat functional. That needs a short diagonal-sum/Fubini argument; it holds because about j^k roots contribute at order j. Harmless, since the multiplicativity in Section 4 gives (13) directly as H_t(E)^k.

No other overstatements found in the note introductions or either README.
