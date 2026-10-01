# Referee report: mixed medium arithmetic (commit 65f08bb)

- **Assigned:** `research/local-completion/MIXED_MEDIUM_ARITHMETIC.md`, which builds on `BRANCHING_ARITHMETIC.md` and `PLANAR_ARITHMETIC.md`. Cross-checked against `python/src/graphlocal/medium_defects.py` and `python/examples/defect_breadth_verification.py`.
- **Revision reviewed:** a snapshot of `65f08bb`.
- **Evidence:** `mm.py` and `t1_marginals.py` … `t8_c4.py` in this folder, with the outputs `t4_r3.out`, `t6_r3.out` and `prof2_B4.out`.
- **Brief:** [BRIEFS.md](../BRIEFS.md#n3-mixed).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder. See the [folder README](../README.md#provenance) for what was not preserved.

---

## Referee report: MIXED_MEDIUM_ARITHMETIC.md (SNAP @65f08bb)

**Verdict:** I found no errors and no gaps. Every numbered claim checks out, both line by line and in code I wrote from scratch. There are three MINOR items. My scratch code is in this folder (`mm.py`, `t1`–`t8`). SNAP and the repo are unmodified, and no `__pycache__` was written.

### Coverage (checked and correct)
- **Main theorem (l.17–27):** for distinct d_i ≥ 3, the closed subalgebra is ≅ O(C^{s+1}), with the unit criterion, spectra and unbounded variation as stated.
- **§1 (1)–(3):**
  - A root is affected exactly when its distance to the nearer endpoint is ≤ r−1. The counts are 2(d−1)^j and 2r².
  - ‖T_rB_d‖₁ = 4Σ_{j<r}(d−1)^j, ‖T_rP‖₁ = 4r², and both have mass 0.
  - Recomputed on buffered trees (depth 2r+1 and 2r+2) for d=2..5, r≤4, and on grids (margin 2r+1 and 2r+2) for r≤5. Affected sets, counts, norms, the unique negative atom with coefficient −c(r), and buffer stability all match exactly.
- **§2 face lemma (4):** correct in both directions. ρ_r is a monoid character, projection onto F_r is a homomorphism at fixed r, and (5) holds (positive atoms lie outside F_r for r ≥ 2).
- **§3 Γ and lemma (6):**
  - Adjacency means "no common non-root neighbour". This is the same as "the two root edges lie in no simple 4-cycle together" (checked on 2,170 pairs).
  - (6) is correct for r ≥ 2: cross-factor pairs always share (u,v), and common neighbours of same-factor pairs stay in that factor. ν_H is additive, and (8) holds.
- **§4 (9)–(12):**
  - s_{r,z,w} is a bounded, unital, multiplicative function on all of M_r. So χ = s∘T_r is a genuine continuous character on all of A_loc.
  - (11) holds, and the image of X is the full closed polydisk.
  - Numerically, (11) and χ(xy)=χ(x)χ(y) hold on computed products at r=2,3 (max error 8e-14).
- **§5 (13)–(16) and the signed-measure corollary:**
  - The face projection gives exactly the lower term, and distinct α give distinct ν-vectors.
  - The degree bound (14), the topology argument, and "bounded variation ⟺ constant" are all correct.
- **§6 (17)–(20):** correct, including c₃c_P = 4r²(2^r−1).
- **§7 criterion:** correct as a sufficient condition. Submultiplicativity of p_{r,k} does suffice for convergence.
- **Commit message numbers:** I recomputed the relative Laplacian moments independently: B4 gives (0,−2,−16,−116,−832) and P gives (…,−848). The heat difference 2t⁴/3 follows.
- **Repo verifier:** rerun gives 1,061 checks, with output identical to the committed JSON. 441 = 21² fixture pairs. Both new test modules pass. I found no tautological checks.

### Independent recomputation
Method: exact rooted classes (WL-bucketed, confirmed with vf2pp using distance labels). Product histograms come from balls in the four product graphs Ga□Ha − Ga□Hb − Gb□Ha + Gb□Hb.

- **Products B3·B4, B3·P, E·P (E=B₂), B4·P, B3², P², E² at r=2 and r=3** (all 16.6k roots for B3·P and E·P at r=2):
  - Direct product histograms equal the convolution of the marginals.
  - The face projection is exactly c_a·c_b·δ_{bg_a⋆bg_b} in every case.
  - The Γ-components of the background products are {K3,K4}, {K3,2K2}, {3K2} and {K4,2K2] as expected (E·P gives 3K2).
- **Polynomials:** r=2 and r=3 at degree ≤3 in (B3,B4,P), r=4 at degree ≤2, and r=3 with B5.
  - Face atoms are distinct, the ν readout is exact, and 300 random polynomials per run satisfy (13).
  - In fact no cancellation occurs at all, because distinct monomials have disjoint supports. So the upper bound in (13) is attained at these radii. The note does not assert this, correctly.
- **Adversarial tests for (a)/(b):**
  - 10,600 random rooted pairs at r=1..4, drawn from 49 graphs: K4, K5, diamond (chorded square), paw, book, wheels, octahedron, king and triangular-lattice patches, Petersen, dense G(n,p). Plus an exhaustive sweep of 10,952 pairs over the triangle- and chord-rich fixtures at r=2,3.
  - The face lemma never fails, in either direction. (6) never fails for r ≥ 2.
  - At r=1, (6) fails in 1,470 of 1,500 pairs (e.g. the path's centre ball squared gives one Γ-component instead of two). This matches the note's explicit r ≥ 2 restriction.
  - The "adjacent when they share a 4-cycle" version would not be additive (fails 5,852 of 6,000 at r=2). The note's complementary relation is essential and correct.

### Findings (most severe first)

1. **MINOR — MIXED_MEDIUM_ARITHMETIC.md:369–374.** The claim "These specific background coordinates therefore do not establish independence of … E and … P" is true but understates the problem.
   - The planar background is the square of the line background: Z_r = L_r ⋆_r L_r (verified for r ≤ 5).
   - So every semicharacter supported on the face has s(Z_r) = s(L_r)². Hence χ(P) = −χ(E)²/2 and χ(E²+2P) = 0 for every character of this method, at every radius.
   - Concretely, the face projection of T_r(E²+2P) is 0 for all r ≥ 2, yet E²+2P ≠ 0: ‖T_r‖₁ = 16r²−8, i.e. 8, 56, 136, 248, 392 for r=1..5. E² and P also share support types.
   - No other choice of coordinates can fix this; ingredient 2 of the §7 criterion fails intrinsically for (E,P). The note should say so.

2. **MINOR — README.md:194–195.** "distinct branching degrees and the planar cut give independent entire coordinates" does not say d ≥ 3. The BRANCHING note and `InfiniteRegularTreeCut(2)` (which is E) treat d=2 as a regular-tree degree, and E together with P is neither proved nor provable by this method (finding 1). research/local-completion/README.md:187 states "d>=3" correctly.

3. **MINOR — verifier coverage (defect_breadth_verification.py:252–270; README "441 rooted-product fixtures").** The fixtures do not test the universal claims (4) and (6) as broadly as described ("arbitrary rooted fixtures", l.266).
   - 882 of the 1,061 checks come from 7 tiny graphs at radius 2 only.
   - All of them have eccentricity ≤ 2, so every ball is the whole graph and truncation is never exercised. 41 of the 441 pairs have a K1 factor.
   - No fixture has a common neighbour among the root's neighbours (diamond, K4) or two non-root common neighbours (K_{2,3}).
   - No product Γ check is made at r ≥ 3, and mixed-coefficient recovery is tested on a single polynomial at r=2.
   - The checks are not tautological, and the proofs do not depend on them. My broader tests above found no counterexample.

**Not checked:** the Bhatt–Patel citation (a classical fact, not load-bearing).
