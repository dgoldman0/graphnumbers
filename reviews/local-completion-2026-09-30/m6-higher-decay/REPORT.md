# Referee report: higher interaction geometry and interaction decay

- **Assigned:** `research/local-completion/HIGHER_INTERACTION_GEOMETRY.md`, `INTERACTION_DECAY.md`, cross-checked against `python/src/graphlocal/interaction_moments.py`, `interaction_bounds.py`, `python/examples/higher_interaction_verification.py` and `geometry_bound_examples.py`.
- **Revision reviewed:** `d6aba96`.
- **Evidence:** `core.py`, `tree_exhaustive.py` (with `tree_exhaustive_11.log`), `tree_random_large.py`, `firstorder_check.py`, `decay_check.py`, `sign_reversal.py` and the other scripts in this folder.
- **Brief:** [BRIEFS.md](../BRIEFS.md#m6-higher-decay).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder. See the [folder README](../README.md#provenance) for what was not preserved.

---

## Referee report: HIGHER_INTERACTION_GEOMETRY.md and INTERACTION_DECAY.md

**Verdict:** I found no ERROR and no substantive GAP. I re-derived every theorem and recomputed every number with my own code, which does not use the repo package. Everything held. The four findings below are all MINOR; one further item is an observation, not an error. Scratch work is in this folder. The repo is unmodified (`git status` is clean; I ran with `PYTHONDONTWRITEBYTECODE=1`).

### Coverage: results checked and found correct

**HIGHER_INTERACTION_GEOMETRY.md**
- **Cyclic expansion (1).** I re-derived it two ways: from the word expansion with the marked-rotation bijection (valid for periodic words too), and from the determinant/log-det identity (2), which is also correct. Orientation invariance and the infinite-graph trace-class remark are fine.
  - Tested against brute-force inclusion–exclusion of Tr L^n on 245 random graphs covering deletions, insertions, mixed edits and random orientations, through order 7: 0 mismatches.
  - Tested on 60 random rational-weighted perturbations: 0 mismatches.
- **First-order test (3)–(4) and spatial formula (5).** The lower bound, the no-self-transition remark, vanishing when the couplings are disconnected, Cayley–Hamilton, and the shortest-path formula (5) are all correct.
  - Random test on 306 graphs with k=2..4: vanishing below ν and the exact value of I_ν held in every case. Four cases had (4)=0, i.e. delayed onset.
- **Three-defect formulas (6)–(7).** Correct, including 37 random cases where minima tie.
- **K_4 repetition example.** Confirmed: c_13≡0, I_4=4, leading term t^4/6.
- **Six-vertex cancellation example.** Confirmed: s=(2,1,0), γ=(−1,−2,1), ν=6, triangle term −24 against +24, I_7=−56, I_8=−1288, leading term t^7/90.
- **Four-defect example.** ν=5: no weight-4 covering word exists, since label 4 has only one weight-1 neighbour. Confirmed +10/−10 at order 5, +264/−408/+144 at order 6, I_7=28, leading term −t^7/180.
- **Tree theorem (10)–(12).** The proof is correct: bridge reduction, (13), (14), and the contour count p(T).
  - My own exhaustive enumeration covered every nonempty cut set of every tree with ≤11 vertices: 310,419 cut sets, including 3,958 single cuts. Onset order, coefficient and sign matched in all of them.
  - Restricted to ≤6 vertices it reproduces the note's 249 cut sets and 51 single cuts, and the repo's 194 positive / 55 negative split.
  - I also tested 150 random trees with 8–18 vertices, k≤7 and p(T) up to 240: all matched.
- **Tree consequences.**
  - Line pair: matches.
  - Spider (2,2,2): t^9/20160. Spider (2,2,2,2): t^12/6652800.
  - Binary quartet: t^6/30. Spider (2,1,1,1): t^6/20.
- **Sign reversal.** Computed by 80–100-digit eigendecomposition and, separately, an exact rational Taylor series with a rigorous tail bound:
  - H(1)=0.02567457567783610…
  - H(3)=−0.03857341890563762…
  - Both lie inside the JSON's rational enclosures and the note's decimal intervals, so the crossing is real and correctly certified.
- **Implementation.** The library's `interaction_moments` and `tree_interaction_leading` agree with my brute force on 420 random cases (mixed edits, with and without bridge reduction). Rerunning the repo verifier without writing any files reproduces the stored JSON exactly.

**INTERACTION_DECAY.md**
- **Proof of (6)–(8).** Correct, including the multiplicity count Σ(a_i+1)=j and the cube-integration identity.
- **(6) and (7).** Tested on 200 random graphs with mixed edits through order 12, and at D+1 and D+4 (240 cases): no violations. 17 cases had onset strictly later than k+τ_*, which the note allows.
- **Heat bounds.** Both forms of (9), the tail bound (10) at M=0,2,5,9, (14), and the per-cycle minimum were tested against exact heat at t∈{0.1, 0.5, 1, 3}: 0 violations.
- **Inequalities.** (11), via Markov on the falling factorial, is correct. (12) was checked exhaustively over small ranges.
- **Profile (13).** Correct, including its extension to larger D; the planar note's Cartesian convolution rule re-derives correctly.
- **Poisson enclosures (15).** 3,077 exact-rational checks, all passing.
- **Tree bound (16).** τ_*=2(s−ℓ) is correct; tested on 360 random tree/time cases with no violation.
- **Stated numbers.** All correct:
  - Path example: onset 22, and a bound far below 10⁻¹⁸ at t=1/4 (about 6e−26 reduced, 1e−24 unreduced).
  - Spider example: onset 9.
  - Cycle examples: bounds 2/17!, 2/29!, 2/47!, against 1 for the original profile.

**Prose summaries.** The intros of both notes, research/local-completion/README.md lines 46–53 and 187–188, and the top-level README.md lines 120–129 do not overstate what is proved.

### Findings, most severe first (all MINOR)

1. **MINOR – the verifier does not test the stated example.**
   - HIGHER_INTERACTION_GEOMETRY.md:340–343 says the independent verifier "checks the repeated-label … examples". The §3 example at lines 176–185 is K_4 with path edges.
   - The verifier's `repeated_defect_triple` fixture (python/examples/higher_interaction_verification.py:22–25) is a different graph: the paw {01,02,03,12} with cuts {01,03,12}.
   - The K_4 example is only covered by tests/test_interaction_moments.py.
   - No mathematical consequence: the two give identical mixed moments through order 16 (0,0,0,0,4,40,252,1288,5852,…).

2. **MINOR – edge case k=1.**
   - INTERACTION_DECAY.md:152–153 says "If every pair of distinct edited supports is at distance at least R, τ_*≥kR."
   - For k=1 the hypothesis is vacuous but τ_*=0, so the conclusion fails. It should say k≥2.

3. **MINOR – terse step in the tree proof.**
   - HIGHER_INTERACTION_GEOMETRY.md:286–289 says "an optimal cyclic leaf order … determines these local cyclic orders."
   - Surjectivity onto rotation systems needs one more line: the successor permutation at each internal vertex is a single cycle. Otherwise, splitting that vertex along the cycles gives a forest whose face tracing has at least two closed walks, contradicting that one walk covers every directed edge.
   - The conclusion is right, as confirmed by the exhaustive check.

4. **MINOR – factor-2 wording.**
   - INTERACTION_DECAY.md:288–289 says the separation in the tail "is the total amount of connecting tree left after its terminal edges are removed."
   - The Poisson threshold in (16) is 2(s−ℓ), which is twice the number of internal edges.

5. **Observation (not an error) – the sign-reversal example.**
   - The fixture's interaction changes sign twice: at t≈1.515635 and again at t≈10.121135.
   - It is positive for t>10.12, because the lowest atom of its signed spectral measure (λ≈0.518806) has coefficient +1.
   - The note claims only a crossing in (1,3), which is correct.
   - The repo's own evidence for this example is just the library's `controlled_heat` certificate. The test only checks the signs of the library's intervals and does no independent recomputation; my independent values confirm them.
   - Similarly, `test_shortest_covering_words_cancel` checks only that the order-6 contributions sum to zero, not the individual values +264/−408/+144 quoted in the note. I confirmed those values independently.
