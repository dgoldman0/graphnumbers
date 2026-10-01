# Referee report: representation theorem and comparison notes

- **Assigned:** `research/local-completion/REPRESENTATION_PROBLEM.md`, `REPRESENTATION_THEOREM.md`, `ANALYSIS.md`, `COMPARISON_LEMMAS.md`; also what `verify_representation.py` and `verify_local_algebra.py` actually test.
- **Revision reviewed:** `d6aba96`.
- **Evidence:** the scripts in this folder (listed at the end of the report).
- **Brief:** [BRIEFS.md](../BRIEFS.md#m1-representation).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder. See the [folder README](../README.md#provenance) for what was not preserved.

---

## Referee report: REPRESENTATION_PROBLEM.md, REPRESENTATION_THEOREM.md, ANALYSIS.md, COMPARISON_LEMMAS.md

**Verdict:** I found no ERROR and no substantive GAP. All four of your key claims (a)–(d) hold, and so do the comparison claims (e). What remains is four MINOR items: wording, a loose bound, weak verifier coverage, and citations I couldn't reach. I edited nothing in the repo (git status is clean). My scratch work is in this folder.

### Coverage: results checked and found correct

**REPRESENTATION_THEOREM.md**
- **§1, divergence:** ∂F is determined by the 2R-ball, and |∂F| ≤ 2C|B_2R|^{k+1}. Each receiver's R-ball lies inside the 2R-ball and its distances are preserved there. Λ_a(∂F) is continuous on 𝒲.
- **§1, countable tests:** the doubly-rooted indicator tests suffice. Absolute summability justifies swapping the sums. The balance condition is stated correctly: any receiver within distance R, transport determined by the sender's R-ball. It is not restricted to edge-rooted pairs, and doesn't need to be.
- **Necessity:** correct.
- **Step A, formula (1):** correct, including A_H, the 1/t! factor, the |S| ≤ N truncation, the Q-sum, connectivity and r = 0. I implemented (1) myself in two ways:
  - Literally, with no degree pruning, for D ≤ 4 and r = 1: 115,424 rooted evaluations, all exact.
  - With degree > D patterns discarded, for (D,r) = (1,2), (2,2), (2,3), (3,1), (4,1): about 139k evaluations, all exact.
- **Step B, finite realization lemma:** the argument is right; (2) plus triangularity gives the conclusion. I tested it independently, separately from the proof. I compared K, the functionals that vanish on all connected-graph histograms, with K′, the span of rerooting differences inj_u(F) − inj_v(F) between ball types. The identity dim K′ + rank(histograms) = number of types held in every case:
  - D = 2, r = 1..4
  - r = 1, D = 1..4
  - D = 3, r = 2: 443 types, dim K′ = 90, rank 353.
- **Step C, degree cutoff:**
  - The (r+1)-ball suffices; I also confirmed the r-ball alone does not (27,340 locality checks).
  - θ pushed forward through T_{r+1}(x) equals T_r(Q_D x), so the cut-off arrays are histograms of real graph combinations.
  - Bound (3) held in 3,240 signed checks.
  - Balance survives the pull-back.
- **Identification:** the homeomorphism and the algebra isomorphism are correct.
- **§3:** the endpoint-rooted P3 counterexample is correct. Positive mass-one elements are exactly the unimodular laws with all moments finite (π-system argument on doubly-rooted cylinders).
- **§4, measure criterion:** sufficiency (monotone b_r, Jordan-type splitting), uniqueness, ‖μ‖ = M(a), and the remark that per-radius summability doesn't bound moments of |μ| (an easy example confirms it) are all correct.
- **§5, J(c):**
  - The series converges for every c.
  - The norms at r = N_m are right; I recomputed them, and total variation jumps at r = 1, 4, 13, 40.
  - J(c) has a finite signed measure exactly when c ∈ ℓ¹.
  - J is continuous for the product topology, P is continuous, and PJ = id. So JP is a continuous projection and J is a topological embedding onto a closed, complemented copy of ℝ^ℕ.

**REPRESENTATION_PROBLEM.md:** the problem statement is accurate, and commit 8298f46 does add the three files it names.

**ANALYSIS.md**
- p_{r,k}(aK1) = |a|, so the scalar line is closed.
- The V and E product rules hold, and E(f(X)) = f′(V(X))E(X) for polynomials (checked on random signed X) and for entire f.
- E(exp tL) = t·eᵗ.
- Q commutes with differentiation and Riemann integration.
- K1 − tK2 is invertible iff |2t| < 1 (the disk characters D_x(1/(2t)) kill it otherwise).
- The unit group is not open. I confirmed this with a witness of my own: the bipartite-ball character χ_r is multiplicative and vanishes on u_n = K1 − C_{2n}/2n + C_n/n for odd n, while u_n → K1.

**COMPARISON_LEMMAS.md**
- §1: the counting-measure picture is right, including injectivity.
- §2: bounded-degree convergence is exactly Benjamini–Schramm convergence.
- §3: the equivalence holds. The walk bound held pointwise on random graphs, and the Sidorenko star bound had 0 violations over 300 random graphs × 9 (j,s) pairs; the uniform-integrability argument is sound.
- §4: E is a dual-number point derivation.
- §5:
  - The vertex-weighted component norm N gives distance 2 between distinct normalized cycles.
  - z_n vanishes through radius R exactly when n > 2R+1, a sharp threshold I checked.
  - Hence no continuous norm exists.
  - The component count c is a character that is not continuous here, so A_loc is not the universal envelope.
- §6: K1 and K2 are at asymptotic-spectrum distance 0; the graphon argument and p₁,₁ = 4 and n are right.

### Findings (most severe first)

**1. MINOR / OVERCLAIM (wording).**
- **Where:** `REPRESENTATION_THEOREM.md:351`, heading "A closed family of elements beyond finite signed measures"; and top-level `README.md:32-33`, "constructs a closed sequence-space family beyond them".
- **Problem:** J(ℝ^ℕ) contains J(ℓ¹), whose elements *do* have finite signed measures. J(c₀₀) is dense in J(ℝ^ℕ), so the part actually beyond the measure class, J(ℝ^ℕ∖ℓ¹), is not closed.
- **Fix:** say "a closed complemented copy of ℝ^ℕ whose elements are measure-representable iff c ∈ ℓ¹". The local README already phrases this correctly.

**2. MINOR.**
- **Where:** `REPRESENTATION_THEOREM.md:242`, "Balance for a is exactly balance for the transformed arrays."
- **Problem:** only "a balanced ⇒ a^(D) balanced" holds, and only that is needed. The converse fails.
- **Counterexample:** take a = point mass at K_{1,5} rooted at its centre (compatible, all moments finite). For D = 1, a^(1) = T(K1) is balanced. But a is not: the transport u→v along an edge with deg u > deg v gives Λ_a(∂F) = 5.

**3. MINOR (loose bound, not an error).**
- **Where:** `REPRESENTATION_THEOREM.md:112-115, 163-165, 186, 297-300`, the bound M(D,r) = (D+1)·b(D,r).
- **Problem:** this bound is never needed. After discarding degree > D patterns, every surviving pattern is a rooted graph of maximum degree ≤ D within radius r, so it has at most b(D,r) vertices. The triangularity argument works unchanged on that smaller graph list, so the quoted M(n,n) is at least a factor n+1 too pessimistic.
- **Evidence:** for D = 3, r = 2, full rank 353 is reached using only graphs on ≤ 10 vertices (M = 52). For D = 2 and r = 1..4, full rank is reached at 3, 5, 7, 9 vertices, against M = 9, 21, 45, 93.
- **Related:** Step B (line 171) assumes *every* a_s is degree-bounded, but the proof only uses this for a_r.

**4. MINOR (weak verifier evidence).**
- **Where:** `REPRESENTATION_THEOREM.md:409-414`, describing `verify_representation.py`. Both repo verifiers reproduce their recorded JSON, but:
  - **Formula (1) isn't tested as written (lines 97-120).** The compiler caps attachments by each vertex's remaining degree, so (1)'s literal |S| ≤ N truncation and discard step are never exercised.
  - **The "transport balance" checks are tautological (lines 245-254).** Σ_o ∂F = 0 holds identically on any finite graph, and only one transport is tested.
  - **Triangularity is tested on only 10 graphs** with ≤ 4 vertices (lines 200-212).
  - **Sufficiency is untested.** Nothing checks Step B's span conclusion or that Step C preserves balance. My rank tests and Step C tests cover these.
  - **`verify_local_algebra.py:310-312` is weaker than claimed.** The ".tex" note's "separation was checked for 19 graph types" means only that 19 individual graphs have pairwise-distinct histograms, not that the histograms are linearly independent. The lemma itself is trivially true.

**5. MINOR (unchecked citations).** I could not verify the cited sources (Knill 2021, de Boer–Buys–Zuiddam, Kurauskas Thm 2.2, Albeverio–Mazzucchi, Bordenave–Caputo) because the network proxy blocks arxiv.org. I checked the mathematics of those comparisons independently, as listed above.

### Scripts (in this folder)
- `glib.py`: independent graph toolkit; canonical form and injective counts validated against networkx and brute force.
- `check_indicator.py`: formula (1), literal and pruned.
- `types_gen.py`: enumerates the ball types.
- `check_realization.py`, `check_realization_32.py`, `rank_small.py`: the K = K′ rank tests.
- `check_misc.py`: Step C and J(c).
- `check_misc2.py`: comparison and analysis numbers.
