# Referee report: heat return, sparse defects, defect interactions, geometry versus spectrum

- **Assigned:** `research/local-completion/EFFECTIVE_ANALYSIS_AND_HEAT.md`, `SPARSE_DEFECTS_AND_RELATIVE_HEAT.md`, `DEFECT_INTERACTIONS.md`, `GEOMETRY_VERSUS_SPECTRUM.md`, cross-checked against `python/src/graphlocal/heat.py`, `defects.py`, `interactions.py`, `prepared.py` and the related examples and results.
- **Revision reviewed:** `d6aba96`.
- **Evidence:** `locality_bruteforce.py`, `heat_values.py`, `sparse_identities.py`, `cuts_local.py`, `interaction_heat.py`, `cospectral.py`, `stress_heat.py`, `stress_relative.py`, `adversarial.py` and `prepared_check.py`, all in this folder.
- **Brief:** [BRIEFS.md](../BRIEFS.md#m4-heat-defects).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder.

---

## Referee report: EFFECTIVE_ANALYSIS_AND_HEAT, SPARSE_DEFECTS_AND_RELATIVE_HEAT, DEFECT_INTERACTIONS, GEOMETRY_VERSUS_SPECTRUM

I found no ERROR-level problems. Every theorem and quantitative claim I checked holds, using my own code plus high-precision and exact arithmetic. The findings are presentation problems, two overclaims in summaries, and small proof gaps. The repo is unchanged (`git status` clean; the `__pycache__` files were already there). My scripts are in this folder.

### Coverage (checked and found correct)
- **Heat definition:** Δ = diag(deg) − A, jump rate one per edge, H_t(U(G)) = tr e^{−tΔ}/|V|, uniformized with P = I − Δ/D and λ = tD. The code (`lazy_returns`, `dense_heat`) uses the same definition.
- **Locality lemma, order ≤ 2R (claim a):** I compared every rooted connected graph with ≤7 vertices (996 graphs) against its induced ball, for R ≤ 3 and three D values: 46,461 comparisons, 0 violations. The first possible discrepancy is exactly order 2R+1 for every R (R = 0..3 by brute force, cycles up to R = 5), and the same holds for Δ^m moments. The C_n sharpness example is right (order 3: 0 vs 1/4).
- **Two-sided enclosures (1), (2) and their local-error versions (claim b):** the monotonicity in E, the [0,mS] / [−CS,CS] intersections and the radius bounds are all valid, and the code matches the proofs. 151 library intervals (random signed finite combinations, L, L², L³ for t ≤ 3, products with finite graphs) all contain mpmath truth. 22 adversarial cases (perturbed oracles, coarse tolerance, t up to 10) are also valid.
- **Continuity (claim c):** the uniform-continuity bound (3) holds. The cycle obstruction holds: T_R X_n = 0 for n ≥ 2R+2 and H_t(X_n) > 0 numerically. Multiplicativity holds.
- **Sparse note:**
  - §1: affected roots and separated-group additivity. The code uses before/after distances, which is valid and tighter than using W.
  - §2: the trace-norm moment bound holds and is attained (ratio 1.0). (3), |H_t| ≤ q·min(1,2t), the sign rules, interval (4) and (5) hold; 78 random mixed-edit cases.
  - §3: formula (6) holds **iff** n ≥ 2r+2. Variation 4r and the p_{r,k} formula are right.
  - §5: the path/doubled-cycle identity holds (error 1e−14), as does (8). d_j(E) = [1−(1−4/D)^j]/2 is exact for D = 2, 3, 4, 7.
  - §6: (9), (10) and budget one hold. The binomial product-moment identity checked exactly for four choices of H.
  - §7: Krylov compression is exact through degree 2m and generally fails at 2m+1 (78 of 84 cases).
  - §9: the benchmark numbers match the JSON, and all references lie inside their intervals.
- **Interactions note:**
  - (1): the threshold n ≥ max(ℓ+2r, 2r+2) is sharp (fails at n−1 in 30/30 cases).
  - (2): first nonzero radius ⌊ℓ/2⌋+1, checked for ℓ = 1..8. Variations 4r and 4r+2ℓ for r ≥ ℓ, but not at r = ℓ−1.
  - (3)–(12): agree with finite cycles (n = 240, 400), the Bessel image series, exact walk moments, sympy coefficients and a Laplace-transform check.
  - (13): checked at histogram level with explicit cycles and all subsets, for k = 3..5, unequal gaps and negative positions (48 cases).
  - Prepared queries: the step count is monotone in t.
  - Library `relative_heat`: 109/109 intervals valid (E, I_ℓ, D_ℓ, multi-cut, connected interaction, E·L, E·L², E·U(H)).
- **Cospectral example (claim f):** A² = 4I+2J, the spectra, 2K3 vs C6 neighbourhoods, Q4 = 8 vs 0, p_{1,k} = 2·7^k, and the Q4 Leibniz rule all check out. For four backgrounds H, R□H and S□H are cospectral and the normalized Q4 difference is 1/2. The 674-check count is consistent (2×331 + 12).

### Findings (most severe first)
1. **OVERCLAIM — top-level `README.md`:84–86.** It says the analysis "shows why heat return requires degree and global variation control." The proof in EFFECTIVE §5 only shows that a degree bound alone does not give continuity. It does not show that global variation control (or degree control) is necessary. SPARSE §4 itself certifies relative heat for E and E·L^d, whose variation is unbounded, using an edit budget instead. Suggested wording: "requires uniform control beyond the topology (e.g. global variation or an edit budget)."
2. **MINOR — grid value (claim b): `python/README.md`:709 and quickstart line 54, linked from EFFECTIVE §6.**
   - The README says the square-lattice value is "approximately 0.2169320140."
   - The true value is (e^{−1}I₀(1))² = 0.21693201206578.
   - The certified interval [0.2169320120483, 0.2169320159240] contains it, only 1.75e−11 above the lower end. The quoted figure is the interval midpoint and is wrong from the 9th decimal.
   - The positive-mass midpoint is biased upward by about mT/(2(S+T)) because the upper bound assumes every tail walk returns. It would be better to quote the interval or ≈0.21693201.
3. **MINOR/OVERCLAIM — benchmark interpretation.**
   - EFFECTIVE:249–251 says neighbourhood extraction is expensive "especially in the irregular example." The data say otherwise: the torus is worst (1923.8 ms vs 14.3 ms dense, 135×) against the irregular graph (77.7 ms vs 0.89 ms, 87×).
   - The "computational gain" at EFFECTIVE:248 and the top README's "measurements demonstrate the value of retaining Cartesian factorizations" (:88–89) only hold against the library's own explicit-graph route. The factorized prism (6.82 ms) is about 3× slower than the dense eigensolve (2.32 ms). There is no factorization-aware conventional baseline; the cycle formula takes 0.03–0.06 ms.
   - The sparse-defect note and the Python README word this carefully.
4. **GAP (minor, claims true).**
   - SPARSE:313–315 extends the binomial identity to completed elements because it is "a finite local polynomial identity." This needs an unstated step: the moment functionals with fixed D are continuous on all of A. Approximants of a finite-measure Y need not respect the degree bound D₂, so (P_B^l)_oo is then unbounded, though only polynomially. Joint continuity of multiplication is also needed.
   - DEFECT_INTERACTIONS:226 ("matching the polynomial moments") needs H_t on K_(D,q) to equal its Laplacian-moment Taylor series. This follows from |δ_m| ≤ 2q·m·(2D)^{m−1}, which is not stated.
5. **MINOR — test description, EFFECTIVE:236–237.** "Heat moments are checked against dense rational full-matrix powers" refers to `test_locality_against_full_matrix`. That test uses only graphs with ≤4 vertices and the single choice D = maxdeg+1, where most balls are the whole graph. It is genuine but weak evidence for the lemma; my brute force above covers the gap.
6. **MINOR wording.**
   - SPARSE:215 "infinite marginal variation": each marginal is finite (4r); only their supremum is unbounded.
   - EFFECTIVE:139 "M retained Poisson terms, with indices 0 through M" is M+1 terms.

### Not verified
I could not check the literature citations (Cortinovis–Kressner–Massei Thm 2, Schaden Eq. 5, Mančinska et al. after Cor 7.4, Rao–Teh §3.1) because the proxy blocks arxiv.org. I did verify the content attributed to the Cortinovis–Kressner–Massei theorem (trace exactness through degree 2m) both analytically and numerically.
