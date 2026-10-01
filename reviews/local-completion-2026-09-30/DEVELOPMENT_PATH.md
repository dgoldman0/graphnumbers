# Development path for the local Cartesian graph completion

1 October 2026. A research plan written after the [review](README.md) of the work of 30 September 2026. It expands the three directions the review identified:

- a spectral (Gelfand-style) theory of local graph geometry;
- an algebraic perturbation theory for infinite backgrounds;
- a signed measure theory for unimodular random graphs.

**Status labels.**
- **Known**: established in the notes and confirmed by the review.
- **Proposed**: a claim with a proof sketch here, not yet written out or independently checked.
- **Proof checkpoint**: a detailed proof has been written and its stated finite fixtures checked; independent referee review is pending.
- **Conjecture**: no proof yet.
- **Problem**: an open question with no conjectured answer.

References marked † are named from memory and were not checked during the review.

**Agreed revision, 1 October 2026.** Apply the audit repairs before extending
the theory. The first foundational results are the radius-one description,
the positive-span characterization below, and the repaired finite-deletion
theorem. Proof sketches retain Proposed status until their hypotheses,
proofs, and claim-specific checks have been written out. The historical
referee reports remain unchanged; a repair register will record their
disposition and the evidence for each implemented correction.

**Follow-up decision, 1 October 2026.** Proceed through Step 4 before the
separate independent review. The atlas checkpoint below implements this
decision; the earlier audit backlog and review statuses remain open.

## 0. How the existing results fit

The explicit computations are the worked examples, and several of them supply lemmas the general theory will reuse:
- link additivity;
- the walk-cumulant well-order;
- the additive sphere and square coordinates;
- the local-regularity face;
- the unit criterion and the representation theorem.

The general theories need a few structural theorems that more examples will not produce. Almost all of them concern one family of objects, the rooted-ball monoids. Section 1 is therefore a prerequisite for Sections 2–4.

## 1. Foundation: the rooted-ball monoids M_r

**Setting.** M_r is the set of rooted r-ball types (finite connected rooted graphs with every vertex within distance r of the root), with product B ⋆_r D = B_r(B□D, (o_B, o_D)) and unit K₁.
- Truncation τ: M_{r+1} → M_r is a surjective monoid homomorphism (product-ball identity).
- E_r = ⋂_k ℓ¹(M_r, |V|^k).
- B_r ⊆ E_r is the closure of T_r(A_loc).
- A continuous character of A_ℂ is bounded by some p_{r,k}, so it is a character of some B_r.

### 1.1 Facts already available (Known)

- **Cancellative, torsion-free.** M_r carries a well-order strictly compatible with ⋆_r (MULTIPLICATION_AND_UNITS §3). Hence it is cancellative, and its group of fractions G_r is a totally ordered abelian group, so G_r is torsion-free.
- **Separation.** The walk cumulants K_F are additive under ⋆_r and separate M_r, so M_r embeds in a ℚ-vector space.
- **Bounded semicharacters.** Every semicharacter bounded by a polynomial in |V| has modulus ≤ 1. The reason: |V(B^{⋆n})| grows polynomially in n at fixed r.
- **Balance is real for r ≥ 2.** The complete degree-three radius-two slice has 443 ball types and balanced dimension 353, with 90 independent rerooting constraints. The new [atlas](../../research/local-completion/RADIUS_TWO_ATLAS.md) supplies explicit rooted-injection constraints, host basis IDs and executed exact rank certificates. The general finite-slice basis proof is at proof-checkpoint status.

### 1.2 Radius one (Proof checkpoint)

Written proof and constructive witnesses:
[RADIUS_ONE_STRUCTURE.md](../../research/local-completion/RADIUS_ONE_STRUCTURE.md).

- **M_1 is free.** A 1-ball is the cone over the root's link. In G□H the link at (g, h) is the disjoint union of the two links, because neighbours in different factors are never adjacent. Every finite graph occurs as a link. So M_1 ≅ (finite graphs, ⊔), the free commutative monoid on connected finite graphs.
- **Exact finite realization at radius 1.** In the finite cone over a link H with n vertices, the roots of degree n are all universal and have the same rooted ball; every other root has smaller degree. Subtract already-realized smaller-degree types and divide by the number of universal roots. Induction realizes every point mass using connected graphs of at most n+1 vertices. Thus B_1 = E_1. This direct proof avoids assuming that an arbitrary one-radius array already extends to an all-radius balanced family before invoking the representation theorem.
- **Consequence.** The characters that factor through radius 1 are exactly the assignments of a value in the closed unit disc D̄ to each connected link type, so the radius-1 spectrum is the product of closed discs D̄^𝒞. The Gelfand transform at radius 1 is a power series in countably many variables.
  - The degree map z^{deg}, the isolated-vertex character, the link characters and the neighbour-link polydisc retracts of GEOMETRIC_ARITHMETIC are all specializations.
  - So is the identification of the closed subalgebra generated by K₂/2 with the smooth disc algebra.
- **Scope.** The note proves density and character classification, not surjectivity of T_1(A_loc) onto E_1 or a continuous all-radius section. Exact finite fixtures give the full radius-one rank and explicit realizations for links through four vertices.

### 1.3 Radius r ≥ 2 (Atlas checkpoint and open problems)

- **P1, factorization.** M_r has finite factorization. Is factorization into ⋆_r-irreducibles unique, or is M_r free? Find the smallest non-unique factorization, or prove freeness.
- **P2, coordinates.** Which finite families of additive invariants separate types of bounded size? Express the sphere and square coordinates, root-edge component counts and link counts in terms of the walk cumulants K_F.
- **P3, faces.** Classify the faces of M_r: submonoids whose complement is an ideal, such as the local-regularity face. Faces index the boundary strata of the spectrum.
- **P4, balanced image.** Describe B_r inside E_r through the rerooting space. Compute its dimension in degree-bounded slices for r = 2, 3.
- **Method.** Build an exhaustive atlas of M_2 up to about 9 vertices at maximum degree ≤ 4. Record factorizations, relations, the rank of the additive-coordinate matrix, candidate faces and rerooting dimensions. Then conjecture a presentation.

**Step 4 checkpoint.** [RADIUS_TWO_ATLAS.md](../../research/local-completion/RADIUS_TWO_ATLAS.md)
and its source/data directory now contain every degree-four radius-two type
through nine vertices (29,032 types), plus the complete degree-three slice
(443 types). Every factorization in these slices is enumerated, with no
nonunique prime factorization found. This supports the freeness conjecture
on M_2 while leaving it open globally.

- **P2 progress.** New decorated-neighbor component counts are additive and
  give bounded semicharacters. They retain relative neighbor degrees,
  adjacency and common-neighbor multiplicities. Adding them increases the
  degree-four coordinate rank from 23 to 1,456 and the distinct vectors from
  3,622 to 9,505. Explicit six-vertex collisions remain. A complete separating
  family and the symbolic dictionary to K_F remain open.
- **P3 progress.** Interior-regular, bipartite and triangle-free faces are
  recorded, together with faces obtained by forbidding decorated component
  types. A classification of all faces remains open.
- **P4 progress, proof checkpoint.** At any radius, the supported slice on
  balls of size at most N and degree at most Delta has a basis consisting of
  histograms of connected hosts within those bounds that have an r-center.
  Rooted injective-count differences at different r-centers give all its
  constraints. This proves exact finite realization and a triangular
  reconstruction algorithm. The degree-four nine-vertex slice has dimension
  11,975 with 17,057 constraints by this theorem and the catalogue; the full
  numerical matrix is not evaluated. Smaller matrices are evaluated, and a
  complete radius-three degree-two control has dimensions 15 = 12 + 3.

The planned finite atlas is complete. It is not a proof of a general free
presentation, nor a resolution of the unrestricted infinite-dimensional
balanced image. New proofs await the separately deferred review.

## 2. Spectral theory

**Known.**
- The completion is a semisimple integral domain, and phase characters separate points.
- x is a unit exactly when T_r(x) is invertible in ℓ¹(M_r) for every r, which happens exactly when no local semicharacter vanishes on x.
- σ(x) = ⋃_r σ_r(x), with σ_r(x) inside the disc of radius ‖T_r x‖₁.
- The polynomial weights do not change units: the reciprocal test reduces to unweighted ℓ¹(M_r) at every radius (spectral invariance).
- The closed subalgebras generated by particular elements are identified: the smooth disc and polydisc algebras, entire functions in one or several variables, and the rook–Shrikhande disc of radius 2.

**T1, character completeness (Conjecture).** Every continuous character of A_ℂ has the form s∘T_r for a bounded semicharacter s of some M_r.
- Equivalently, for each r, every character of B_r extends to ℓ¹(M_r).
- Hewitt–Zuckerman† describes the characters of ℓ¹(M_r). The issue is B_r, because a proper closed subalgebra can have extra characters.
- The known unit criterion is a statement about one element at a time. What is needed is a version for tuples at each radius: if b₁, …, b_n ∈ B_r have no common zero among the semicharacters, then Σ b_i c_i = 1 has a solution with c_i ∈ B_r. This is a corona-type statement. Even the one-element version inside B_r is not automatic when the local spectrum surrounds 0.
- One possible route: a B_r-module projection E_r → B_r (a "balancing" conditional expectation). If it exists, T1 follows.

**T2, spectral geometry (Problem).** Describe the semicharacter space Σ_r.
- Nowhere-vanishing semicharacters correspond to homomorphisms G_r → ℂ^× of modulus ≤ 1 on M_r. Their log-moduli form the dual cone of M_r; their arguments form a torus.
- Semicharacters that vanish somewhere are supported on faces and give the boundary strata.
- Truncation embeds Σ_r in Σ_{r+1}. If T1 holds, the spectrum is the increasing union of these compact pieces.
- First target: compute Σ_2 on a degree-bounded slice.

**T3, functional calculus (Proposed; standard).** Holomorphic functional calculus in several variables (Shilov, Arens–Calderón†) passes to Fréchet locally m-convex algebras through their Arens–Michael decomposition (Waelbroeck†).
- It turns the scattered subalgebra identifications into one statement: f(x₁, …, x_n) ∈ A whenever f is holomorphic near the joint spectrum.
- The polynomial weights |V|^k account for smoothness up to the boundary in the disc-type cases.

**T4, automorphisms (Problem).** Classify the continuous automorphisms of A_ℂ. Each permutes characters and, by the INTRINSIC rigidity results, respects the neighbourhood-kernel filtration up to cofinal refinement.
- Decide whether the reflection H ↦ −H (z ↦ −z in the degree coordinate) extends.
- Restate the lifting criterion of REFLECTION_EXTENSION in terms of the cones of M_r and their compatibility with balance across radii.

**T5, reconstruction (Problem).** Find characterizations that use only the algebra and its topology for:
- the positive cones (see Section 4);
- the finite graphs, which are the basis of A₀.

The partial rigidity theorems of INTRINSIC_GRAPH_STRUCTURE are the starting point.

## 3. Perturbation theory for infinite backgrounds

**Known.**
- Single cuts generate entire-function algebras with spectrum ℂ: the line E, d-regular trees B_d, and the plane P.
- Distinct tree degrees are independent, and so are tree degrees ≥ 3 together with P.
- Regular-face characters at r ≥ 2 cannot establish E/P joint independence: E² + 2P ≠ 0 is killed by all of them, because B_r(ℤ²) = B_r(ℤ)^{⋆2}. The radius-one face is the whole monoid and does not satisfy that quadratic identity.
- Also known: interaction elements, the bridge-cut reduction, crossing cuts, heat and resolvent formulas, decay with separation, and certified evaluation.

**D1, general finite-deletion theorem (Proof checkpoint).**

The detailed proof is in
[FINITE_DELETION_ARITHMETIC.md](../../research/local-completion/FINITE_DELETION_ARITHMETIC.md).
It supplies the uniform threshold r ≥ |S|+1, exact affected-root count,
finite-component classification and the bounds below. Independent finite
fixtures include isolated vertices, isolated edges, a nonregular path and
an isolated lattice square. The following retains the proof outline.

*Setting.* Γ is infinite, connected, d-regular, bipartite and vertex-transitive. ε deletes a finite nonempty set of edges. D is the unnormalized defect (Γ with ε applied) − Γ. D is a limit of buffered finite differences, since unaffected roots cancel exactly.

*Claim.*
- Write T_r(D) = A_r − c_r δ_{B_r(Γ)}, where A_r is nonnegative with total mass c_r and c_r → ∞ is the number of affected roots. A_r may contain interior-regular atoms: deletions can split off isolated vertices, edges, or other finite regular components.
- The closed subalgebra generated by D is topologically isomorphic to the entire functions; real coefficients give the real form.
- σ(D) = ℂ, and f(D) is a unit exactly when f has no zeros in ℂ.

*Sketch.*
1. Let o be an affected root. Among the deleted edges inside B_r(o), pick an endpoint u* nearest to o.
   - Bipartiteness gives d(o, u*) ≤ r − 1.
   - A shortest path from o to u* avoids every deleted edge; otherwise a nearer endpoint would exist.
   - So u* stays interior after the edit, with degree < d.
   - For all sufficiently large r, an interior-regular positive atom must be the whole rooted graph of a split-off finite regular component. Prove a uniform radius threshold using the finite set of endpoints of deleted edges; a root in an infinite component can reach an unedited vertex along a simple path.
   - Every vertex of such a regular component has lost an incident edge. Their total vertex count N is therefore at most 2|ε|. N counts rooted mass, not components.
2. On the local-regularity face F_r use w^{deg root}, extended by zero outside F_r, for |w| ≤ 1. This is an ambient bounded semicharacter, and
   χ_{r,w}(D) = q_r(w) − c_r w^d,
   where q_r collects the regular positive atoms and |q_r(w)| ≤ N on the unit circle.
3. For |λ| < c_r − N, Rouché's theorem applied on |w| = 1 gives a solution of χ_{r,w}(D) = λ in the unit disc. Consequently, for a polynomial f,
   max_{|λ|≤c_r−N}|f(λ)| ≤ ‖T_r(f(D))‖₁.
   The closed-disc bound follows by continuity from the open-disc assertion. Since c_r − N → ∞, these estimates control all compact-open seminorms of entire functions.
4. Local submultiplicativity gives p_{r,k}(f(D)) ≤ Σ_n |a_n| p_{r,k}(D)^n. Cauchy estimates compare this sum with a supremum on a larger disc. The two bounds identify the closed generated algebra with the entire functions. Expanding ambient character discs detect all zeros; a zero-free entire reciprocal gives the inverse.

The finite-component exception changes the lower-bound argument, without
requiring that the edited background stay connected. For the line with both
edges at one vertex deleted, D = 1 + D′, where D′ is the vertex-deletion
defect; these generate the same unital closed algebra.

*Coverage and open extensions.* This covers edge deletions in ℤ^d, regular trees, the hexagonal lattice and their Cartesian products. Two extensions are Problems:
- **Non-bipartite backgrounds.** A deletion can then affect a ball only on its boundary sphere, which leaves the ball in F_r. The same sufficient argument works if c_r − a_r → ∞, where a_r is the total positive mass retained by F_r. Establishing that condition in further backgrounds is a separate problem.
- **Insertions and degree-preserving edits.** Affected balls can then stay regular.

**D2, joint defects (Proposed criterion, then Conjecture).**
- *Criterion.* For defects D₁, …, D_s over backgrounds Γ₁, …, Γ_s, suppose that for all large r:
  1. distinct exponent vectors give distinct products ∏ B_r(Γ_i)^{α_i}, and
  2. there are additive coordinates, nonnegative on F_r, that separate the B_r(Γ_i).

  Then the face argument gives the entire functions of s variables and joint spectrum ℂ^s. This is the MIXED_MEDIUM_ARITHMETIC §7 criterion in general form.
- *Conjecture.* Condition (1) holds whenever the Γ_i are pairwise non-isomorphic and Cartesian-prime. Its key ingredient would be a local version of unique prime factorization for connected infinite graphs under the weak Cartesian product (Imrich†).
- *Test cases at r = 2, 3:* (ℤ, hexagonal lattice), (T₃, ℤ²), (ℤ³, ℤ² □ T₃).

**D3, line and plane (Problem).** Decide whether E and P are algebraically independent. Regular-face characters cannot do it, so look for faces defined by the irregular atoms: the half-line ends of E against the cut-lattice balls of P.

**D4, spectral-measure homomorphism (Proposed on a subdomain).**
- Write H_t(x) = ∫e^{−tλ} dμ_x(λ). For normalized finite graphs, μ_x is the spectral measure averaged over roots. For finite-edit defects it is the relative trace measure, the Krein spectral shift†.
- Heat return is multiplicative and Laplace transforms are injective. So μ_{xy} = μ_x ∗ μ_y (additive convolution) wherever these measures exist with bounded total variation: finite graphs, finite-edit defects, and their products and controlled limits.
- Consequences:
  - heat and resolvent responses of product defects by convolution;
  - SPARSE_DEFECTS §6's binomial moment identity as a moment identity of convolution;
  - sign changes of interactions read from the signed spectral measure. In the HIGHER_INTERACTION fixture, the lowest atom (λ ≈ 0.5188, coefficient +1) fixes the sign for large t, which explains the second sign change.
- *Problem.* Extend μ_x to the whole controlled domain of PLANAR_DEFECTS §4. This needs a signed Hausdorff moment condition, not just moment growth bounds. Compare with density-of-states theory for unimodular graphs†.

## 4. Signed measure theory for unimodular random graphs

**Known.**
- Positive elements of mass one are the unimodular random rooted graphs with all polynomial size moments.
- An element has a finite signed measure exactly when its local variation is bounded across radii.
- The completion contains a closed complemented copy of ℝ^ℕ.
- Multiplication is the independent rooted product.
- P_fin ⊆ P_loc, where P_loc is the cone of positive balanced elements and P_fin is the closed cone generated by finite graphs.

**M1, strict inclusion P_fin ⊊ P_loc (route identified).**
1. At bounded degree, the mass-one part of P_fin is exactly the sofic laws. Normalized positive finite approximants are uniformly rooted finite graphs, and at bounded degree this topology is local weak convergence (COMPARISON_LEMMAS §2).
2. Bowen–Chapman–Lubotzky–Vidick (2024) give a non-sofic unimodular network. Encode its labels and orientations by rigid gadgets, keeping unimodularity, bounded degree and non-soficity.
3. Conclude.

The encoding is standard in spirit but must be written out and checked. Self-contained; likely the quickest substantial new result.

**M2, order structure (Problems).**
- **Non-generation (immediate consequence of Known results).** P_loc does not generate all of A_loc. Every positive element has radius-independent local variation equal to its finite vertex mass. A difference of two positives therefore has uniformly bounded local variation, whereas ‖T_r E‖₁ = 4r.
- **Positive-span characterization (Proof checkpoint).** P_loc − P_loc consists exactly of elements represented by a finite signed measure μ with ∫|B_r|^k d|μ| finite for every r and k. [POSITIVE_SPAN.md](../../research/local-completion/POSITIVE_SPAN.md) now proves the local-cylinder-to-edge-measure balance step, preservation of Jordan parts under the edge lift, and the application of Aldous–Lyons Proposition 2.2. Finite variation alone does not supply the absolute-measure moment hypotheses.
- **Normality and lattice structure (same checkpoint).** The defining seminorms are monotone on P_loc, so the cone is normal. Its span is a vector lattice under global measure Jordan decomposition, but the lattice modulus is discontinuous in the inherited local topology, as normalized cycle differences show.
- What order-unit structure does the bounded-variation part have?

**M3, generating-function dictionary (Proposed; definitional).**
- For a positive mass-one x and an additive statistic c: M_r → ℤ_{≥0}^m, the semicharacter z^c gives χ_z(x) = E[z^{c(root ball)}], the joint generating function of c at the root.
- Products become independent sums.
- For f with nonnegative coefficients, f(x) is a random Cartesian power of x with the number of factors distributed by f's coefficients. Example: e^{−t} exp(tH) is the uniformly rooted Poisson(t)-dimensional hypercube.
- *Development:*
  - limit theorems for additive local statistics under high Cartesian powers;
  - which characters are probabilistic (values in [0, 1] on [0, 1]^m);
  - the joint development of Sections 2 and 4.

**M4, comparing positive elements (Problem).** Characterize asymptotic (and catalytic) domination x^{⊗n} ≤ y^{⊗n} by monotone characters, in the framework of Strassen's spectral theorem and Fritz's abstract Vergleichsstellensätze†. Compare with Zuiddam's asymptotic spectrum of graphs, which uses a different product and topology.

**M5, signed moment problems (Problem).**
- Which signed balanced families have prescribed finite moments?
- The notes show two boundary cases. Moments determine local geometry under degree and radius bounds. Without bounds, every clique moment can vanish on a nonzero element.

## 5. Order of work

| Step | Deliverable | Depends on | Status |
|---|---|---|---|
| 0 | Audit repairs: certificate validation, reconstruction and axis compatibility; proof omissions and qualifiers; accurate documentation and independent evidence | FINDINGS.md and reproductions | First repair pushed as 2d1b5e2; remaining work in AUDIT_REPAIRS.md |
| 1 | Radius-1 theory (§1.2) | Cone induction; local product identity | Proof checkpoint with exact finite realization witnesses |
| 2 | Positive-span characterization M2 | Representation theorem; Jordan decomposition; involution invariance | Proof checkpoint; includes normality and discontinuous modulus |
| 3 | General finite-deletion theorem D1 | Face lemma; finite-component classification; Rouché bounds | Proof checkpoint with finite-component edge-case fixtures |
| 4 | Atlas of M_2 (§1.3) | Explicit monoid operations and independent canonicalization | Completed staged atlas: degree 4 through 9 vertices, complete degree 3; finite-slice and decorated-neighbor proofs await review |
| 5 | Spectral-measure homomorphism D4 on its subdomain | Existence of finite signed spectral measures; heat multiplicativity | Proposed; establish measure hypotheses first |
| 6 | Strict cone inclusion M1 | Verify Bowen–Chapman–Lubotzky–Vidick; explicit gadget encoding | Route identified |
| 7 | Character completeness T1 and spectral geometry T2 | Steps 1 and 4; semigroup-algebra literature | Open |
| 8 | Joint defects D2, D3 | Steps 3 and 4 | Criterion proposed; conjecture open |
| 9 | Automorphisms T4 and reconstruction T5 | Step 7 | Open |

Steps 1–4 now have written proofs and explicitly scoped finite evidence;
independent review is deferred as requested. The audit register preserves
unfinished performance, evidence and bibliography tasks. Step 4's planned
nine-vertex catalogue is complete, with larger degree-four and higher-radius
catalogues left as extensions. Step 5 is the next research milestone. The
later measure, soficity and character-completeness tasks retain their open
status.

## 6. Literature to read first†

- **Semigroup algebras:** Hewitt and Zuckerman (1956) on the ℓ¹-algebra of a commutative semigroup; Berg, Christensen and Ressel, *Harmonic Analysis on Semigroups* (1984); Dales, Lau and Strauss (2010).
- **Locally m-convex algebras and functional calculus:** Michael (1952); Arens–Calderón; Waelbroeck.
- **Affine monoids and cones:** Bruns and Gubeladze, *Polytopes, Rings and K-Theory* (2009).
- **Unimodular random graphs and soficity:** Aldous and Lyons (2007); Bowen, Chapman, Lubotzky and Vidick (2024).
- **Spectral shift and densities of states:** the Krein spectral shift and Birman–Krein formula; density of states and Lück approximation for unimodular graphs.
- **Asymptotic spectra:** Strassen; Zuiddam; Fritz.
- **Infinite-graph factorization:** Imrich, on the weak Cartesian product.

Much of Section 2 is probably standard once M_r is understood. Reading the semigroup-algebra literature first avoids re-deriving it and moves the originality to where it likely belongs: the combinatorics of M_r and the graph-specific computations.

## 7. Working practices

- State each general theorem with its hypotheses before computing further examples, and test the edge cases (r = 0, 1, small buffers, non-bipartite backgrounds) adversarially.
- Tie every verification check to a specific claim. Use independent oracles. Don't report check counts, and don't count tautologies.
- Keep summaries to what is proved, and keep the status labels above when results move into the notes.
- Preserve the degree restriction d ≥ 3 for the joint tree/planar theorem. Independence of E and P remains open, and regular-face characters cannot establish it.
- Record meaningful commit messages explaining the problem, change, validation and remaining limits. Push the agreed plan first, then publish validated work in separate checkpoints.
