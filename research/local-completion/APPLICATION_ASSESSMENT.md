# Applications and physical readings: an assessment

1 October 2026. This note records two assessments of the local Cartesian
graph completion, made after the positive-cone checkpoint:

1. a search for real-world uses;
2. a structure-driven derivation of the physics and network behaviour the
   completion implies, with each derivation checked against existing
   mathematics.

It also records the new mathematical observations that arose, with their
status. The literature verdicts are assessments, not theorems. New
mathematical statements are **Proposed** in the sense of the
[development path](../../reviews/local-completion-2026-09-30/DEVELOPMENT_PATH.md)
unless marked classical. Their finite fixtures are in
[verify_application_assessment.py](verify_application_assessment.py), with
results in [application_assessment_results.json](application_assessment_results.json).

The work was done in a Claude Code session with four research subagents
(two per assessment, 78 web lookups in total). A source counts as opened
only if its passage was read; quotations were spot-checked against saved
copies. Section 7 lists the sources and the limits of the search.

## Summary

- **No real-world use was found** where the completion, as it stands or
  with one modest extension, would do better than existing methods. None of
  14 candidates scored 3 or more out of 5 (Section 2).
- **Read through its own structure, the completion is an algebra of
  non-interacting composite systems and their defects.** For networks, it is
  an algebra of independently composed networks. Every phenomenon derived
  from that reading is already covered by existing mathematics. What remains
  are lattice-specific exact formulas and three mathematical questions
  (Section 3).
- **No phenomenon was found that depends on both the graph structure and the
  arithmetic.** Cartesian multiplication is independent composition: a
  uniform root of a product is an independent pair of roots, and the link of
  a product vertex is the disjoint union of the factor links. Because the
  characters separate points, everything measurable reduces to classical
  character values.
- **What may be new is the arithmetic itself**: this particular completion
  and its theorems. The standard for that claim remains the one in the
  [literature review](LITERATURE_REVIEW.md): not located is not the same as
  new.
- **Directions** (Section 6): marks, and interacting components, in which
  case the most natural form is a weighted defect along the diagonal of a
  product.

## 1. Scope: the construction versus the library

The search brief inherited a list of limits that mixed properties of the
construction with properties of the `graphlocal` library. They separate as
follows.

| Stated limit | Belongs to | Status for the construction |
| --- | --- | --- |
| Only Cartesian products of small pieces | Library inputs | Positive mass-one elements are exactly the unimodular laws with finite ball-size moments ([REPRESENTATION_THEOREM §3](REPRESENTATION_THEOREM.md)). The fcc, bcc and honeycomb lattices and random media such as diluted lattices are therefore elements. Products add factorization where it exists. |
| Combinatorial Laplacian only | Notes and library | Walk sums for any operator fixed by local structure (adjacency, a potential such as 4 − deg) are local functionals of the same kind. Their continuity is not proved in the notes. |
| Bounded degree | Library certification | The completion allows unbounded degree when every polynomial ball-size moment is finite. Heavy-tailed degree laws are excluded. |
| Return-to-start values and whole-graph totals only | Library API | Walks from a root to a structurally located vertex are rooted local observables. Data attached to a named site needs marks. |
| Slower than sparse linear algebra | Library implementation | Not a property of the construction. |

The genuine limits of the construction are:

- **The topology is local.** Global quantities are not continuous
  observables: spectral gaps, logical error rates, multigrid convergence
  factors, spectral radii, long-time transport.
- **The graphs are unmarked.** Jump rates, chemical species, spring
  constants, resistances and site-specific sources all need edge or vertex
  marks.
- **For finitely many defects on periodic backgrounds, exact answers already
  exist** from Green's-function (Dyson, T-matrix), low-rank Krylov and
  spectral-shift methods ([SPARSE_DEFECT_COMPARISON](SPARSE_DEFECT_COMPARISON.md)).
  The relevant question is whether the completion gives answers they do not.

## 2. Real-world uses

**Question.** Is there a question practitioners need answered, where current
methods demonstrably struggle, and where the completion (or a modest
extension) would do better? A mathematical problem with an application label
does not count.

**Scale.** 5 = practitioners struggle with a quantity the completion
computes, on graphs it supports, and it would plausibly beat current methods.
4 = the same after a modest extension. 3 = real need and documented struggle
that the completion's distinctive capability addresses, but it needs a
substantial extension or its advantage is unproven. 2 = real need, but no
documented struggle, or one the completion does not address. 1 = no real
need, or a structural mismatch.

| Candidate | Evidence (opened unless marked) | Why it falls short | Score |
| --- | --- | --- | --- |
| Vibrational binding free energies of defect pairs and clusters | "a satisfactory convergence with respect to the size of the supercell could not be reached" ([Posselt et al.](https://arxiv.org/pdf/1608.00720)); dense Hessian diagonalization "is too costly for supercell convergence" ([Torabi et al. 2026](https://arxiv.org/pdf/2605.26588)) | The exact interaction term on an infinite lattice is the right kind of quantity. But it needs vector couplings, continuous weights and a log-determinant. Green's-function determinant formulas are exact for a finite change (background knowledge), and the real bottleneck is the cost of the force constants. | 2 |
| Chip power-grid voltage-drop verification | "The difficulty, with locality, is to develop rigorous methods for determining exactly where the neighborhood is" ([Najm group, DAC 2009](https://www.eecg.utoronto.ca/~najm/papers/dac09-nahi.pdf)); what-if edits use "an empirically-chosen safety factor" ([Boghrati–Sapatnekar 2010](http://www.ece.umn.edu/users/sachin/conf/aspdac10bb.pdf)) | The closest stated wish, but the same 2009 method already "guarantees the specified over-estimation δ for all entries". Pads, loads and resistances need marks. | 2 |
| Variance normalization of lattice spatial models (LatticeKrig) | "this normalization step constitutes a subsequent computational bottleneck" ([2024](https://arxiv.org/pdf/2405.13821)) | A native fit (square lattice, constant shift, edge and corner corrections), but the same paper's Kronecker and FFT methods are exact or have mean error near 0.01%, which its authors call "essentially negligible". | 2 |
| Vacancy-mediated solute diffusion | One solute: "This Dyson equation solution is exact" ([Trinkle](https://arxiv.org/pdf/1608.01252)). Concentrated alloys: kinetic Monte Carlo "is hindered by the low occurrence of important events" ([Athènes et al.](https://arxiv.org/pdf/2112.01978)). Small clusters: KineCluE ([abstract](https://arxiv.org/abs/1809.05324)). | The struggle is dense disorder and noise, not a few defects. | 2 |
| Lattice Green's-function boundary conditions for dislocations | Periodic cells "may not accurately reproduce the correct bulk response to an isolated defect" ([abstract](https://arxiv.org/abs/cond-mat/0607388)) | Already solved with error control ([abstract](https://arxiv.org/abs/1005.4339), [abstract](https://arxiv.org/abs/2210.05573)); long-range tails are the worst case for a local method. | 2 |
| Gene flow across barriers | Ringbauer et al. 2018, PMC5844333 (abstract only) | No computational struggle; a straight barrier reduces to a one-dimensional problem. | 2 |
| Defective square-lattice qubit chips | Chips are ranked with "a modified version of breadth-first search" ([Lin et al.](https://arxiv.org/pdf/2305.00138)) | The needed quantities are global, and already cheap. | 2 |
| Landscape connectivity in ecology | "computational constraints hinder these advances" ([abstract](https://arxiv.org/abs/1906.03542)) | Dense, continuous per-pixel resistances. | 2 |
| Diffusion-MRI simulation | Monte Carlo cost in voxelized cells ([abstract](https://arxiv.org/abs/2012.06478)) | Dense irregular geometry; a flat membrane is classical. | 2 |
| Kronecker-sum Markov chains | "limited to about d < 25 automata" ([Georg et al.](https://arxiv.org/pdf/2006.08135)) | Directed, weighted, interacting everywhere; the output is a global distribution. | 1 |
| Quantum-walk search | The cited papers concern general, Johnson and strongly regular graphs ([abstract](https://arxiv.org/abs/2203.14384)) | Marked vertices already reduce exactly to a small matrix; the quantities are global. | 1 |
| Multigrid analysis near holes | [Brannick et al.](https://arxiv.org/pdf/1310.8385) never mention holes or cuts | The convergence factor is global. | 1 |
| Crystal graph networks and periodicity | "in principle the GNNs cannot capture the long periodicity" ([Gong et al.](https://arxiv.org/pdf/2208.05039)) | The missing information is global or geometric; descriptors already fix it. | 1 |
| Fitness landscapes | Robustness "is given by the matrix spectral radius" ([abstract](https://arxiv.org/abs/adap-org/9903006)) | A global eigenvalue; missing genotypes are the majority, not sparse. | 1 |

An earlier, unrecorded snippet-based search had described four of its
sources wrongly. PRA 102, 022227 is [arXiv:1807.05957](https://arxiv.org/abs/1807.05957),
a general hitting-time result. arXiv:1310.8385 never mentions holes or cuts.
arXiv:2203.14384 concerns Johnson graphs, and arXiv:1403.2228 strongly
regular graphs.

## 3. Structure-driven derivations

The second assessment let the structure lead. Its physical and network
reading:

| Structure | Reading |
| --- | --- |
| Disjoint union | Placing independent systems or networks side by side |
| Cartesian product | Composing independent subsystems: the generator L⊗I + I⊗L; a uniform root of a product is an independent pair of roots |
| Heat trace (a ring homomorphism) | Single-particle partition function at inverse temperature t |
| Positive mass-one elements | Statistically homogeneous random media (unimodular laws) |
| Signed elements | "System minus reference": excess quantities and defects |
| Inclusion–exclusion elements | Irreducible many-body interactions |
| Kernel of the spectral characters | Structure invisible to every non-interacting probe |

| Derived consequence | Already covered by | What is left |
| --- | --- | --- |
| Thermodynamics of composites: partition functions add and multiply, free energy is extensive, exp(zG) is the ideal gas with Gibbs' 1/N! | Textbook statistical mechanics | Nothing |
| Densities of states convolve; spectral dimensions add | Kronecker-sum spectra | Nothing |
| Box boundary terms multiply; the corner term is the square of the edge term | Separation of variables; Kac's corner terms | Nothing |
| Irreducible interactions among several defects | Schaden's irreducible many-body Casimir energies: "Only loops that touch all N objects … contribute" ([arXiv:1011.2475](https://arxiv.org/pdf/1011.2475)); sign failures are the case he names, boundary conditions "like Neumann's" | Lattice-specific exact formulas: signed covering-word onsets and their cancellations, the tree coefficient, the leaf-parity sign rule, and Section 4.1. All concern the fixed-time heat trace, not a measured force or free energy. |
| Excess thermodynamics without an excess density of states | Spectral-shift and surface-density-of-states theory ([Kostrykin–Schrader](https://arxiv.org/pdf/math-ph/0011019); [Poltoratski](https://arxiv.org/pdf/math/9601206)) | An explicit deterministic example (Section 4.7) |
| Structure invisible to non-interacting probes (rook vs Shrikhande) | Strongly regular graphs; Weisfeiler–Leman tests: "the nodes in the Rook's 4x4 graph are incident to 4-cliques" ([Papp–Wattenhofer](https://arxiv.org/pdf/2201.12884)); interacting two-particle walks (Gamble et al., abstract) | The \|t\| < ½ unit threshold of [GEOMETRIC_ARITHMETIC §5](GEOMETRIC_ARITHMETIC.md) is internal to the algebra, with no physical counterpart. |
| Products as statistical independence: convolution, additive cumulants, a central limit theorem, squares | Aldous–Lyons independent product (Proposition 4.11); local product recognition ([Hellmuth–Imrich–Kupka](https://arxiv.org/pdf/1303.6803)); Wagner–Stadler local factorization | A factorization theory for extremal unimodular laws (Section 4.4) |
| Realizability of local statistics | dK-series, joint-degree realizability, soficity; [STRICT_POSITIVE_CONES](STRICT_POSITIVE_CONES.md) for the simple-graph case | A local radius-one obstruction (Section 4.5) |
| Asymptotic comparison of Cartesian powers | Strassen, Zuiddam and Fritz asymptotic spectra | The natural setting is trivial; a nontrivial one is proposed in Section 4.6 |
| Synergy between link failures | Matrix-function perturbation (Arrigo–Benzi; Schweitzer); power-grid double outages "can amplify or attenuate" ([Kaiser et al.](https://arxiv.org/pdf/1909.00774v1)) | The exact Laplacian tree formula |

## 4. New observations

### 4.1 Bond cuts are site deletions for the edge operator

Let B be an oriented incidence matrix of G and K = BᵀB, the edge-space
operator. For S ⊆ E, the Laplacian L_{G∖S} = B_{E∖S}B_{E∖S}ᵀ and the principal
submatrix K[E∖S] = B_{E∖S}ᵀB_{E∖S} have the same nonzero spectrum. Hence

    Tr e^{-tL_{G∖S}} = Tr e^{-tK[E∖S]} + |V| - |E∖S|.

For k ≥ 2 cuts the affine term cancels in inclusion–exclusion. So the k-cut
heat interaction H_F of the notes equals the interaction for deleting the k
sites F from K, which is a Dirichlet condition on the edge space.

**Proposition (Proposed).** Suppose every vertex of G has degree at most two,
and the k ≥ 2 cut edges lie in one component. Then (−1)^k H_F(t) > 0 for every
t > 0.

*Proof sketch.* Orient each path and cycle consecutively. Every off-diagonal
entry of K is then −1 or 0, so −K generates an entrywise positive
semigroup. Expand e^{−tK[E∖S]} over jump paths killed on entering S. In the
alternating sum over S ⊆ F, a closed path survives only if it visits every
site of F. Each surviving path contributes a positive weight with overall
sign (−1)^k. This is Schaden's argument (arXiv:1011.2475, eqs. (5) and (15))
with edges in place of objects. Some closed path visits all of F because F
lies in one component.

**Branch points.** At a vertex of degree three or more, three incident edges
give off-diagonal entries of K whose product is +1 under every orientation,
since each entry is s_a(v)s_b(v). So no orientation makes K a Z-matrix and
the positivity argument fails. This accounts for the sign violations in the
notes. The k-edge star has H = e^{−t}(1−e^{−t})^k > 0 for every k
([BRANCHING_DEFECTS §3](BRANCHING_DEFECTS.md)). The
[HIGHER_INTERACTION_GEOMETRY](HIGHER_INTERACTION_GEOMETRY.md) §6 fixture is
positive at small t. The tree theorem there (eq. (11)) gives the small-time
sign (−1)^{k−ℓ}, which agrees with (−1)^k exactly when ℓ is even. The line
law of [DEFECT_INTERACTIONS](DEFECT_INTERACTIONS.md) (13) is the
degree-two case.

*Fixtures.* The trace identity is checked for all cut subsets of six graphs,
at powers 1–10. The orientation criterion is checked by brute force on eight
graphs. Certified heat signs on paths and cycles agree with (−1)^k for k = 2,
3, 4 at t = 1/4, 1 and 3. The star and the §6 fixture violate it.

### 4.2 Heat-trace signs do not transfer to free energies

For edges e ≠ f with G − e − f connected,
τ(G)τ(G−e−f) ≤ τ(G−e)τ(G−f), where τ counts spanning trees. This is the
classical negative correlation of edges in uniform spanning trees
(Kirchhoff; Feder–Mihail; background, not re-checked). The two-bond
interaction of log τ, which is the Gaussian free energy up to sign and
constants, therefore has a fixed sign. The two-bond heat interaction does
not. In the seven-vertex graph G7 of the verifier, cutting 04 and 24 gives
certified heat interactions +0.00923 at t = 1/2 and −0.00419 at t = 1, while
the spanning-tree ratio is 840935/857064 < 1. Three-bond log τ interactions
take both signs (345 positive, 769 negative and 35 zero over K4, K5, the
wheel W6, the Petersen graph and G7). Physical sign statements need the
integrated quantity, not the fixed-time heat trace.

### 4.3 The adjacency interaction is sign-definite (elementary)

Each closed walk counted by Tr exp(βA) survives the deletion of S exactly
when it uses no edge of S. Inclusion–exclusion therefore gives

    Σ_{S⊆F} (−1)^{k−|S|} Tr e^{βA(G∖S)} = (−1)^k Σ_W β^{|W|}/|W|!,

summed over closed walks W that use every edge of F. The communicability
(Estrada) interaction is therefore sign-definite, and its onset order is
exactly the length of the shortest closed walk covering F. For the Laplacian
heat trace, that length is only a lower bound, because deleting an edge also
changes diagonal terms; it is exact on trees (HIGHER_INTERACTION_GEOMETRY
§§2–4). The fixtures check the identity through power 14; onsets range from
4 to 8.

### 4.4 Factorization from local data (Proposed; bears on P1 and D2)

- **Local statistics cannot certify a global product.** The circulant
  C₁₀₁(1, 10) is vertex-transitive and Cartesian-prime, since 101 is prime.
  Its rooted balls of radius at most 4 nevertheless equal square-lattice
  balls (checked); radius 5 differs. In general C_N(1, k) with N prime and
  k ≈ √N looks like Z² out to a radius proportional to √N. These prime
  graphs therefore converge locally to Z², so Cartesian primality is not
  preserved by local limits. A factorization theory built from local data
  concerns local, bundle-like product structure.
- **Mixtures factor non-uniquely.** With H = K₂/2,
  (1+H+H²)/3 · (1+H³)/2 = (1+H)/2 · (1+H²+H⁴)/3. All four factors are
  positive mass-one elements. Unique factorization can be expected at most
  for extremal (ergodic) laws. Aldous–Lyons define the independent product
  and show that it preserves extremality. No factorization or identifiability
  theory for extremal laws under that product was found.

### 4.5 A radius-one obstruction to positivity (Proposed)

No graph, finite or infinite, has every vertex link equal to the path P₃.
Suppose v has link a–b–c and a also has link P₃. Then a has a third
neighbour x. It is not v's neighbour, since a and c are not adjacent. The
link of a must therefore be the path v–b–x, so b is adjacent to v, a, c and
x and its link is not P₃. By unimodularity, a law whose root almost surely
has link P₃ would have that link at every vertex. So the point mass at the
cone over P₃ is a radius-one marginal of the completion
([RADIUS_ONE_STRUCTURE §2](RADIUS_ONE_STRUCTURE.md)), but not of any
positive element. Positive realizability thus has local obstructions in
addition to the global soficity obstruction of STRICT_POSITIVE_CONES.

### 4.6 Asymptotic comparison of Cartesian powers (unchecked analysis; bears on M4)

- **Homomorphism and covering preorders** fail Strassen's embedding of the
  natural numbers: edgeless graphs collapse, or comparison reduces to
  divisibility.
- **The Strassen preorder of injective maps sending non-edges to non-edges**
  is compatible with the Cartesian product, but its spectrum is the single
  point |V|. The reason is that |V|ⁿ/χ(G) ≤ α(G^□ⁿ) ≤ |V|ⁿ, using Sabidussi's
  χ(G□H) = max(χ(G), χ(H)). The nontrivial asymptotics are at the ratio scale
  (the Hahn–Hell–Poljak ultimate independence ratio; snippet only).
- **Subgraph inclusion** is compatible with both operations but is not
  Archimedean globally. On a finitely generated subsemiring, Fritz's
  Theorem 2.4 ([arXiv:2112.05949](https://arxiv.org/pdf/2112.05949v5)) should
  apply. Its monotone homomorphisms include Σ_v s^{deg v} for s ≥ 1, root-clique
  generating functions with variables at least 1, and Tr e^{tA} for t ≥ 0.
  The Laplacian heat traces and the disc characters are not monotone under
  inclusion.

### 4.7 Spectral shifts that are not of bounded variation

Kostrykin and Schrader find that the integrated density of surface states
"is a measurable locally integrable function rather than a signed measure or
a distribution". They note that whether it has bounded variation "remains
unclear". Poltoratski shows that "Any function u ∈ L∞(R), 0 ≤ u ≤ π with
compact support is the Krein spectral shift of some perturbation problem".
Shifts without bounded variation are therefore not special to this
completion. The cycle-series element of
[SPECTRAL_MEASURE_HOMOMORPHISM §5](SPECTRAL_MEASURE_HOMOMORPHISM.md) is an
explicit deterministic per-volume example. By the assessment's check, each
summand's shift is bounded by 1/(2N_j), and the total is bounded but not of
bounded variation.

## 5. Conclusions

- Neither assessment found a use, or a phenomenon, that depends on both the
  graph structure and the arithmetic. Cartesian multiplication composes
  independently, so graph statistics only add or convolve. The other
  standard products do the same: random walks on tensor and strong products
  are independent walks moving all coordinates at once.
- The distinctive content found so far is internal to the algebra, together
  with the exact lattice formulas above.
- **Withdrawn suggestion.** Giving the kernel of the spectral characters a
  physical meaning as a hidden sector, by analogy with the earlier ghost
  edges, is not recommended. Kernels of invariants exist in any graph ring
  (see the correction note
  [Ghosts and Phantoms](../../Ghosts_and_Phantoms__A_Correction_to_an_Exploratory_Framework_for_Completing_Graph_Arithmetic.pdf)).
  The rook–Shrikhande element is a phantom in that sense (16 vertices and 48
  edges each), and the move would apply equally to any algebra. The
  legitimate content behind the analogy is mathematical. The old ghost edge
  is the element U(ℤ) − 1 of this completion, and the old complete-graph
  route to it diverges. The ghost edge is transcendental, the stub has no
  inverse, and no ghost is nilpotent; see [GHOST_EDGE](GHOST_EDGE.md).

## 6. Directions

- **Marks.** A finite palette of edge and vertex marks covers most
  model-level needs (species, a few jump rates, per-layer resistances) and
  keeps exact arithmetic. The materials near-misses need continuous or
  matrix-valued marks. Marked unimodular networks are standard (Aldous–Lyons),
  so the representation theorem plausibly extends; this is unverified.
- **Interacting components.** No single-particle probe can distinguish the
  rook and Shrikhande graphs. Neither can two non-interacting particles: their
  symmetric squares are cospectral (Audenaert et al., abstract). Interacting
  bosons can (Gamble et al., abstract). In this algebra an on-site
  interaction is a weighted defect along the diagonal of G□G, which needs
  marks for its strength. Before investing, check it against cluster and
  virial expansions, the Beth–Uhlenbeck formula and the Bethe ansatz.
- **Mathematical questions with specialist audiences:** factorization of
  extremal unimodular laws (Section 4.4); a Fritz-type duality for Cartesian
  powers under subgraph inclusion (Section 4.6); the sign theory at branch
  points and its fate for log-determinants (Sections 4.1–4.2).
- **Cheap decisive tests, if applications remain a goal:** a scalar spring
  divacancy against Green's function plus a small determinant; and
  bulk/edge/corner variance tables against the published LatticeKrig FFT and
  Kronecker normalizations.

## 7. Sources and limits

Full texts opened: Trinkle (arXiv:1608.01252); Athènes, Adjanor and Creuze
(arXiv:2112.01978); Chattopadhyay–Trinkle (arXiv:2401.06046); Georg et al.
(arXiv:2006.08135); Brannick et al. (arXiv:1310.8385); Gong et al.
(arXiv:2208.05039); Posselt et al. (arXiv:1608.00720); Torabi et al.
(arXiv:2605.26588); Lin et al. (arXiv:2305.00138); the Najm group
(DAC 2009); Boghrati–Sapatnekar (ASP-DAC 2010); the LatticeKrig
normalization paper (arXiv:2405.13821); Schaden (arXiv:1011.2475);
Kostrykin–Schrader (math-ph/0011019); Poltoratski (math/9601206);
Papp–Wattenhofer (arXiv:2201.12884); Paladugu et al. (Nature Commun.
7, 11403); Aldous–Lyons (math/0603062); Hellmuth–Imrich–Kupka
(arXiv:1303.6803); Wagner–Stadler; Wigderson–Zuiddam; Fritz
(arXiv:2112.05949); Orsini et al. (arXiv:1505.07503); Kaiser, Strake and
Witthaut (arXiv:1909.00774); Schweitzer (arXiv:2303.01339).

Abstracts only: KineCluE, Trinkle 2008, Ghazisaeidi–Trinkle, Braun et al.,
Strikis et al., landscape connectivity (arXiv:1906.03542), diffusion MRI
(arXiv:2012.06478), quantum-walk search papers, Shajesh–Schaden,
Kenneth–Klich, Gamble et al., Audenaert et al., Rudinger et al., and the
Europe PMC record for Ringbauer et al.

Not opened: publisher pages behind bot challenges (APS, Science,
ScienceDirect, ACS, Wiley, ACM, IEEE), for which arXiv copies were used
instead; Boghrati–Sapatnekar 2014; Qian–Nassif–Sapatnekar 2003 (snippet only,
not used as evidence); Englisch–Kirsch–Schröder–Simon (known only through
Kostrykin–Schrader).

The searches were targeted, not exhaustive. They are strong evidence about
the fields examined, not a proof that no use exists.

Reproduce the fixtures from this directory:

```sh
python3 -S verify_application_assessment.py --output application_assessment_results.json
```
