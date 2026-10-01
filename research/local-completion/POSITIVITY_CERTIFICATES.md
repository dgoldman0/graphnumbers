# Positivity certificates, ghost mass and the finite–unimodular gap

1 October 2026. This note makes precise what the
[positive-cone checkpoint](STRICT_POSITIVE_CONES.md) (Step 6) means inside the
local Cartesian graph completion:

- which local inequalities can be certified;
- how much negative graph mass ("ghost mass") a finite description must
  carry;
- how both behave under the completion's multiplication.

It also records the exact correspondence with the Connes-embedding
literature that Step 6's external theorem comes from.

Results are **Proposed** in the sense of the
[development path](../../reviews/local-completion-2026-09-30/DEVELOPMENT_PATH.md):
the proofs are written below but unreviewed. The finite mechanisms they rely
on are checked by
[verify_positivity_certificates.py](verify_positivity_certificates.py), with
results in [positivity_certificates_results.json](positivity_certificates_results.json).

## Summary

1. **Certificates exist for the limit objects** (Theorem 1). A local
   statistic is nonnegative on every unimodular law of bounded degree
   exactly when, for every ε > 0, adding ε makes it a nonnegative function
   of a larger ball plus a rerooting difference, the divergence of a local
   transport. The convergence of the radius-by-radius linear programs is
   the hierarchy of Bowen–Chapman–Lubotzky–Vidick, which
   Lyons–Terlov–Vidnyánszky transported to neighbourhood statistics. The
   certificate form is its Farkas dual.
2. **They fail for finite graphs** (Corollary 2). Every inequality valid on
   all finite graphs of degree at most Δ has such certificates exactly when
   every unimodular law of that degree is sofic. By Step 6 this fails, and
   Step 6's separator is an inequality that every finite graph satisfies but
   that has no certificate.
3. **The same shape as Connes' embedding problem** (Section 3).
   Klep–Schweighofer certificates use sums of hermitian squares and
   commutators; here they use nonnegative ball functions and rerooting
   differences. For Schreier graphs, the trace identity φ(ab) = φ(ba) is
   literally a rerooting identity.
4. **Ghost mass is robustness** (Theorem 3). The least negative mass that
   finite signed approximations of a positive element must keep equals its
   robustness with respect to the sofic laws. By the standard robustness
   duality, this equals the largest normalized violation of an inequality
   valid on all finite graphs.
5. **Products** (Proposition 4). 1 + 2R is submultiplicative under the
   Cartesian product, and multiplying by a sofic law never increases R.
6. **Open questions** (Section 6): amplification under Cartesian powers (a
   parallel-repetition analogue), curing by sofic factors in the unlabelled
   setting, and finiteness of R.

Section 7 records what is credited to prior work. The new content is the
certificate form in this completion's transport language, the
Klep–Schweighofer-shaped dictionary for unimodular laws, the robustness
reading of Step 6's negative mass, and the product bound. All of it is
routine once stated, and none was found in the sources examined.

## 0. Setting

Fix a degree cap Δ ≥ 1.
- **Space.** Let X_Δ be the compact space of rooted connected graphs with
  degree at most Δ, in the local topology. Let 𝔅_ρ be the finite set of
  rooted ρ-ball types of degree at most Δ, with truncations π_{ρ,r}.
- **Local statistic.** A local statistic of radius r is a function
  f : 𝔅_r → ℝ. Write E_μ f for its mean under a probability law μ on X_Δ,
  which is the pairing Λ_x(f) for the corresponding element x.
- **𝒰_Δ**, the unimodular laws: the mass-one part of P_loc ∩ A_Δ
  ([REPRESENTATION_THEOREM §3](REPRESENTATION_THEOREM.md)).
- **𝒦_Δ**, the sofic laws: the mass-one part of P_fin ∩ A_Δ, that is, local
  weak limits of finite graphs of degree at most Δ
  ([STRICT_POSITIVE_CONES](STRICT_POSITIVE_CONES.md), Lemma 1).
- **Comparison.** Both sets are compact and convex, and 𝒦_Δ ⊆ 𝒰_Δ. For some
  Δ the inclusion is strict (Step 6). On these sets the completion's
  topology is local weak convergence.

**Rerooting differences.** A local transport F of radius s has divergence
∂F(o) = Σ_v F(o, v) − Σ_v F(v, o), which is a function of the 2s-ball.
Balance means Λ(∂F) = 0 for every local transport
([REPRESENTATION_THEOREM §1](REPRESENTATION_THEOREM.md)). Let 𝓗_ρ be the
span, as functions on 𝔅_ρ, of the divergences of transports with 2s ≤ ρ.
Let L_ρ be the probability vectors on 𝔅_ρ that annihilate 𝓗_ρ: the
balanced radius-ρ arrays. Each L_ρ is a polytope.

## 1. Certificates for unimodular laws

**Theorem 1.** Let f be a local statistic of radius r, and define
m_ρ(f) = min over p in L_ρ of ⟨p, f∘π_{ρ,r}⟩ for ρ ≥ r. Then m_ρ(f) is
non-decreasing in ρ and converges to m(f) = min over μ in 𝒰_Δ of E_μ f.
Consequently:

- (a) m(f) ≥ 0 if and only if, for every ε > 0, there are a radius ρ, a
  function g ≥ 0 on 𝔅_ρ and an element h of 𝓗_ρ with f∘π_{ρ,r} + ε = g + h;
- (b) if m(f) > 0, then f∘π_{ρ,r} = g + h exactly, with g ≥ 0 and h in
  𝓗_ρ, at some finite radius.

*Proof.*
- **Monotonicity.** A transport of radius s with 2s ≤ ρ also qualifies at
  radius ρ + 1, and its divergence there is the old one composed with
  truncation. So truncation maps L_{ρ+1} into L_ρ.
- **Upper bound.** The radius-ρ marginal of every unimodular law lies in
  L_ρ, by the mass-transport principle. Hence m_ρ(f) ≤ m(f).
- **Lower bound.** Suppose m_ρ(f) ≤ m(f) − δ for all ρ, and choose
  minimizers p_ρ. A diagonal subsequence makes every truncation π_{ρ,s} p_ρ
  converge to some q_s.
  - The q_s are consistent and each lies in the closed set L_s.
  - A consistent family of ball laws on degree-Δ balls defines a probability
    law on X_Δ.
  - This law annihilates every divergence, so it is balanced. At mass one,
    balance is unimodularity (REPRESENTATION_THEOREM §3).
  - Its mean of f is lim m_ρ(f) ≤ m(f) − δ, a contradiction.
- **Certificates.** The cone {p ≥ 0} ∩ 𝓗_ρ^⊥ is polyhedral, and its dual cone
  is {g ≥ 0} + 𝓗_ρ. Farkas' lemma turns m_ρ(f) ≥ −ε into
  f∘π + ε ∈ {g ≥ 0} + 𝓗_ρ. If m(f) > 0, some m_ρ(f) is already
  nonnegative. ∎

Each m_ρ(f) is a finite linear program, and the
[radius-two atlas](RADIUS_TWO_ATLAS.md) already describes the constraint
spaces 𝓗_ρ at small radius. The primal statement, that these programs
converge, is not new. It is the bounded-degree, unlabelled form of the
pseudo-IRS polytopes of Bowen–Chapman–Lubotzky–Vidick, Part I
([arXiv:2408.00110](https://arxiv.org/abs/2408.00110)): Lemma 2.16 gives
"computable polytopes", and Theorem 1.10(2) a non-increasing sequence
converging to the ergodic value. Their Remark 1.11 compares this with the
Navascués–Pironio–Acín (NPA) hierarchy. Lyons, Terlov and Vidnyánszky
([arXiv:2505.06215](https://arxiv.org/abs/2505.06215)) carry the same
computation to r-neighbourhood statistics of bounded-degree graphs. What
Theorem 1 adds is the Farkas-dual certificate form (a)–(b), written with
the completion's local transports.

**Two checked examples.** In both, a positive but unbalanced array violates
the inequality, so it is the balance that does the work.

- **Degree inequality.** E[Σ_{v∼o} deg v] ≥ E[2 deg o − 1] on every
  unimodular law. The certificate is
  Σ_{v∼o} deg v − 2 deg o + 1 = (deg o − 1)² + ∂F, with F(o, v) = deg(v)
  when v ∼ o. A point mass at the centre of the star K_{1,3} gives −2, while
  the uniform root on the same star gives +1.
- **Link inequality at degree three.** At degree at most 3,
  Pr[link = P₃] ≤ Pr[link ∈ {K₂, K₂ ⊔ K₁}]. The certificate is
  2[good] − 2[P₃] = (2[good] − received) + ∂F, where F(o, v) = 1 when v has
  link P₃ and o is an end of that path. Here received(o) counts the
  vertices with link P₃ that have o as an end.

  The slack 2[good] − received is nonnegative pointwise. Suppose o = a is an
  end of the path a–b–c forming the link of v. Then v and b are adjacent
  neighbours of o. A third neighbour x of o, if there is one, is not
  adjacent to v, whose neighbours are a, b, c with a not adjacent to c. Nor
  is x adjacent to b, which would give b degree four. So the link of o is K₂
  or K₂ ⊔ K₁, with K₂ = {v, b}. Every vertex with link P₃ that has o as an
  end lies in that K₂, so received(o) ≤ 2.

  The point mass at the cone over P₃ violates the inequality, consistent
  with [APPLICATION_ASSESSMENT §4.5](APPLICATION_ASSESSMENT.md).

## 2. Finite graphs: where certificates fail

Write m_fin(f) for the infimum of E_{U(G)} f over finite graphs G of degree
at most Δ. By density this equals the minimum over 𝒦_Δ.

**Corollary 2.** Every local statistic with m_fin(f) ≥ 0 has the
certificates of Theorem 1(a) if and only if 𝒦_Δ = 𝒰_Δ.

*Proof.*
- **If.** When 𝒦_Δ = 𝒰_Δ, m_fin(f) = m(f), so Theorem 1 applies.
- **Only if.** Suppose x ∈ 𝒰_Δ ∖ 𝒦_Δ. Separating x from the compact convex
  set 𝒦_Δ by a continuous function, and approximating it uniformly by local
  statistics, gives f with m_fin(f) ≥ 0 > E_x f ≥ m(f). Then Theorem 1(a)
  fails for every ε < −E_x f. ∎

Step 6 makes this concrete. Its separator, eq. (11), is ℓ = aV − Λ(ψ).
Restricted to degree-Δ graphs, this is the local statistic a − f. It is
nonnegative on every finite graph and negative on the witness x. So no
nonnegative-plus-rerooting certificate exists for it at any ε < −ℓ(x).

## 3. The correspondence with Connes' embedding problem

Klep and Schweighofer
([arXiv:math/0607615](https://arxiv.org/abs/math/0607615), abstract) showed
that Connes' embedding conjecture "is equivalent to the existence of certain
algebraic certificates for a polynomial in noncommuting variables" whose
trace is nonnegative on matrices, and that "These algebraic certificates
involve sums of hermitian squares and commutators." They also prove "that
they always exist for a similar nonnegativity condition where elements of
separable II_1-factors are considered instead of matrices". Theorem 1 and
Corollary 2 have exactly this shape:

| Connes' embedding | This completion |
| --- | --- |
| matrices | finite graphs |
| separable II₁-factors (tracial) | unimodular laws (balanced) |
| commutators, annihilated by traces | rerooting differences, annihilated by unimodular laws |
| sums of hermitian squares | nonnegative functions of the rooted ball |
| certificates always exist for II₁-factors | Theorem 1 |
| certificates for matrices ⇔ Connes' conjecture | Corollary 2: certificates for finite graphs ⇔ 𝒦_Δ = 𝒰_Δ |
| Connes' conjecture is false | P_fin ≠ P_loc (Step 6) |

Ozawa's survey ([arXiv:1212.1700](https://arxiv.org/abs/1212.1700)) states
the template compactly: "CEC holds true if and only if
H_d^fin = arch(k⟨x_1, …, x_d⟩_+ + K_h)", where K_h is spanned by
commutators. Corollary 2 is this statement with finite graphs in place of
matrices, nonnegative ball functions in place of hermitian squares, and
transport divergences in place of commutators.

Unimodularity is the trace property here. Aldous and Lyons
([math/0603062](https://arxiv.org/abs/math/0603062), §5) show that
E(T o, o) is a trace on the algebra of a unimodular network, and that
Tr(ST) = Tr(TS) "follows from the Mass-Transport Principle". For Schreier
graphs of the free group, the commutator row is literal. Let φ(g) be the
probability that g fixes the root. The root o is fixed by ab
exactly when the rerooted vertex o·a is fixed by ba. So invariance under
moving the root along an a-edge gives φ(ab) = φ(ba): the trace identity is a
rerooting identity. The verifier checks this mechanism on random finite
permutation actions. Step 6's external theorem comes from exactly this
setting. Bowen, Chapman and Vidick
([arXiv:2501.00173](https://arxiv.org/abs/2501.00173), abstract) use "a
variant of the compression technique developed in MIP*=RE" and note that "As
a byproduct, we are reproving the negation of Connes' embedding problem".

**Contrast with dense graph limits.** For dense limits, Lovász and Szegedy
([arXiv:0902.1327](https://arxiv.org/abs/0902.1327), abstract) prove that
every valid linear inequality between subgraph densities "can be
approximated arbitrarily well by another valid inequality that is a 'sum of
squares'". Hatami and Norine
([arXiv:1005.2382](https://arxiv.org/abs/1005.2382), abstract) show that
exact certificates can fail, using "the fact that there are positive
polynomials that cannot be expressed as sums of squares". So in the dense
setting approximate certificates always exist. In the local setting they
exist for unimodular laws (Theorem 1), but by Corollary 2 and Step 6 they
can fail even approximately for finite graphs. The local failure is at the
approximate level and comes from the MIP* = RE phenomenon. The dense one is
an exact-level, real-algebraic failure. Csóka
([arXiv:1108.4995](https://arxiv.org/abs/1108.4995)) notes that "For sparse
graphs, the closure of the set of distributions is convex, so contrary to
the dense case, linear inequalities are sufficient to completely describe
it". He also proves exact feasibility undecidable, both for limits of
finite graphs and for unimodular random graphs, which is consistent with
the ε in Theorem 1.

## 4. Ghost mass is robustness

For a finite signed combination y = Σ c_G G of connected graphs, Step 6
defines its negative vertex mass N(y) = Σ_{c_G < 0} |c_G||V(G)|. For a
positive mass-one element x of A_Δ define:
- its **persistent negative mass**, N*(x) = inf liminf N(y_n), over finite
  signed combinations y_n → x;
- its **robustness**, R(x) = inf{m ≥ 0 : x = (1+m)s − mq with s, q in 𝒦_Δ},
  with inf ∅ = ∞.

**Theorem 3.** For x in 𝒰_Δ,

    N*(x) = R(x) = sup_f (E_x f − max_𝒦 E f)⁺ / (max_𝒦 E f − min_𝒦 E f),

with the supremum over local statistics and the convention t/0 = ∞ for
t > 0. Moreover:
- R vanishes exactly on 𝒦_Δ;
- R is convex and lower semicontinuous on 𝒰_Δ;
- R does not depend on the cap, provided the cap is at least the maximum
  degree of x.

*Proof.*
- **N* ≥ R.** Let y_n → x with N(y_n) → m.
  - Apply the degree cutoff Q_Δ of STRICT_POSITIVE_CONES to every graph. It
    preserves vertex mass and fixes x. It is continuous, since
    p_{r,k}(Q_Δ y) ≤ p_{r+1,k}(y): the cut r-ball lies inside the original
    one and is determined by the original (r+1)-ball. It does not increase
    N.
  - Write the cut combinations as P_n − M_n, with P_n and M_n nonnegative
    combinations of graphs of degree at most Δ and V(M_n) = N(Q_Δ y_n).
  - If V(M_n) → 0, then M_n → 0 at bounded degree and x ∈ P_fin.
  - Otherwise the normalized parts lie in the compact set 𝒦_Δ. Extracting
    limits s and q gives x = (1 + m′)s − m′q with m′ ≤ m.
- **N* ≤ R.** Approximate s and q by finite graphs (Lemma 1 of Step 6). The
  combinations (1 + m)U(G_n) − mU(H_n) converge to x and have negative mass
  at most m.
- **Duality.** K_m = (1 + m)𝒦_Δ − m𝒦_Δ is compact and convex, and increases
  with m. Hahn–Banach separation of x from K_m, in the space of measures on
  X_Δ, gives the formula over continuous functions. Local statistics are
  uniformly dense in those by Stone–Weierstrass.
- **Remaining properties.** Convexity follows by mixing decompositions.
  Lower semicontinuity holds because R is a supremum of continuous
  functions. Independence of the cap follows by applying Q_Δ to a
  decomposition made at a larger cap. ∎

Step 6's bound (12), liminf N(y_n) ≥ −ℓ(x)/2, is the special case of the
duality obtained from its normalization 0 ≤ ℓ(G) ≤ 2|V(G)|. Theorem 3 says
the best such bound is exact. The minimal ghost mass and the best normalized
certificate failure are the same number.

The duality is the standard one for robustness measures. Vidal and Tarrach
introduced robustness as "the minimal amount of mixing with locally
prepared states which washes out all entanglement"
([quant-ph/9806094](https://arxiv.org/abs/quant-ph/9806094), abstract).
Howard and Campbell's robustness of magic,
"R(ρ) = min ||x||1 subject to Ax = b", for which "strong duality holds"
([arXiv:1609.07488](https://arxiv.org/abs/1609.07488)), corresponds to
1 + 2R here. No robustness measure of non-soficity, or of the gap between
finite-dimensional and commuting-operator correlations, was found. The
closest is the gap between the ergodic and sofic values of a single test in
Bowen–Chapman–Lubotzky–Vidick, which is one normalized violation rather
than a per-law robustness.

## 5. Products

**Proposition 4.** For x in 𝒰_Δ and y in 𝒰_Δ′,

    1 + 2R(x⊠y) ≤ (1 + 2R(x))(1 + 2R(y)).

In particular, R(x⊠s) ≤ R(x) for every sofic s.

*Proof.* Products of unimodular laws are unimodular, and
𝒦_Δ ⊠ 𝒦_Δ′ ⊆ 𝒦_{Δ+Δ′}, because U(G)⊠U(H) = U(G□H) and multiplication is
continuous. Let x = (1+m)s − mq and y = (1+n)s′ − nq′. Expanding,

    x⊠y = [(1+m)(1+n) s⊠s′ + mn q⊠q′] − [(1+m)n s⊠q′ + m(1+n) q⊠s′].

The second bracket has total mass m + n + 2mn, and both brackets normalize
to sofic laws by convexity. Hence R(x⊠y) ≤ m + n + 2mn. ∎

This is the analogue of the submultiplicativity of the robustness of magic,
"R(ρ1 ⊗ ρ2) ≤ R(ρ1)R(ρ2)" (Howard–Campbell, eq. (4)). Sofic ⊠ sofic is
sofic by the finite-graph product argument in Aldous–Lyons, Proposition
4.11.

## 6. Open questions

- **Amplification.** How does R(x^{⊠n}) grow? The bound gives at most
  ((1 + 2R(x))ⁿ − 1)/2. Finite graphs approximating x⊠x need not be
  products, just as optimal strategies for a repeated game need not be
  product strategies. This is the parallel-repetition question for these
  inequalities.
  - On the game side, Bowen–Chapman–Vidick (Remark 3.49, Lemma 3.50 of
    [arXiv:2501.00173](https://arxiv.org/abs/2501.00173)) relate the
    product of games to the Cartesian product of Schreier graphs, and use
    anchored parallel repetition to reduce game values.
  - No amplification result for sofic values or for R was found.
- **Curing.** Can x⊠s be sofic when x is not and s is?
  - *Edge-labelled Schreier graphs: no.* Take independent invariant random
    subgroups H₁ of F(S₁) and H₂ of F(S₂). The Cartesian product of their
    Schreier graphs is the Schreier graph of K = π₁⁻¹(H₁) ∩ π₂⁻¹(H₂) in
    F(S₁ ⊔ S₂), where π_i retracts onto F(S_i), and K ∩ F(S₁) = H₁.
    Restricting finite actions to F(S₁) is continuous and preserves
    finiteness, so a co-sofic K would make H₁ co-sofic. In graph terms,
    deleting the second factor's labelled edges is a local map. This short
    argument was supplied by the literature check; it was not found in a
    source.
  - *Unlabelled completion.* The same argument works whenever the factor of
    each edge of x⊠s can be read locally. Whether it always can is open.
- **Computability.** The programs m_ρ(f) bound the unimodular value from one
  side and finite graphs bound m_fin(f) from the other. Bowen–Chapman–
  Lubotzky–Vidick (Corollary 1.12) and Lyons–Terlov–Vidnyánszky already
  observe that a positive Aldous–Lyons answer would make such values
  computable. No explicit separating radius is known: Lyons–Terlov–
  Vidnyánszky pose "Find an explicit r > 0 such that Ar ≠ A′r" as their
  Problem 16. Whether R(x) is computable in general therefore inherits these
  undecidability results. Transferring them through Step 6's encoding needs
  a quantitative version of its decoder, which is not checked here.
- **Finiteness.** Is R(x) finite for every bounded-degree unimodular law, or
  can a unimodular law lie outside every K_m?

## 7. Literature and novelty

A targeted search (21 lookups; passages quoted above were read in the
opened papers or their abstracts) gives these verdicts.

| Claim | Prior work | Status here |
| --- | --- | --- |
| Convergence of the radius-by-radius programs | Bowen–Chapman–Lubotzky–Vidick, Lemma 2.16 and Theorem 1.10(2); Lyons–Terlov–Vidnyánszky for neighbourhood statistics | Credited; not new |
| Certificate form, Theorem 1(a)–(b) | Not found; the template is Klep–Schweighofer and Ozawa's Theorem 6 | Routine Farkas dual, stated in transport language |
| Corollary 2 and the Connes dictionary | Shape of Ozawa's Corollary 8; trace from mass transport in Aldous–Lyons §5; "every IRS induces a character" (BCLV, footnote 6) | Statement for unimodular laws not found |
| Dense contrast | Lovász–Szegedy (approximate certificates exist); Hatami–Norine (exact certificates fail) | Credited |
| Theorem 3 | Standard robustness duality (Vidal–Tarrach; Howard–Campbell; witness formulations by Brandão and Regula) | Applied to non-soficity, which was not found elsewhere |
| Proposition 4 | Howard–Campbell's submultiplicativity; Aldous–Lyons Proposition 4.11 | Routine |
| Curing, labelled case | Not found | Short argument recorded in Section 6 |

No source examined gives a per-law measure of non-soficity, or of the gap
between finite-dimensional and commuting-operator correlations.
Al-Safi–Short is not a source for the duality: it shows that non-signalling
correlations arise once positivity is relaxed, and leaves bounds on the
negativity open. Not opened: Lovász's book (page 358 is known only through
Lyons–Terlov–Vidnyánszky), Hatami–Lovász–Szegedy, Elek, Netzer–Thom,
Kun–Thom, Arzhantseva–Păunescu, Harangi, and the full texts of Brandão,
Regula and Abramsky–Barbosa–Mansfield.

## 8. Status and reproduction

Theorem 1, Corollary 2, Theorem 3 and Proposition 4 are Proposed, with the
proofs above. The verifier checks the finite mechanisms:
- divergences of local transports sum to zero on finite graphs;
- the two certificates hold pointwise (including every graph with at most
  six vertices and degree at most three for the P₃ certificate), and
  unbalanced positive arrays violate them;
- finite permutation actions satisfy the trace identity in its rerooting
  form;
- the product decomposition has the stated masses.

Reproduce from this directory:

```sh
python3 -S verify_positivity_certificates.py --output positivity_certificates_results.json
```
