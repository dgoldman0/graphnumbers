# Follow-up identification of the local graph completion

30 September 2026. Compared against the representation, domain, and unit
results at commit `0eaa1d600298013af53849381863b1f219da4a58`.
This supplements the [original review](LITERATURE_REVIEW.md); it does not
replace its bibliography or establish priority.

**The new results identify the local formal algebras and the normalized-edge
subalgebra with established constructions. No source examined identifies
the entire signed, balanced, weighted completion as a previously named
specific algebra.** The most precise placement presently justified is:

> A closed mass-transport-balanced subalgebra of a projective limit of
> weighted semigroup Fréchet convolution algebras, whose local ambient
> formal algebras are generalized power-series rings.

This is an application of established definitions to our construction.
It is not a name located in a paper for the whole object.

## 1. The exact object to identify

For each positive radius let $M_r=(\mathcal B_r,\star_r)$, where
$\mathcal B_r$ consists of finite connected rooted simple graphs of root
eccentricity at most $r$, and

$$B\star_r D=B_r(B\square D,(o_B,o_D)).$$

Write $|B|$ for vertex count and put

$$
w_k(B)=|B|^k,\qquad
L_r=\bigcap_{k\ge1}\ell^1(M_r,w_k),\qquad
\mathscr W=\varprojlim_r L_r.
$$

The bonding maps push coefficients forward under ball truncation. Radius
zero is the scalar coordinate. Each $L_r$ carries the norms
$\sum_B |a(B)|w_k(B)$; the projective limit carries all these seminorms.
The [representation theorem](REPRESENTATION_THEOREM.md) gives the exact
topological algebra identification

$$
\mathcal A_{\mathrm{loc}}
=\{a\in\mathscr W:\Lambda_a(\partial F)=0
\text{ for every local transport }F\}.
$$

Here transports have finite range and polynomial growth as defined in
that theorem. Their divergence functionals are continuous, so this is
a closed subspace; closure under multiplication follows from the
representation theorem and the original graph product. Merely matching
one $L_r$, the positive probability slice, or the finite graph ring does
not identify this entire topological algebra.

## 2. The strongest exact matches

### Generalized power series at each radius

The [multiplication note](MULTIPLICATION_AND_UNITS.md), Section 3, constructs
a strict multiplicative well-order on $M_r$. Consequently the full formal
coefficient algebra is exactly $\mathbb R((M_r))$ in
Imrich–Klep–Smertnig [F1], Definition 3.4 and the construction following it.
Their Propositions 3.7–3.8 supply the general ordered-series domain argument.

The underlying construction is also Ribenboim's generalized power-series
construction, presented in [F2], Section 2. Its support requirement is
automatic here: every subset of a well-ordered total set is artinian and
has no nontrivial antichain. The coefficient formula is precisely

$$
(a*b)(B)=\sum_{C\star_r D=B}a(C)b(D).
$$

Each sum is finite. Our independent graph proof bounds the sizes of both
factors by $|B|$, giving only finitely many possible rooted simple graphs.
The identification is the identity on coefficients, with exactly the same
addition and multiplication.

This changes the attribution of the domain proof: the least-support
coefficient argument is standard. The step requiring work for this
construction is producing separating additive rooted invariants and a
multiplicative **well-order** on the truncated ball monoid. The passage
from local domains to the compatible completion then uses a common radius
at which both nonzero factors remain nonzero. The source comparison does
not establish priority for that graph-specific ordering step.

The full generalized-series ring imposes no weighted summability or
cross-radius balance. Those conditions remain essential to the object
under study. In particular, formal inverses need not belong to it.

### Weighted convolution Fréchet algebras

The inequality

$$|B\star_r D|\le |B||D|$$

makes every $w_k$ a submultiplicative positive weight. Thus
$\ell^1(M_r,w_k)$ is exactly a weighted semigroup Banach algebra as defined
in [F3], Section 2. Intersecting an increasing sequence of such weighted
spaces gives the familiar Beurling–Fréchet construction; [F4], Example 1.2,
and [F5] treat established versions of this procedure.

Applying these definitions gives the displayed description of $L_r$ and
$\mathscr W$. Neither source supplies the specific monoids, truncation
maps, or finite-graph approximation characterization. The full algebra
also belongs to the established class of commutative Fréchet
Arens–Michael algebras. It is not the universal Arens–Michael envelope of
the bare finite graph algebra: component count is a multiplicative
functional whose absolute value is a submultiplicative seminorm, but it
is discontinuous here. See [the original comparison](COMPARISON_LEMMAS.md).

### The normalized edge generates the classical smooth disk algebra

For $H=K_2/2$, the multiplication note proves

$$
\overline{\mathbb R[H]}
\cong
\left\{\sum_{n\ge0}c_nz^n:
c_n\in\mathbb R,\quad
\sum_n|c_n|(n+1)^k<\infty\ \forall k\right\}.
$$

This is the real-coefficient form of $A^\infty(\overline{\mathbb D})$,
explicitly treated in Bhatt–Patel [F4], Example 1.5, using smooth boundary
Fourier series. The identification is a topological algebra isomorphism,
not just an analogy with analytic functions.

The graph construction additionally supplies a continuous retraction onto
this subalgebra via the degree generating function. Its units are the
functions without zeros on the closed disk, and it is inverse-closed in
the full completion. These statements concern a particular subalgebra;
they do not identify the entire completion with $A^\infty$.

## 3. Concrete obstructions to over-identification

The following deductions use our established formulas. They test proposed
identifications more strongly than differences in notation do.

### Original connected-component series have a different convergence

Let $M_{\mathrm{fin}}$ be the monoid of connected finite graphs under
Cartesian product. In the full component-series algebra
$\mathbb R((M_{\mathrm{fin}}))$, equipped with its coefficient topology,

$$\frac{C_n}{n}\longrightarrow0.$$

Every fixed connected-graph coefficient is eventually zero. In our
completion the same sequence converges to the infinite-line element
$L$, with $V(L)=1$. Therefore the canonical finite-graph embedding into
that component-series algebra cannot extend to a **continuous injective**
map from our completion: continuity would send $L$ to zero. This rules
out that particular graph-preserving topological identification. It says
nothing about arbitrary discontinuous maps or other topologies on series.

### The full formal local ring has a different unit structure

Root degree is additive on $M_r$, with only the unit $e$ in degree zero.
The finite coefficient recursion in the multiplication note shows that,
in the full formal ring, an array is invertible exactly when $a(e)\ne0$.
Thus $\mathbb R((M_r))$ is a local ring, with unique maximal ideal
$\{a:a(e)=0\}$.

Our completion has at least two distinct maximal ideals, $\ker I$ and
$\ker V$, where $I$ counts isolated vertices and $V$ counts all vertices.
Both characters map onto $\mathbb R$, and $I(K_2)=0$ whereas $V(K_2)=2$.
Hence the whole completion cannot even be algebraically isomorphic to
one of these full formal local rings.

Concretely, $K_1+K_2$ has constant local coefficient one. At radius one
its formal reciprocal has coefficient $(-2)^n$ on the rooted star with
$n$ leaves. Since that star has $n+1$ vertices, the reciprocal violates
every weighted $\ell^1$ condition. This is exactly where formal inversion
fails to give inversion in the analytic completion. This argument does
not exclude arbitrary subalgebras of generalized-series rings.

### The full topology admits no continuous norm

This is stronger than saying that no norm induces the topology. Every
continuous seminorm $q$ on our completion is bounded by a constant times
a maximum of finitely many defining seminorms. Choose $j$ so large that
the nonzero cycle difference

$$z_j=\frac{C_{2\cdot3^j}}{2\cdot3^j}-\frac{C_{3^j}}{3^j}$$

has zero histogram at all radii appearing in that finite maximum. Then
$q(z_j)=0$, so $q$ cannot be a norm.

In contrast, a fixed-coordinate space $\bigcap_k\ell^1(S,w_k)$ with
strictly positive weights always has a continuous norm: any one of its
defining weighted norms. Consequently the full completion is not
topologically linearly isomorphic to such a space, although each $L_r$
is exactly of that form. This does not rule out semiweights that vanish
on some coordinates or inverse limits with noninjective bonding maps.
It also excludes topological identification with the entire algebra
$A^\infty(\overline{\mathbb D})$, which has a continuous supremum norm.

### A nonopen unit group is not by itself a novelty criterion

For comparison, in the standard compact-open algebra
$\mathcal O(\mathbb C)$, the nonunits $1-z/n$ converge to the unit $1$.
Thus our nonopen unit group is compatible with familiar Fréchet-algebra
behavior. The graph-specific result is the explicit odd-cycle
construction, including nonunits whose degree generating function is
identically one. It does not by itself identify a new class of algebras.

## 4. Other nearby objects checked

| Candidate identification | What the comparison establishes |
| --- | --- |
| Finite signed measures on rooted graphs | This covers exactly those coherent arrays with uniformly bounded total variation across radii. Our cycle-series family contains elements outside it. General signed projective-extension obstructions already occur in [F6], Section 5.1; the countable discrete sufficient condition has a direct proof in our representation note. |
| Unimodular or rerooting-invariant laws | Unimodularity supplies the probability version of mass transport. Kaimanovich [F7], Theorem 50 and Remark 53, distinguishes it from ordinary rerooting invariance via a modular cocycle. A theorem about genuine measures does not identify our arbitrary signed projective arrays or their multiplication. |
| Graph cumulants | [F8], Sections 5.1–5.2.2 and equation (2), partitions edges into subgraphs and uses injective homomorphism densities. Our coproduct contracts complementary edge colors and uses rooted homomorphism counts. These formulas do not give the same coalgebra identification. No claim of abstract nonisomorphism follows from that comparison alone. |
| Graph Hopf or bialgebras | In [F9], disjoint union is a multiplication; it is addition in our algebra. Its splitting and contraction-extraction operations also differ from our Cartesian convolution. Our auxiliary edge-coloring coalgebra is a tool in the proof, not the completed algebra itself. |

## 5. Revised assessment

The earlier description can now be made more precise: there are exact
standard objects at the formal local level and inside the completion.
The generic ordered-series domain argument should be credited accordingly.
Membership in the broad classes of generalized series, weighted
convolution, and Fréchet algebras supplies useful existing theory.

The remaining identification question concerns the **particular** rooted
ball monoids, their truncations, the polynomial weights, and the balance
subalgebra simultaneously. A matching source would need either these
definitions or an explicit isomorphism carrying the topology and product
to them. Identifying the finite ring, a subalgebra, or the positive slice
alone does not settle that question.

No such source was located in this follow-up. That result establishes
neither originality nor absence of an abstract isomorphism with some
other known algebra. In particular, the priority of the signed
finite-graph approximation theorem, the local-ball ordering construction,
and the balance argument for summable reciprocals remains unresolved.
The [search log](SEARCH_LOG.md) records the source and coverage limits.

## Primary sources used in this follow-up

- **[F1]** W. Imrich, I. Klep, D. Smertnig, *Monoid algebras and graph
  products*, 2024 preprint / accepted manuscript.
  [Author PDF](https://math.smertnig.at/paper/graphproduct.pdf),
  [arXiv:2407.02615](https://arxiv.org/abs/2407.02615).
  Definition 3.4, Propositions 3.7–3.8, Section 3.1.
- **[F2]** R. Blute, R. Cockett, P.-A. Jacqmin, P. Scott,
  *Finiteness spaces and generalized power series* (2018).
  [arXiv:1805.09836](https://arxiv.org/abs/1805.09836).
  Section 2, Definition 2.2, Proposition 2.3, and Theorem 1.
- **[F3]** M. Abolghasemi, A. Rejali, H. R. E. Vishki,
  *Weighted Semigroup Algebras as Dual Banach Algebras* (2008).
  [arXiv:0808.1404](https://arxiv.org/abs/0808.1404). Section 2.
- **[F4]** S. J. Bhatt, S. R. Patel, *On Fréchet algebras of power series*,
  Bulletin of the Australian Mathematical Society 66 (2002), 135–148.
  [Institutional PDF](https://repository.ias.ac.in/59672/1/9_PUB.pdf).
  Examples 1.2 and 1.5, journal pages 137–138.
- **[F5]** T. V. Pedersen, *A class of weighted convolution Fréchet
  algebras* (2009 preprint).
  [arXiv:0909.2749](https://arxiv.org/abs/0909.2749). Introduction.
- **[F6]** S. Albeverio, S. Mazzucchi, *A unified approach to infinite
  dimensional integration* (2014 preprint; 2016 publication).
  [arXiv:1411.2853](https://arxiv.org/abs/1411.2853).
  Section 5.1, Theorems 4–5. The general sufficient theorem includes
  topological tightness hypotheses.
- **[F7]** V. A. Kaimanovich, *Invariance, quasi-invariance and
  unimodularity for random graphs* (2015 preprint).
  [arXiv:1512.08479](https://arxiv.org/abs/1512.08479).
  Theorem 50, Corollary 52, Remark 53.
- **[F8]** G. Bravo-Hermsdorff, L. M. Gunderson, P.-A. Maugis,
  C. E. Priebe, *Quantifying Network Similarity using Graph Cumulants*,
  Journal of Machine Learning Research 24 (2023), 1–27.
  [Publisher PDF](https://jmlr.org/papers/volume24/21-082/21-082.pdf).
  Sections 5.1–5.2.2 and equation (2).
- **[F9]** L. Foissy, *Hopf-algebraic structures on mixed graphs*
  (2023 preprint; revised 2024).
  [arXiv:2301.09449](https://arxiv.org/abs/2301.09449).
  Sections 1.2 and 2.2–2.3.

The relevant portions of these primary texts were accessible. Ribenboim's
1994 paper *Rings of generalized power series II: Units and zero-divisors*
was traced bibliographically, but its full text was not retrieved; no
specific theorem from that paper is asserted to have been checked.
