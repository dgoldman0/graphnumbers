# Mathematical comparisons with the prior literature

30 September 2026. These calculations supplement the
[literature review](LITERATURE_REVIEW.md). They use the definitions and results
of candidate v0.1. They establish specific identifications or distinctions;
none is asserted to be an original theorem.

## 1. Counting measures and convolution

Let G_* be the space of isomorphism classes of connected locally finite rooted
graphs. For a finite graph G, put

$$\mu_G=\sum_{v\in V(G)}\delta_{[G(v),v]},$$

where G(v) denotes the connected component containing v. Extend this definition
linearly to x in A0. This map is injective: counting measures arising from
different connected unrooted graph types have disjoint supports, and each has
positive total mass.

The rooted Cartesian product induces convolution, initially on finite atomic
signed measures:

$$\delta_{[G,g]}*\delta_{[H,h]}
=\delta_{[G\square H,(g,h)]}.$$

Counting pairs of roots gives

$$\mu_{G\sqcup H}=\mu_G+\mu_H,
\qquad\mu_{G\square H}=\mu_G*\mu_H.$$

If pi_r sends a rooted graph to its induced radius-r ball, then

$$T_r(x)=(\pi_r)_*\mu_x.$$

The product-ball identity in the v0.1 note says pi_r respects multiplication
when the target uses the truncated product star_r. Accordingly,

$$p_{r,k}(x)=\| (\pi_r)_*\mu_x\|_{\ell^1(w_k)},
\qquad w_k(B)=|V(B)|^k.$$

This is an exact measure-convolution description of the finite algebra and its
seminorms. Its completion is still defined as the closure of the graph-derived
diagonal image. The calculation does not assert that every compatible signed
family belongs to this closure or defines a finite signed measure on G_*.

The later [representation theorem](REPRESENTATION_THEOREM.md) supplies those
additional results: membership is equivalent to local mass-transport balance
within the compatible weighted arrays, and finite signed-measure
representability is equivalent to uniformly bounded local total variation.

For normalized positive graphs, mu_G/|V(G)| is precisely the empirical
neighborhood distribution U(G) of
[Bordenave–Caputo, Section 1.1](https://arxiv.org/abs/1308.5725v2).

## 2. Bounded-degree normalized convergence

Suppose G_n have maximum degree at most D and let x_n=G_n/|V(G_n)|. For each
fixed r there are only finitely many possible rooted r-ball types: their size
is bounded by 1+D+...+D^r. All weighted l1 norms on this finite coordinate space
induce the same topology.

Consequently x_n converges in every p_(r,k) to the local marginals of a rooted
probability law mu if and only if each rooted-ball frequency converges to the
corresponding mu-frequency. This is exactly local weak / Benjamini–Schramm
convergence. The diagonal limit is in our completion because x_n are its
approximating elements.

For n>2r+1, every rooted r-ball in C_n is the centered path on 2r+1 vertices.
Thus C_n/n converges to the deterministic rooted infinite line. For fixed d,
Cartesian products of these cycles converge to the rooted integer lattice
Z^d. These identify our examples with established local limits rather than
introducing additional geometric objects.

Definition source:
[Benjamini–Schramm, Section 1.2](https://arxiv.org/html/math/0011019v4).

The later [positive-cone checkpoint](STRICT_POSITIVE_CONES.md), Lemma 1,
extends this comparison to the full closed finite positive cone. Rational
disjoint unions replace finite positive mixtures by individual normalized
graphs. If the limiting law has degree at most D, isolating vertices of
original degree greater than D preserves local convergence and produces
degree-D approximants. Thus the bounded-degree mass-one slice of P_fin
is exactly the sofic laws, even when its original approximants have
arbitrary finite degrees.

## 3. Positive normalized convergence

Let mu_n=U(G_n) for nonempty finite graphs, and let mu be a probability law on
connected locally finite rooted graphs. Write D for the root degree and b_r
for the number of vertices in the induced rooted r-ball. The following are
equivalent:

1. For every r>=0 and k>=1, the radius-r marginals converge in weighted total
   variation with weight b_r^k.
2. mu_n converges locally weakly to mu and, for every integer j>=1,
   the j-th root-degree moments converge to finite limiting moments:

$$\int D^j\,d\mu_n\longrightarrow\int D^j\,d\mu<\infty.$$

Here weighted total variation is the full weighted l1 sum used in our
seminorms, without the optional factor 1/2 in probabilistic conventions.

**Proof.** The forward implication follows from local marginal convergence and
D^j <= b_1^j. For the reverse implication we need uniform integrability of
each b_r^k.

Let W_j(G,o) count length-j walks starting at o, with W_0=1. Every vertex in
the r-ball is reached by at least one such walk of length at most r, so for
integer s>=1,

$$b_r^s\leq (r+1)^{s-1}\sum_{j=0}^r W_j^s.$$

For j>=1, W_j^s counts rooted homomorphisms of the tree formed by s paths of
length j sharing their initial vertex. It has js+1 vertices. The standard
connected-pattern bound, attributed to Sidorenko and stated as Theorem 2.2 in
[Kurauskas](https://arxiv.org/html/1504.08103v3), gives

$$\int W_j^s\,dU(G)\leq\int D^{js}\,dU(G).$$

Consequently, for fixed r>=1 and s,

$$\sup_n\int b_r^s\,d\mu_n
\leq (r+1)^{s-1}\left(1+r\sup_n\int D^{rs}\,d\mu_n\right)<\infty.$$

Take s=2k. Local weak convergence and truncation give the same finite bound
for the limiting b_r moment. The weighted tails are uniformly bounded by

$$\int b_r^k\mathbf1_{\{b_r>M\}}\,d\mu_n
\leq M^{-k}\int b_r^{2k}\,d\mu_n,$$

and similarly for mu. There are finitely many r-ball types of size at most M;
their coordinates converge. First letting n grow on this finite set and then
letting M grow proves weighted l1 convergence. Radius zero is immediate.

This equivalence is a comparison proved here using an existing bound. It does
not assert that Kurauskas defines our algebra. It also does not extend to
arbitrary signed combinations: positivity was used in the walk and tail
bounds. In particular, it does not classify all elements of A_loc.

## 4. The edge observable is a point derivation

The degree polynomial, already studied in the literature, is

$$D_G(z)=\sum_{v\in V(G)}z^{\deg_G(v)}.$$

Its definition and earlier references are given in
[Brown–George, introduction](https://arxiv.org/html/2505.04882v1).
For Cartesian products the degree of (v,w) is deg_G(v)+deg_H(w). Directly,

$$D_{G\sqcup H}=D_G+D_H,\qquad
D_{G\square H}=D_GD_H.$$

Thus G -> D_G extends to an algebra homomorphism A0 -> R[z]. Moreover,

$$D_G(1)=V(G),\qquad D'_G(1)=2E(G).$$

Evaluating at z=1+epsilon/2 in the dual numbers, epsilon^2=0, gives

$$D_G(1+\varepsilon/2)=V(G)+\varepsilon E(G).$$

Since V and E are continuous, this last homomorphism extends to A_loc. Expanding
powers in the dual numbers proves, for every polynomial f,

$$f(V(X)+\varepsilon E(X))
=f(V(X))+\varepsilon f'(V(X))E(X).$$

Entire real power series follow by the established seminorm estimates and
continuity. This proves the compatibility formula in ANALYSIS.md as an
ordinary point-derivation identity.

Only V and E were extended in this argument. It does not assert that the full
degree generating function can be evaluated continuously at every real z on
every completed element. Polynomial moment control alone does not supply
arbitrary exponential degree moments.

The later [multiplication note, Section 5](MULTIPLICATION_AND_UNITS.md#5-a-retract-onto-a-familiar-analytic-function-algebra)
does extend the full degree generating function continuously to the closed
complex unit disk, including all derivatives there. It identifies its
range with the real-coefficient smooth holomorphic disk algebra and gives
a continuous algebra section using normalized hypercubes. This conclusion
still makes no claim about evaluation outside the closed disk.

## 5. Distinguishing the topology from Banach and universal completions

For a finite combination x=sum_C c_C C, the connected-component vertex norm is

$$N(x)=\sum_C |c_C|\,|V(C)|.$$

For different m,n>=3,

$$N(C_n/n-C_m/m)=2.$$

Our normalized cycles converge, so these two topologies disagree even on
sequences of finite combinations. N is the component norm with vertex-count
weight in [Knill's 2021 discussion](https://arxiv.org/abs/2106.10093).

A stronger obstruction is nonnormability. For fixed R,K and n>2R+1, let

$$z_n=C_{2n}-2C_n.$$

At every radius r<=R the local rooted types agree and the total root counts
cancel. Hence p_(R,K)(z_n)=0, although z_n is a nonzero element of A0.
Any continuous norm q would be bounded by a constant times a finite maximum
of defining seminorms, and therefore by some C p_(R,K), using monotonicity.
It would vanish on z_n, a contradiction. There is no continuous norm inducing
this topology and no linear homeomorphism of A_loc with a Banach space.

This also distinguishes the topology from the **universal** Arens–Michael
envelope of the bare algebra. Let c count connected components, extended
linearly. Products of connected graphs are connected, so

$$c(xy)=c(x)c(y).$$

Thus |c| is a submultiplicative seminorm. But c(z_n)=-1 while z_n -> 0 in
our topology. The universal envelope uses all submultiplicative seminorms and
must include |c|; our selected topology does not make it continuous. This is
consistent with calling A_loc an Arens–Michael algebra (over the reals), and
inconsistent with calling it the universal envelope of A0. See
[Pirkovskii, introduction and Section 1](https://arxiv.org/abs/math/0406352)
for the terminology over the complex field.

## 6. Distinguishing two other graph-limit frameworks

**Asymptotic spectrum distance.** In
[de Boer–Buys–Zuiddam, Sections 2.1–2.4](https://arxiv.org/abs/2404.16763v2),
distance is the supremum of differences over normalized monotone
semiring-homomorphic invariants, using the strong graph product. K2 and K1
admit cohomomorphisms in both directions: there are no distinct nonadjacent
pairs to obstruct the map from K2 to K1. Every such invariant therefore has
value 1 on both graphs, and their distance is zero. In our algebra
V(K2-K1)=1. The canonical finite-graph identifications differ.

**Dense graphons.** A graphon associated with C_n has integral 2/n and hence
cut norm at most 2/n. Both C_n and the n-vertex edgeless graph have zero as
their dense-graphon limit, as the definition in
[Lovász–Szegedy](https://arxiv.org/abs/math/0408173) makes explicit.
Our normalized versions have different limits: C_n/n -> L with E(L)=1,
whereas the normalized edgeless graph equals K1 with E(K1)=0. For n>=4,

$$p_{1,1}(C_n/n-K_1)=3+1=4.$$

Conversely, dense cliques have a graphon limit while

$$p_{1,1}(K_n/n)=n,$$

so they are not Cauchy in the candidate topology. These comparisons concern
the natural graph representations; they are not claims about arbitrary
abstract bijections between unrelated limit spaces.
