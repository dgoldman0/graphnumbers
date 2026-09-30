# An intrinsic representation of the local completion

30 September 2026. Follow-up to [the representation problem](REPRESENTATION_PROBLEM.md).
This note supplies a proof for the specific v0.1 construction. It does not
change its arithmetic or topology, and makes no claim of originality.

## Result

The completion consists exactly of **compatible signed neighborhood arrays
with every polynomial size moment finite, satisfying local mass-transport
balance**. The conditions refer only to the arrays and to explicit local
counting operations. They do not require a chosen approximating sequence.

The sufficiency proof gives a finite, generally very large, linear system
for each approximation. Signed coefficients are essential to that proof.
It also follows that some completed elements have no representation as a
finite signed measure on whole rooted graphs.

## 1. Arrays and the balance condition

Use the rooted induced ball types $\mathcal B_r$ from v0.1. Let
$\pi_{s,r}:\mathcal B_s\to\mathcal B_r$ denote truncation, for $s\ge r$.
Write $\mathscr W$ for the real families $a=(a_r)_{r\ge0}$ satisfying

$$
\|a_r\|_{1,k}:=\sum_{B\in\mathcal B_r}|V(B)|^k|a_r(B)|<\infty
\quad(r\ge0,\ k\ge1),
\qquad
(\pi_{s,r})_*a_s=a_r.
$$

All pushforward sums converge absolutely. Give $\mathscr W$ the topology
of these seminorms. If a rooted local function $f(G,o)$ is determined by
$B_R(G,o)$ and has polynomial growth in its size, put

$$\Lambda_a(f)=\sum_{B\in\mathcal B_R}f(B,o_B)a_R(B).$$

Compatibility makes this independent of increasing the observation radius.

A **local transport** $F(G,u,v)$ is invariant under isomorphism, is zero
when $\operatorname{dist}(u,v)>R$, and is determined, when nonzero, by the
induced $R$-ball around $u$, with $v$ distinguished. Here $R$ is
some fixed finite radius depending on $F$. Require, for some finite $C,k$,

$$|F(G,u,v)|\le C|B_R(G,u)|^k,\qquad k\ge0.$$

Graphs may be locally finite and infinite. In a disconnected graph, transport
between different components is zero. Define the local divergence

$$
\partial F(G,o)=
\sum_v F(G,o,v)-\sum_v F(G,v,o).
$$

This is determined by the $2R$-ball, and

$$|\partial F(G,o)|\le2C|B_{2R}(G,o)|^{k+1}.$$

Thus $\Lambda_a(\partial F)$ is well defined and continuous in $a$.
The **local balance condition** is

$$\boxed{\Lambda_a(\partial F)=0\quad\hbox{for every local transport }F.}$$

This is the local, signed version of the usual mass-transport identity.
The standard probability-measure definition is in
[Aldous–Lyons, Definition 2.1](https://arxiv.org/pdf/math/0603062).
The argument below treats arrays directly and uses no positive approximation
theorem.

There is a countable set of explicit tests. For each $R$, finite rooted
$R$-ball $H$, and second distinguished vertex $w\in H$, use the indicator
that $(B_R(G,u),u,v)$ has that doubly distinguished type. Their divergences
suffice. Indeed, every allowed $F$ is a countable linear combination of these
indicators. Both outgoing and incoming sums are absolutely summable against
$|a_{2R}|$, bounded by the displayed polynomial estimate. Consequently the
identities extend to $F$ by summation.

## 2. Representation theorem

**Theorem.** Under the original neighborhood map $T$,

$$
\mathcal A_{\mathrm{loc}}\cong
\{a\in\mathscr W:\Lambda_a(\partial F)=0
\text{ for every local transport }F\}.
$$

The identification is a linear homeomorphism and an algebra isomorphism,
where the array product is the original coordinatewise truncated Cartesian
convolution.

**Necessity.** For every finite graph $G$,

$$
\sum_{o\in V(G)}\partial F(G,o)
=\sum_{o,v}F(G,o,v)-\sum_{o,v}F(G,v,o)=0.
$$

The same holds for real linear combinations. Compatibility, summability,
and these continuous balance identities persist in the completion.
Sufficiency follows from the next three steps.

### Step A. Bounded-degree neighborhood indicators have finite expansions

Fix an integer $D\ge1$ and a radius $r$. Let

$$
b(D,r)=\sum_{j=0}^rD^j,\qquad M(D,r)=(D+1)b(D,r).
$$

These are convenient loose bounds, rather than optimized ones.
There are finitely many $r$-ball types of maximum degree at most $D$,
each of size at most $b(D,r)$.

For a finite connected graph $F$ with a distinguished vertex $u$, write
$\operatorname{inj}_u(F;G,o)$ for the number of injective edge-preserving maps
$F\to G$ sending $u$ to $o$. Nonedges need not be preserved.

**Indicator lemma.** On graphs of maximum degree at most $D$, each indicator
$\mathbf1_{\{B_r(G,o)\cong H\}}$ is a finite real linear combination of
$\operatorname{inj}_u(F;G,o)$, with $F$ connected, maximum degree at most
$D$, and at most $M(D,r)$ vertices.

**Proof with an explicit expansion.** Label the $h$ vertices of $H$, with
its root fixed. Let $I$ be the vertices at distance less than $r$ from the
root, let $N=D|I|$, and let $A_H$ be the number of root-preserving
automorphisms of $H$.

An induced embedding of $H$ taking its root to $o$ is the whole $r$-ball
exactly when no vertex in the image of $I$ has an edge to a vertex outside
the image of $H$. For any such embedding, the set of those external edge
incidences has size at most $N$. Inclusion-exclusion over its subsets tests
whether it is empty.

More explicitly, add $t$ labeled new vertices, $0\le t\le N$. Choose
$S\subseteq I\times[t]$ so every new vertex occurs in $S$ and
$|S|\le N$. For $t=0$, allow the empty $S$. Also choose any subset $Q$
of the nonedges inside $H$. Let $F(H,Q,S)$ have the edges of $H$,
the added edges $Q$, and the edges specified by $S$, retaining the original
root. Then

$$
\mathbf1_{\{B_r(G,o)\cong H\}}
=\frac1{A_H}
\sum_{t=0}^N\frac1{t!}
\sum_{\substack{S\subseteq I\times[t]\\
                 \text{every new vertex used}\\|S|\le N}}
\ \sum_{Q\subseteq\overline E(H)}
(-1)^{|S|+|Q|}
\operatorname{inj}_{o_H}(F(H,Q,S);G,o).
\tag{1}
$$

The $Q$-sum enforces the internal nonedges of $H$. The $S$-sum performs
the external-edge inclusion-exclusion. Each chosen set of $t$ distinct
external vertices is counted $t!$ times before division. Each resulting
whole-ball embedding is counted $A_H$ times.

All patterns are connected: every added vertex attaches to $I$.
They have at most $h+D|I|\le M(D,r)$ vertices. Patterns with a vertex of
degree greater than $D$ have zero injective count on this class and can
be discarded. This proves the lemma. The case $r=0$ gives the constant
indicator through the single-vertex pattern.

### Step B. Balanced bounded-degree arrays have exact finite realizations

Suppose $a\in\mathscr W$ is balanced and every $a_s$ is supported on balls
of maximum degree at most $D$.

For any finite connected $F$ and vertices $u,v\in V(F)$, use the transport
that counts injective embeddings of $F$ with $u$ at the sender and $v$
at the receiver. It is a local transport with polynomial growth. Balance gives

$$
\Lambda_a(\operatorname{inj}_u(F))
=\Lambda_a(\operatorname{inj}_v(F)).
\tag{2}
$$

**Finite realization lemma.** For every radius $r$ there is a finite real
linear combination $x$ of connected graphs of maximum degree at most $D$
and at most $M(D,r)$ vertices, such that $T_r(x)=a_r$.

**Proof.** Work in the finite-dimensional vector space on the possible
degree-$D$ $r$-ball types. Suppose a function $f$ annihilates the histogram
of every connected graph in the stated finite list. Expand $f$ using (1),
grouping isomorphic rooted patterns:

$$
f(G,o)=\sum_{F,u}\alpha_{F,u}\operatorname{inj}_u(F;G,o).
$$

On summing over the root, the root choice in an injective count disappears:

$$
0=\sum_o f(G,o)
=\sum_F\beta_F\operatorname{inj}(F,G),
\qquad \beta_F=\sum_u\alpha_{F,u}.
$$

The functions $\operatorname{inj}(F,\cdot)$, for connected $F$ of maximum
degree at most $D$ and size at most $M(D,r)$, are linearly independent on
that same finite graph list. Order both patterns and test graphs by vertex
count and then edge count. The matrix of injective counts is triangular:
a nonzero entry requires $|V(F)|\le|V(G)|$, and when the vertex counts
agree it requires $|E(F)|\le|E(G)|$.
Within equal vertex and edge counts it is zero between nonisomorphic
graphs, and its diagonal is $|\operatorname{Aut}(F)|>0$.
Consequently every $\beta_F=0$.

Equation (2) now gives $\Lambda_a(f)=0$. Thus $a_r$ annihilates every
linear functional annihilating the finite list of histogram vectors.
Finite-dimensional linear algebra puts $a_r$ in their span, as required.
This also provides an explicit reconstruction procedure: enumerate the list,
form its integer histogram matrix, and solve for real coefficients.

The lemma imposes no sign or size bound on those coefficients.

### Step C. Remove the degree bound without changing the topology

For any graph $G$, let $Q_DG$ keep every vertex and delete every edge
incident to a vertex whose degree in the original $G$ exceeds $D$.
Those vertices become isolated. The resulting graph has maximum degree
at most $D$.

Its rooted $r$-ball is determined by the original rooted $(r+1)$-ball:
the original degrees of vertices at distance at most $r$ are known there.
Let $\theta_{D,r}:\mathcal B_{r+1}\to\mathcal B_r$ be this local map and set

$$a_r^{(D)}=(\theta_{D,r})_*a_{r+1}.$$

These arrays are compatible, satisfy every weighted summability condition,
and are supported on degree-$D$ balls. They remain balanced. To see this,
pull a transport on $Q_DG$ back to $G$, assigning zero to pairs in different
components of $Q_DG$. Its range remains finite, its required observation
radius increases by at most one, and its growth remains polynomial.
Balance for $a$ is exactly balance for the transformed arrays.

If $|B|\le D$, truncation and $\theta_{D,r}$ agree on $B$.
Both maps decrease size, so

$$
\|a_r^{(D)}-a_r\|_{1,k}
\le
2\sum_{\substack{B\in\mathcal B_{r+1}\\|V(B)|>D}}
|V(B)|^k|a_{r+1}(B)|
\longrightarrow0.
\tag{3}
$$

For $n\ge1$, apply Step B with $D=r=n$ to $a^{(n)}$, obtaining a finite
graph combination $x_n$ with

$$T_n(x_n)=a_n^{(n)}.$$

For every fixed $r\le n$, compatibility gives $T_r(x_n)=a_r^{(n)}$.
Equation (3) therefore proves

$$\|T_r(x_n)-a_r\|_{1,k}\longrightarrow0
\quad\hbox{for every fixed }r,k.$$

Thus $a\in\mathcal A_{\mathrm{loc}}$, proving sufficiency.
The identification preserves exactly the defining seminorms. The product
assertion follows from the original continuous convolution formula and
density. This completes the representation theorem.

## 3. What this characterization includes and excludes

**Compatibility alone is insufficient.** Put unit mass at a three-vertex path
rooted at an endpoint, and take its compatible ball marginals. Every weighted
sum is finite. Send one unit along each edge from a degree-one vertex to a
degree-two vertex. At the specified root the outgoing count is one and the
incoming count is zero. This violates balance, so the array is excluded.
The uniform-root law of the same path satisfies balance.

**The positive cone has a familiar description.** If every $a_r\ge0$, their
common total mass $m$ is finite. Consistency extends them uniquely to a
positive measure on connected locally finite rooted graphs. For $m=1$,
local balance is equivalent to ordinary unimodularity. To check this last
statement, compare the outgoing and incoming measures on doubly rooted
graphs, first restricted to pairs at distance at most $s$. Both have finite
mass $\Lambda_a(|B_s|)$. Local cylinder tests determine these finite
measures; then increase $s$.

Consequently the positive mass-one elements are exactly the unimodular
probability laws with finite $\int |B_r|^k$ for every $r,k$.
Their approximants in the proof are signed graph combinations.
There is no assertion that those approximants can be chosen positive or
as individual normalized finite graphs. The positive finite-approximation
issue discussed in the [literature review](LITERATURE_REVIEW.md) is separate.

**The computation is finite but potentially enormous.** At the $n$-th step,
the displayed bound permits graphs on up to

$$M(n,n)=(n+1)(1+n+\cdots+n^n)$$

vertices. The proof establishes a characterization and an existence
procedure, not a practical general-purpose implementation. Efficient
reconstruction and coefficient bounds remain open tasks.

## 4. Exactly when the arrays define a finite signed measure

For any coherent absolutely summable family, set

$$M(a)=\sup_r\sum_{B\in\mathcal B_r}|a_r(B)|.$$

The unweighted norms used here are well defined and bounded at each fixed
radius by the original weighted norms.

**Measure criterion.** There is a finite signed Borel measure $\mu$ on
connected locally finite rooted graphs with these marginals if and only if
$M(a)<\infty$. It is unique and $\|\mu\|_{\mathrm{TV}}=M(a)$.
This criterion applies to coherent arrays whether or not they satisfy balance.

**Proof.** Necessity follows because pushforward contracts total variation.
For sufficiency define, for $s\ge r$,

$$b_r^{(s)}=(\pi_{s,r})_*|a_s|.$$

As $s$ increases these nonnegative arrays increase coordinatewise, since
$|(\pi_{s+1,s})_*a_{s+1}|\le(\pi_{s+1,s})_*|a_{s+1}|$.
Their pointwise limits $b_r$ have common total mass $M(a)$, dominate
$|a_r|$, and are compatible by monotone convergence. Hence
$(b_r+a_r)/2$ and $(b_r-a_r)/2$ are compatible nonnegative finite arrays.
The ordinary countable positive extension theorem gives two finite measures
on the inverse-limit path space. A compatible path of finite ball types
determines a connected locally finite rooted graph: choose successive
representatives with matching truncations and take their union. The ball
coordinates generate the usual Borel structure.

Subtract the two positive measures. Its variation is at most $M(a)$, and
the reverse inequality follows by projecting to each radius. Agreement on
cylinders gives uniqueness.

For signed arrays, weighted summability at each radius alone need not bound
the corresponding moments of the total variation of the global measure.
No stronger global moment assertion is used here.

The uniform-variation obstruction is standard in projective measure theory;
see [Albeverio–Mazzucchi, Section 5.1, Theorems 4–5](https://arxiv.org/pdf/1411.2853).
Their general topological statement also has a tightness requirement. The
direct proof above supplies sufficiency for this countable discrete system.

## 5. A closed family of elements beyond finite signed measures

Let $N_j=3^j$, $j\ge1$, and define normalized cycle differences

$$z_j=\frac{C_{2N_j}}{2N_j}-\frac{C_{N_j}}{N_j}.$$

For any real sequence $c=(c_j)_{j\ge1}$, the series

$$J(c)=\sum_{j\ge1}c_jz_j$$

converges in $\mathcal A_{\mathrm{loc}}$. Indeed, for fixed $r$, all terms
with $N_j>2r+1$ have exactly zero radius-$r$ histogram: both normalized
cycles give the same rooted path ball. Every local coordinate of the series
is therefore eventually constant, regardless of coefficient growth.

At $r=N_m$, the first $m$ pairs of cycles are fully visible, all their
cycle types are distinct, and all later pairs cancel locally. Hence

$$
\|T_{N_m}(J(c))\|_1=2\sum_{j=1}^m|c_j|,
\qquad
\|T_{N_m}(J(c))\|_{1,k}
=\sum_{j=1}^m|c_j|\big((2N_j)^k+N_j^k\big).
$$

It follows from the measure criterion that

$$J(c)\text{ is represented by a finite signed measure}
\quad\Longleftrightarrow\quad c\in\ell^1.$$

For example $c_j=1$ gives a completed element with finite data at every
radius and unbounded total variation as the radius increases.

There is also a topological conclusion. $J$ is a continuous linear embedding
of $\mathbb R^{\mathbb N}$ with its product topology. The coefficient $c_j$
is recovered by the coordinate of the rooted $C_{2N_j}$ type at radius
$N_j$. These coordinate functionals define a continuous map
$P:\mathcal A_{\mathrm{loc}}\to\mathbb R^{\mathbb N}$, with $PJ$ the
identity. Thus $JP$ is a continuous linear projection, and the image of
$J$ is a closed complemented subspace.

This describes a concrete part of the completed space that requires
arbitrarily large signed cancellation. It also explains why a single global
finite-measure model would miss some of its elements.

## 6. Scope, verification, and remaining work

The argument answers the stated membership question. The subsequent
[multiplication and units note](MULTIPLICATION_AND_UNITS.md) proves that the
completion is a domain and supplies an exact recursive unit criterion,
using this representation theorem to establish balance of the reciprocal.
The multiplicative character space, more tractable unit tests, efficient
approximation, and useful analytic applications remain subjects for further work.
The local transport principle and the combinatorial/functional-analytic
methods have substantial prior literature. Whether the exact representation
theorem or its signed formulation has appeared elsewhere requires a focused
bibliographic comparison; this note establishes no priority claim.

The [companion verifier](verify_representation.py) checks the delicate finite indicator expansion by
independent embedding enumeration, the degree-cutoff locality and error
bound, transport balance and the excluded rooted-path example, and explicit
cycle cancellation formulas. All
[13,758 exact checks passed](representation_results.json). Its finite checks support the calculations;
the all-radius and all-graph statements follow from the proofs above.
