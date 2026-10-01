# Strict inclusion of the finite positive cone

1 October 2026. Step 6 of the [development path](../../reviews/local-completion-2026-09-30/DEVELOPMENT_PATH.md).
This note concerns the real **local Cartesian graph completion** A.
It supplies the simple-graph encoding and approximation arguments required
to apply the cited non-soficity theorem. The external theorem is an input;
its long complexity-theoretic proof is not independently audited here.
Independent review of the new arguments remains deferred.

Write

\[
 P_{\rm loc}=\{x\in A:T_rx\ge0\text{ for every }r\},\qquad
 P_{\rm fin}=\overline{\operatorname{cone}\{G:G\text{ finite simple}\}}.
                                                                    \tag{1}
\]

The closure uses all the completion's weighted local seminorms.

**Main theorem.** Using the non-co-sofic invariant random subgroup theorem
of Bowen–Chapman–Lubotzky–Vidick and Bowen–Chapman–Vidick [1–2],

\[
 P_{\rm fin}\subsetneq P_{\rm loc}.                          \tag{2}
\]

A witness can be chosen with vertex mass one and a finite maximum degree.
Its rooted law is unimodular and has all polynomial ball-size moments.
It admits finite signed graph approximations, as supplied by the existing
representation theorem, while every such approximation retains a positive
amount of negative coefficient mass. A bounded finite-radius observable
separates this element from every positive finite graph combination.

The encoding below is explicit for every finite number d of generators.
The source law is supplied existentially by [1–2]; this note does not
produce its neighborhood probability table, a numerical degree for that
particular law, or numerical separating coefficients. This is an application
of the negative Aldous–Lyons result to the two cones of this completion.

## 1. The external result and its precise use

The input is the existence, for some finite d, of a unimodular probability
law mu on connected directed graphs carrying labels 1,...,d, with exactly
one incoming and one outgoing edge of each label at every vertex, which
is not a local limit of uniform-root finite permutation actions. Loops,
oppositely directed edges, and edges with different labels between the
same vertices are allowed. These are Schreier graphs of the free group
F_d. Each generator acts as a permutation sigma_i of the vertex set.

Part I [1], Corollary 7.5, obtains the non-co-sofic IRS from Theorems 1.10,
7.3 and 7.4. Its Theorem 7.4 is supplied by Part II [2], Theorem 1.1
(formal version Theorem 2.31). Part I's introduction explains the
IRS/Schreier-law correspondence and its footnote 2 notes the standard
decoration route to unlabelled graphs. Here that route is written out with
constant vertex expansion, root averaging, and an explicit finite repair.
No existence of a non-sofic group is used or asserted.

## 2. Positive finite combinations and bounded-degree sofic laws

For a finite nonempty graph put U(G)=G/|V(G)|. Let A_D denote the elements
supported at every radius on graphs of degree at most the integer D.
The representation theorem identifies the mass-one part of
P_loc intersect A_D with the unimodular rooted probability laws of that
degree cap. On this class the topology is local weak convergence:
there are finitely many radius-r ball types and their sizes are uniformly
bounded. See [COMPARISON_LEMMAS.md](COMPARISON_LEMMAS.md), Section 2.

**Lemma 1.** For x in P_loc intersect A_D with V(x)=1, these are equivalent:

1. x belongs to P_fin, allowing approximants with arbitrary finite degrees.
2. Its law is a local weak limit of uniform-root finite simple graphs.
3. There are finite simple graphs G_n with maximum degree at most D and
   U(G_n) converging to x in every defining seminorm.

**Positive mixtures become single finite graphs.** Suppose
y_n=sum_G c_{n,G}G, c_{n,G}>=0, converge to x. Their masses tend to one,
so normalize them. Each normalized combination is a finite convex mixture
sum_G alpha_G U(G), where alpha_G=c_G|V(G)|/V(y_n).
Approximate the alpha_G by rational probabilities beta_G. Choose an
integer L with L beta_G/|V(G)| integral for every G, and take that many
disjoint copies of G. The resulting graph has L vertices and normalized
law sum_G beta_G U(G).

For any fixed finite family of seminorms, the difference can be made
arbitrarily small by choosing the rational probabilities sufficiently
close: the family of graphs in that mixture is finite. At stage n use
error at most 1/n in p_{n,n}, which dominates p_{r,k} for r,k<=n.
A diagonal choice therefore gives a single finite-graph sequence converging
to x in A. In particular (1) implies (2). Disconnected graphs are allowed,
as they are in the original finite graph algebra.

**Excess degrees can be removed.** For any graph H let Q_D H keep all
vertices and delete all edges incident to vertices whose original degree
exceeds D. The result has degree at most D. Let Bad_D(H) be that set of
vertices. A rooted r-ball can change only if its root lies within distance
r of Bad_D(H) in H. Whether this occurs is determined by the original
(r+1)-ball, since the degrees of its vertices at distance at most r are
visible there.

If U(H_n) converges locally to a law supported on degree at most D, the
probability of this event tends to zero. Indeed the limiting distribution
at radius r+1 is supported on finitely many degree-D ball types, none
of which has the event. Thus, with full l1 distance,

\[
 \|T_rU(Q_DH_n)-T_rU(H_n)\|_1
 \le2\Pr_{U(H_n)}\{\operatorname{dist}(o,\operatorname{Bad}_D(H_n))\le r\}
 \longrightarrow0.                                        \tag{3}
\]

All vertices are retained, so normalization is unchanged. The cut graphs
have a common degree cap; their local weak convergence is convergence in
every weighted local seminorm. This proves (2) implies (3), and (3) implies
(1) follows directly. QED.

Consequently the bounded-degree mass-one slice of P_fin is exactly the
sofic laws, even though the definition of P_fin permits real positive
coefficients and arbitrarily large degrees in the approximants. Neither
permission enlarges that slice.

## 3. A constant-size encoding into simple graphs

Fix d>=1. For each vertex v of a Schreier graph G create a center c_v,
two leaves attached to c_v, and two ports t_{v,i}, h_{v,i} for each label
i=1,...,d. Join both ports to c_v. Attach 2i+1 new leaves to t_{v,i} and
2i+2 new leaves to h_{v,i}. Finally, for every v and i, add the edge

\[
 t_{v,i}\;--\;h_{\sigma_i(v),i}.                             \tag{4}
\]

Call the resulting unlabelled undirected graph E_d(G). The ports and all
leaves created at v, together with c_v, form its fiber F_v. The constants
are

\[
 b_d=|F_v|=3+\sum_{i=1}^d(4i+5)=2d^2+7d+3,\qquad
 \Delta_d=2d+4.                                             \tag{5}
\]

Every center has degree 2d+2; tail and head ports have degrees 2i+3
and 2i+4. Leaves have degree one. Thus the degree cap is Delta_d, and
|V(E_d(G))|=b_d|V(G)| when G is finite. Each fiber is a tree with b_d-1
internal edges; the d inter-port edges per original vertex give
|E(E_d(G))|=(b_d-1+d)|V(G)|.

All edges are simple. A generator loop makes a triangle c_v,t_{v,i},h_{v,i}.
Different labels use different ports, and inverse or repeated connections
use different tail/head pairs. Connectivity of G implies connectivity of
E_d(G).

### Recognition without stored labels

In an encoded graph, a center is recognized by its exactly two leaf
neighbors. A port with ell leaf neighbors has ell in {3,...,2d+2}; this
number determines its label and its tail/head role. Every port has
exactly two nonleaf neighbors: its center and the opposite port in (4).
Following a center–tail–head–center path recovers sigma_i. All fibers
and generator maps are therefore recoverable from the unlabelled graph.
Permutations of a group of pendant leaves do not affect the decoder.

Every fiber vertex is within distance two of its center, and an original
generator step is a path of length three. These uniform bounds imply
continuity of encoding for local laws. A radius-R observation at any
fiber root can, for example, be obtained from the source (R+2)-ball.

## 4. Root averaging and unimodularity

The correct probability law on the enlarged graph is

\[
 \int f(H,z)\,d\widehat\mu(H,z)
 =\frac1{b_d}\int\sum_{z\in F_o}f(E_d(G),z)\,d\mu(G,o).     \tag{6}
\]

Every fiber has the same size, so this is a probability law without a
degree-dependent reweighting. For a finite uniform-root source it is
exactly U(E_d(G)). Centers have probability 1/b_d. Conditional on the
root being a center, decoding recovers mu.

**Lemma 2.** If mu is unimodular, then hat-mu is unimodular.

**Proof.** Given a nonnegative measurable transport f on the encoded
doubly rooted graphs, define the source transport

\[
 F(G,u,v)=\sum_{a\in F_u}\sum_{b\in F_v}f(E_d(G),a,b).
\]

The expected outgoing mass under (6) is
b_d^{-1} E_mu sum_v F(G,o,v). Source unimodularity changes this to
b_d^{-1} E_mu sum_u F(G,u,o), the expected incoming mass. Nonnegativity
justifies the sums even when they are infinite. The construction is
equivariant under source isomorphisms, so F is a legitimate transport.
QED.

Center-only rooting would generally fail this statement: the transport
from each noncenter to its fiber center has expected outgoing mass zero
and incoming mass b_d-1 under that rooting. Under (6) both are
(b_d-1)/b_d. Root averaging is part of the construction.

In particular hat-mu has all polynomial ball-size moments, since its
degree is at most Delta_d. It therefore represents an element of P_loc
by the signed representation theorem.

## 5. Decoding imperfect finite approximations

Exact invertibility of the encoding alone does not settle soficity:
finite approximating graphs need not themselves be encoded graphs.
The following decoder works on arbitrary finite simple graphs H.

Call a vertex a port of type ell if it has degree ell+2 and exactly ell
neighbors of degree one. Let C(H) be the vertices with exactly two leaf
neighbors and, among their remaining neighbors, exactly one port of
each type 3,...,2d+2, with no other neighbors. Thus their degree is 2d+2.
Port type is a radius-two property and membership in C(H) is a
radius-three property. In a perfect encoding C(H) is exactly its center
set.

For u in C(H) and label i, take its unique port t of type 2i+1. Its
nonleaf neighbor other than u is unique. If that neighbor is a port h
of type 2i+2, and h's other nonleaf neighbor w belongs to C(H), define
tau_i(u)=w. Otherwise leave tau_i(u) undefined.

**Lemma 3.** Each tau_i is a partial injection of C(H) into itself.

**Proof.** There is at most one outgoing path by the unique tail at u
and the two-nonleaf-neighbor condition. At a possible target w, the
unique head of type 2i+2 has only one nonleaf neighbor other than w;
this fixes t, whose other nonleaf neighbor fixes u. Therefore at most
one source maps to w. This includes u=w. QED.

Complete tau_i to a permutation sigma_i^H of C(H) by any bijection between
its unused domain and unused range. They have equal sizes. The resulting
d permutations define a finite F_d action. No group relations beyond the
free generators are imposed.

Let B(H) consist of centers missing an incoming or outgoing partial edge
of some label. This is a radius-six property in H: each candidate path
has length three, and the other endpoint's center test has radius three.
All new permutation edges have both endpoints in B(H). In the repaired
Schreier graph, whose number of neighbors is at most 2d, at most

\[
 |B(H)|\,b(2d,r),\qquad b(a,r)=\sum_{j=0}^r a^j,             \tag{7}
\]

roots can lie within distance r of those endpoints. Every other rooted
r-ball agrees with the partial graph's r-ball. The partial rooted r-ball
at a center is determined by its radius-(3r+6) ball in H: paths between
centers use three edges, with six additional steps sufficient to check
all endpoint memberships and incident partial maps.

**Theorem 4 (soficity is preserved and reflected).** For a unimodular
Schreier law mu, hat-mu is a sofic unlabelled simple-graph law if and only
if mu is a limit of uniform-root finite permutation actions.

**Forward encoding.** Encode each finite action. Constant fiber size
identifies the normalized finite law with (6), and the local dependence
from Section 3 passes convergence to hat-mu.

**Reverse decoding.** Suppose U(H_n) converges locally to hat-mu. Lemma 1's
degree-cutoff argument lets us replace H_n by Q_{Delta_d}H_n. Local
center recognition gives

\[
 \frac{|C(H_n)|}{|V(H_n)|}\longrightarrow\frac1{b_d}>0,
 \qquad \frac{|B(H_n)|}{|C(H_n)|}\longrightarrow0.            \tag{8}
\]

The second limit follows because every center in a perfect encoding
has every incoming and outgoing edge, and both predicates are local.
The center sets are nonempty eventually. For each fixed r, conditional
local convergence at a center and the radius-(3r+6) dependence show that
the uniform-root partial Schreier laws converge to mu. Completion to
permutations changes their radius-r distributions by at most

\[
 2\,b(2d,r)\frac{|B(H_n)|}{|C(H_n)|},                        \tag{9}
\]

in full l1 distance, by (7). Thus the completed finite actions converge
to mu, proving the reverse implication. QED.

The arbitrary choices used to pair missing ports in a finite graph need
not be canonical: (9) is uniform in those choices. The decoding handles
loops, opposite arcs and repeated endpoints with different labels.

## 6. Strict cone inclusion and a finite-radius separator

Take the non-sofic law mu from Section 1. Lemma 2 produces a bounded-degree
unimodular simple-graph law hat-mu. Its element x has V(x)=1 and belongs
to P_loc. If x belonged to P_fin, Lemma 1 would give finite simple-graph
approximants, and Theorem 4 would give finite permutation approximants
to mu. This contradicts the external theorem. Since every finite positive
combination has nonnegative marginals and P_loc is closed, (2) follows.

The argument also clarifies the role of signs in the completion. Fix
Delta=Delta_d and let K_Delta be the set of sofic laws on rooted simple
graphs of degree at most Delta. It is a compact convex set: it is the
closed set of local limits of finite uniform-root laws in the compact
bounded-degree law space, and disjoint-union rational approximation
establishes convexity. Its closure is unchanged if all finite approximants
are required to have cap Delta, by Lemma 1.

Since hat-mu is outside this set, there are a radius r, a real function
f on the finitely many degree-Delta r-ball types, and a threshold a such
that

\[
 \mathbb E_{\hat\mu}f(B_r)>a>
 \sup_{\nu\in K_\Delta}\mathbb E_\nu f(B_r).                 \tag{10}
\]

One direct proof uses compactness: if the projection of hat-mu belonged
to every projected K_Delta, the nested nonempty compact sets matching
its first r-ball distribution would have an element matching every
radius, hence hat-mu itself. Some finite projection therefore excludes
hat-mu, and finite-dimensional separation gives (10). Rescale f so
||f||_infinity<=1 and choose a strictly between the two values. Then
|a|<1. Rational f and a can be chosen by a sufficiently small perturbation
of the finite list of coefficients.

For arbitrary-degree graphs define the bounded local observable

\[
 \psi(G,o)=f(B_r(Q_\Delta G,o)),\qquad
 \ell(y)=aV(y)-\Lambda_y(\psi).                              \tag{11}
\]

The rooted (r+1)-ball of G determines psi; |psi|<=1. Thus ell is a
continuous finite-radius linear functional on the full signed completion.
Every finite graph satisfies ell(G)>=0 because Q_Delta G is a finite
degree-Delta graph on the same vertex set. But Q_Delta fixes hat-mu,
so ell(x)<0. This gives a bounded local separating inequality valid
against **all** finite simple graphs, including those with excess degrees.
The proof supplies existence of this inequality, not its numerical
coefficients for the existential source law.

**Necessary negative mass.** For a finite signed combination
y=sum_G c_GG define its negative vertex mass by
N(y)=sum_{c_G<0}|c_G||V(G)|, combining equal connected graph classes
first. From (11), 0<=ell(G)<=2|V(G)| for every finite G. Therefore
ell(y)>=-2N(y). Any sequence y_n converging to x consequently satisfies

\[
 \liminf_n N(y_n)\ge-\tfrac12\ell(x)>0.                    \tag{12}
\]

For coefficient vertex variation C(y)=sum_G|c_G||V(G)|=V(y)+2N(y), this
also gives liminf C(y_n)>=1-ell(x)>1. The witness itself is positive
and has rooted measure variation one. Its finite graph approximations
require persistent signed cancellation.

## 7. Exact verification and what it establishes

Run from the repository root:

```sh
python -S research/local-completion/verify_positive_cone_encoding.py
```

The standalone verifier imports no graphlocal or third-party code. The
[result record](positive_cone_encoding_results.json) specifies its actual
finite action catalogues and fixtures. It checks the vertex/edge formulas,
degree bound, decoding from graph adjacency alone, arbitrary vertex
relabeling, generator loops, inverse arcs, and coincident label endpoints.
It separately exercises malformed encodings, completion of partial maps,
local decoding radii, and the affected-root bound. Root averaging is
checked against a transport to the centers; rational mixture weights
are checked against disjoint unions. Degree-cutoff locality is compared
with cutting full graphs before taking their balls.

These computations verify the finite mechanisms of the reduction. They
do not construct or test a non-sofic law, verify the external undecidability
proof, or compute the existential separator. The general statements use
the written arguments and the explicitly cited external theorem. This
checkpoint changes no package API or version and leaves intrinsic
automorphism invariance of the two cones open.

## References and checked passages

1. Lewis Bowen, Michael Chapman, Alexander Lubotzky, Thomas Vidick,
   *The Aldous–Lyons Conjecture I: Subgroup Tests*,
   [arXiv:2408.00110v1](https://arxiv.org/abs/2408.00110v1),
   submitted 31 July 2024. Checked: introduction's IRS/Schreier discussion
   and footnote 2; Theorem 1.10, Corollary 1.12, Theorems 7.3–7.4,
   Corollary 7.5 and its proof; Remark 1.14 for the existential witness.
2. Lewis Bowen, Michael Chapman, Thomas Vidick,
   *The Aldous–Lyons Conjecture II: Undecidability*,
   [arXiv:2501.00173v1](https://arxiv.org/abs/2501.00173v1),
   submitted 30 December 2024; PDF dated 3 January 2025.
   Checked: Theorem 1.1 and its formal statement, Theorem 2.31, against
   the hypotheses quoted as Theorem 7.4 in Part I. The remaining 207-page
   proof is an external dependency.

The encoding's root averaging and repair are proved above rather than
left to the cited decoration observation. Its application depends also
on this repository's [representation theorem](REPRESENTATION_THEOREM.md).
