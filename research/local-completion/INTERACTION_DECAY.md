# Geometric vanishing orders and decay of finite edge interactions

30 September 2026. This note continues
[the planar-defect analysis](PLANAR_DEFECTS.md). All graph coefficients and
operator traces are unnormalized unless an explicit scalar is supplied.
The new bounds connect the geometric separation of finitely many edits to
their joint heat response, and produce moment profiles that remain valid
under Cartesian multiplication.

## 1. Setup and the common support graph

Let G be a finite graph, or a graph of uniformly bounded degree, and let
e_1,...,e_k be distinct valid edge edits, k>=1. An edit may insert an absent
edge or delete a present edge. For S contained in [k], let G_S apply the
edits in S, and define

\[
 C_F=\sum_{S\subseteq[k]}(-1)^{k-|S|}[G_S].                 \tag{1}
\]

Let U contain every edge present in any G_S: it is G with all inserted
edges added, before any deletions. Fix D>=maximum degree(U), D>0. Write
b_i for either orientation of the incidence vector of e_i, and let

\[
 \delta_{ij}=\operatorname{dist}_U(\operatorname{supp}b_i,
                                      \operatorname{supp}b_j). \tag{2}
\]

The supports consist of the two endpoints. Set delta_ii=0. Distances
between supports need not satisfy the triangle inequality; the arguments
below use only finite propagation and sums along specified cyclic orders.

Interpolate each edited edge's weight linearly between its original and
final value. Every interpolated graph has nonnegative weights, weighted
degree at most D, and Laplacian L(s), s in [0,1]^k. Consequently
P(s)=I-L(s)/D is a self-adjoint contraction supported on U together with
diagonal entries. Its derivative in direction i is

\[
 V_i=\partial_i P(s)=\eta_i b_i b_i^T/D,\qquad \eta_i\in\{-1,1\}.
                                                                    \tag{3}
\]

The sign is positive for a deletion and negative for an insertion.
For all a>=0,

\[
 |b_i^TP(s)^a b_j|\le2,\qquad
 b_i^TP(s)^a b_j=0\quad\hbox{if }a<\delta_{ij}.              \tag{4}
\]

The first statement follows from the contraction bound and ||b_i||=sqrt2;
the second follows because a matrix power of degree a cannot move between
supports more than a graph steps apart.

## 2. Cyclic tours bound the mixed moments

Let S_k^circ be the (k-1)! cyclic orders anchored at label 1; orientations
are retained separately. For sigma=(1,sigma_2,...,sigma_k), set

\[
 \tau_\sigma=\sum_{i=1}^k
       \delta_{\sigma_i,\sigma_{i+1}},\qquad \sigma_{k+1}=\sigma_1,
 \quad \tau_* = \min_\sigma\tau_\sigma.                       \tag{5}
\]

For k=1 the sole tour has cost zero. Write d_j for the mixed trace of
P^j associated with (1), interpreting the alternating operator sum before
taking a trace on infinite graphs. Then

\[
 |d_j|\le \frac{2^k}{D^k}\,j
     \sum_{\sigma\in S_k^\circ}
         \binom{j-\tau_\sigma-1}{k-1}.                       \tag{6}
\]

The binomial term is defined as zero unless j>=k+tau_sigma. In particular,

\[
 d_j=0\quad(j<k+\tau_*).                                    \tag{7}
\]

Proof. Cube integration identifies the mixed finite difference of Tr P^j
with the integral of its mixed derivative. The derivative consists of all
words containing each V_i exactly once, separated by powers of P(s).
For a fixed anchored cyclic order, cyclic invariance of the trace combines
the first and last powers. If the resulting cyclic gaps are a_1,...,a_k
with sum j-k, the total multiplicity is
sum_i(a_i+1)=j. Thus the integrand is a sum, over anchored cyclic orders
and such gaps, of j times

\[
 \operatorname{Tr}(V_{\sigma_1}P^{a_1}\cdots V_{\sigma_k}P^{a_k})
   =\frac{\prod_i\eta_i}{D^k}
       \prod_i b_{\sigma_i}^TP^{a_i}b_{\sigma_{i+1}}.          \tag{8}
\]

Equation (4) requires a_i>=delta_(sigma_i,sigma_(i+1)). The number of
nonnegative gap choices satisfying these restrictions is the binomial
coefficient in (6), and every remaining product has absolute value at
most 2^k. Integrating over the unit cube proves (6).

When every support distance is zero, (6) is exactly the earlier bound
2^k(j)_k/D^k. Geometric onset is a guaranteed vanishing order; incidence
cancellations can force further vanishing, including an identically zero
heat interaction. No nonzero leading coefficient or sign is claimed here.

If the edited supports lie in different connected components of U, every
tour has an infinite gap and the whole interaction is zero. In the finite
graph algebra this also follows immediately from disjoint-union addition:
each component's term is independent of at least one edit, so its
inclusion-exclusion coefficient cancels. The runtime treats the empty edit
set as zero, consistently with EdgeInteraction's existing convention.

## 3. Separation gives an all-time heat bound

Put lambda=tD and let N be Poisson with mean lambda. Uniformization gives
H_t(C_F)=exp(-lambda) sum_j lambda^j d_j/j!. Since

\[
 j\binom{j-\tau-1}{k-1}
 \le\frac{(j)_k}{(k-1)!}\,1_{j\ge k+\tau},
\]

equation (6) gives the all-time bound

\[
 |H_t(C_F)|\le
 \frac{2^kt^k}{(k-1)!}\sum_{\sigma\in S_k^\circ}
       \Pr\{N\ge\tau_\sigma\}
 \le 2^kt^k\Pr\{N\ge\tau_*\}.                              \tag{9}
\]

The identical argument bounds the uniformization tail after moment M:

\[
 \left|e^{-\lambda}\sum_{j>M}\frac{\lambda^j d_j}{j!}\right|
 \le\frac{2^kt^k}{(k-1)!}\sum_{\sigma\in S_k^\circ}
       \Pr\{N\ge\max(\tau_\sigma,M+1-k)\}.                  \tag{10}
\]

These estimates handle mixed insertions and deletions and make no sign
assumption. Multiplying the graph interaction by a scalar c multiplies
all bounds by |c|.

For fixed k, D and t, they give factorial suppression with separation.
For example, if the edits split into two nonempty groups and the endpoint
support of one group is at distance at least R from that of the other,
every cyclic order crosses between the groups at least twice. Hence
tau_*>=2R and |H_t|<=2^k t^k Pr{N>=2R}. If k>=2 and every pair of distinct edited
supports is at distance at least R, tau_*>=kR. More generally m>=2 groups
with pairwise support distance at least R require at least m crossings,
so tau_*>=mR. The elementary bound

\[
 \Pr\{N\ge q\}\le\min\{1,\lambda^q/q!\}                  \tag{11}
\]

follows by applying Markov's inequality to the falling factorial (N)_q.
Thus the estimates are effective at every separation and demonstrate
decoupling of distant groups at fixed time. The constants grow with t;
no decay as t tends to infinity is asserted.

## 4. A geometric moment profile and its products

The elementary inequality

\[
 j\binom{j-\tau-1}{k-1}
   \le\frac{(j)_{k+\tau}}{(k+\tau-1)!}                     \tag{12}
\]

follows, on writing j=k+tau+m, from
binom(m+k-1,k-1)<=binom(m+k+tau-1,k+tau-1). Choose the degree cap D_0.
Equations (6) and (12) supply the finite profile

\[
 A_r=\sum_{\sigma:\,k+\tau_\sigma=r}
          \frac{2^k D_0^{\tau_\sigma}}{(k+\tau_\sigma-1)!},
 \qquad |d_j(C_F;D_0)|\le\sum_r A_r\frac{(j)_r}{D_0^r}.       \tag{13}
\]

The existing binomial averaging argument extends this same profile to
every D>=D_0. Its coefficients are fixed once D_0 is chosen; they are not
recomputed at the larger D. Consequently all profile addition, scaling
and Cartesian product rules from the planar note apply. Geometric
separation therefore becomes analytic data that propagates through the
graph arithmetic.

Each anchored cycle also gives the simple total-heat bound

\[
 B_\sigma(t)=\frac{2^kD_0^{\tau_\sigma}
                     t^{k+\tau_\sigma}}{(k+\tau_\sigma-1)!}. \tag{14}
\]

One may take the smaller of B_sigma and the corresponding contribution
in (9), separately for each cycle, and then sum. Both (6) and (13) are
valid certificates; the higher-order profile can be sharper at small
times and coarser at large times. This is an additional profile for the
same element, not a change to its topology or arithmetic.

## 5. Exact rational probability enclosures

For lambda>=0 choose M with M+2>lambda. Set

\[
 S_M=\sum_{j=0}^M\lambda^j/j!,\quad
 T_M=\frac{\lambda^{M+1}}{(M+1)!}
             \frac{1}{1-\lambda/(M+2)}.
\]

Then S_M<=exp(lambda)<=S_M+T_M. For 1<=q<=M+1,

\[
 1-\frac{S_{q-1}}{S_M}
 \le\Pr\{N\ge q\}
 \le1-\frac{S_{q-1}}{S_M+T_M}.                              \tag{15}
\]

For q>M+1 the interval [0,T_M/(S_M+T_M)] is valid; q<=0 gives [1,1].
Equations (9), (10), (14) and (15) give rational magnitude certificates
without evaluating any rooted local histogram or signed graph subset sum.
Increasing M decreases the probability uncertainty to zero. If a magnitude
bound is already below the requested tolerance, the calculation can stop
before resolving each probability accurately.

## 6. Infinite-volume interpretation and implementation

For a bounded-degree infinite graph and finitely many edits, polynomial
differences are finite rank and supported within a finite distance of the
edited endpoints. Heat differences are trace class by Duhamel's formula.
All derivative terms in (8) contain finite-rank factors, making cyclic
trace manipulation legitimate. The bounds are uniform on finite
exhaustions with the same degree cap. Local polynomial moments stabilize,
and the Poisson majorants justify the heat limit. Thus these statements
apply to the corresponding locally stabilized elements of the completion.
They do not cover arbitrary infinite collections of edited edges or all
elements of the completion.

The finite implementation is `python/src/graphlocal/interaction_bounds.py`:

* `interaction_geometry(value)` uses all edges ever present, then counts
  the costs of the (k-1)! anchored cyclic orders. It records support
  distances, a guaranteed vanishing order and the profile (13).
* `geometry.moment_bound(j)` evaluates (6) with exact rational arithmetic.
* `interaction_heat_bound(value_or_geometry,t)` returns a symmetric heat
  interval using (9), (14) and rational probability enclosures. Its
  tolerance controls uncertainty in the analytic bound, or certifies that
  the whole magnitude is below tolerance. A general interaction's returned
  interval need not have that small a radius. `after_step=M` bounds the
  uniformization tail instead, using (10).
* `GeometricInteraction(source)` retains the same source element and
  forwards its local and finite operations while using profile (13).
  Existing `controlled_heat` and Cartesian profile propagation therefore
  apply directly to the wrapped element and its algebraic expressions.

Exact bridge reduction is applied by EdgeInteraction before the geometry
is computed, so its retained cuts and absolute normalization govern the
certificate. The reduction sign does not affect a magnitude bound. Users
can compare unreduced and reduced certificates because both describe the
same graph element. Neither bound is promised to dominate the other.

Six focused tests compare the moment bounds with independent full rational
matrix expansions, test mixed insertion/deletion support geometry, check
profiles at a larger degree parameter, verify heat and omitted-tail
enclosures against a 70-digit star closed form, and check wrapped products
against finite-graph heat. A path example with cuts (0,1), (6,7), (11,12)
has guaranteed onset 22; at t=1/4 the geometry-only magnitude certificate
is below 10^-18. A three-arm spider with two edges per arm and one terminal
cut per arm has onset 9, agreeing with the previously verified first
nonzero moment. These onset matches are examples, not an assertion that
geometric lower bounds are always attained.

For selected cuts in an ambient tree, the stronger theorem in
[the higher-interaction geometry note](HIGHER_INTERACTION_GEOMETRY.md)
does identify the first nonzero coefficient. If their minimal spanning
subtree has s edges and ell leaf edges, bridge reduction leaves ell
terminal cuts and the exact onset is nu=2s-ell. Its reduced order-ell
profile therefore gives the especially transparent estimate

\[
 |H_t(C_F)|\le 2^\ell t^\ell
       \Pr\{N\ge 2(s-\ell)\}.                              \tag{16}
\]

The separation entering the analytic tail is twice the number of edges
of the connecting tree left after its terminal edges are removed.

The [reproducible bound example](../../python/examples/geometry_bound_examples.py)
compares the original and geometric profiles for three equally spaced
cuts on cycles of lengths 18, 30 and 48, always at t=1/2 and degree cap
two. The original order-three profile gives magnitude bound one.
The geometric profiles give respective bounds 2/17!, 2/29! and 2/47!,
all below 10^-10, and `controlled_heat` needs only its zeroth moment.
The [result record](../../python/results/geometry_bound_examples.json)
also contains the generic-profile certificates and all rational data.
This demonstrates an analytic improvement in the certified magnitude and
required moment order. No timing comparison is implied.

## 7. Relation to established methods

The support-distance argument is a standard ingredient in localization
estimates for sparse matrix functions. Michele Benzi and Nader Razouk,
“Decay bounds and O(n) algorithms for approximating functions of sparse
matrices,” *Electronic Transactions on Numerical Analysis* 28 (2007),
16–39, develops this connection between graph distance, matrix powers
and decay of matrix functions. The publisher's
[full paper](https://etna.ricam.oeaw.ac.at/vol.28.2007-2008/pp16-39.dir/pp16-39.pdf)
provides the primary reference.

Martin Schaden's
[“Irreducible Many-Body Casimir Energies of Intersecting Objects”](https://arxiv.org/abs/1011.2475)
uses inclusion-exclusion to isolate joint responses involving every
object. The operator settings and hypotheses differ, so its conclusions
are not imported into the present graph calculation. Bernhard Beckermann,
Daniel Kressner and Marcel Schweitzer's
[“Low-rank updates of matrix functions”](https://arxiv.org/abs/1707.03045)
is relevant to the finite-rank computational viewpoint.

The explicit derivation here specializes these classical ingredients to
cyclic finite-edge interactions, sums their gap constraints into a tour
histogram, and carries the resulting bounds into the completion's
compositional moment profiles. This note makes no priority claim for
those ingredients or for an algorithmic advantage over established
matrix-function techniques.
