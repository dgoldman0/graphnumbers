# Sparse defects and relative heat

30 September 2026. This note develops a controlled application of the local
completion. The local cancellation and limiting-element statements concern
our construction; trace-norm perturbation, uniformization, Cartesian heat
factorization, and localized sparse propagation are classical methods. No
priority or general numerical superiority claim is made.

The implementation is `graphlocal` version 0.2.0; see the
[library documentation](../../python/README.md),
[focused literature comparison](SPARSE_DEFECT_COMPARISON.md), and
[reproducible experiment](../../python/examples/defect_benchmark.py).

## 1. Exact cancellation near finitely many edited edges

Let G and G' be finite simple graphs on the same n-vertex set, differing in m
edges. Let W have edge set E(G) union E(G'), and let S contain the endpoints
of the edited edges. For radius R set A_R = {v: dist_W(v,S) <= R}.
Every root outside A_R has the same induced R-ball in G and G'. Therefore

T_R(G'-G) = sum_{v in A_R} [delta_{B_R(G',v)}-delta_{B_R(G,v)}].

One need only inspect the affected roots. Isomorphic rooted types are then
combined, which can produce further exact cancellation. A conservative
estimate is |A_R| <= 2m sum_{j=0}^R D_W^j, capped by n. The union degree
D_W, which may exceed the degrees of G and G', belongs in this bound.
The degree bound used for heat below instead bounds G and G' themselves.

For several groups of edits, let S_i be their endpoint sets, use the union
of the baseline and all modified graphs for distances, and suppose

    dist_W(S_i,S_j) > 2R  for i != j.

A rooted R-ball encounters at most one edit group. Consequently the full
radius-R correction is exactly the sum of the corrections obtained by
applying each edit group separately to the baseline. Separation is a
sufficient condition; interacting edits can sometimes cancel without it.
The full heat correction need not be additive: paths of sufficiently many
steps can visit multiple groups.

## 2. A trace bound gives a much smaller correction tail

Fix D>0 bounding the maximum degree of both G and G'. Write

    P_G = I - Delta_G/D,     lambda = tD.

These are real symmetric stochastic contractions in operator norm. An
edited edge e={u,v} contributes plus or minus b_e b_e^T to the Laplacian,
where b_e=e_u-e_v and ||b_e||^2=2. Thus the trace norm satisfies

    ||P_G'-P_G||_1 <= 2m/D.

Telescoping noncommuting powers and applying |tr(ABC)| <=
||A||_op ||B||_1 ||C||_op gives

    |tr(P_G'^j-P_G^j)| <= 2mj/D.                     (1)

For normalized graph differences divide this inequality by n. More
generally, for a finite combination

    X = sum_i c_i (G_i'-G_i),
    q = sum_i |c_i| m_i,

with common maximum-degree bound D, put d_j(X)=sum_i c_i
tr(P_G_i'^j-P_G_i^j). Then

    d_0(X)=0,     |d_j(X)| <= 2qj/D.                (2)

Here q is an absolute edit budget, including coefficients. A one-edge
unnormalized difference has q=1; its normalized version has q=1/n.
A usual global total-variation bound such as 2n is unnecessary.

Every finite combination X=sum_G c_G G with V(X)=0 has some edit
certificate: write X=sum_G c_G(G-|V(G)|K1), obtaining each parenthesized
pair by deleting all |E(G)| edges. Thus q=sum_G |c_G||E(G)| works.
A supplied sparse pairing can give a much smaller bound than this generic
presentation; a single cut has q=1 despite both terms having large size.

The relative heat trace is

    H_t(X) = exp(-lambda) sum_{j>=0} lambda^j d_j(X)/j!.

The induced rooted radius R=ceil(M/2) determines d_j for all j<=M.
As in EFFECTIVE_ANALYSIS_AND_HEAT.md, a boundary holding step that could
notice an omitted edge requires 2R+1 steps to leave the root and return;
leaving the ball altogether and returning requires 2R+2 steps.

For N distributed as Poisson(lambda), the omitted correction obeys

    |exp(-lambda) sum_{j>M} lambda^j d_j/j!|
       <= (2q/D) E[N 1_{N>M}]
       = 2qt Pr[N>=M].                             (3)

This depends on edited-edge mass, rather than total vertex mass. It can
therefore remain useful for corrections on increasing finite volumes.
For a normalized correction use q=m/n throughout.

There is also a simple whole-correction bound |H_t(X)|<=2qt. The stronger
bound |H_t(X)|<=q follows by changing edges one at a time: a positive rank-one
Laplacian update interlaces eigenvalues and decreases the heat trace by a
number in [0,1]. Combining the two gives |H_t(X)|<=q min(2t,1). In a pure
edge deletion the heat-trace correction is nonnegative, and in a pure
addition it is nonpositive. Mixed edits have no prescribed sign.

### A fully rational interval

Assume rational t>=0 and coefficients, and M+2>lambda. Put

    term = lambda^M/M!,
    S = sum_{j=0}^M lambda^j/j!,
    T = term * lambda/(M+1) / [1-lambda/(M+2)],
    beta = 2qt,
    A = sum_{j=0}^M lambda^j d_j/j!.

Then exp(lambda)=S+E with 0<=E<=T. The omitted normalized correction has
absolute value at most beta(term+E)/(S+E). This ratio is increasing in E
because S>=term, so a rational tail bound is

    B = beta (term+T)/(S+T).

The retained contribution lies in hull{A/S,A/(S+T)}. Adding [-B,B] gives
a valid interval. Since |A|<=beta(S-term), its radius is at most

    beta (term + 3T/2)/(S+T).                     (4)

For local weighted histogram error eta, the retained numerator differs
from Ahat by at most S eta. Intersect [Ahat-S eta,Ahat+S eta] with
[-beta(S-term),beta(S-term)], take the hull of all quotients of the two
numerator endpoints by S and S+T, and add [-B,B]. The radius is bounded
by (4) plus eta. This avoids sign errors for signed corrections and
requires no numerical exponential. The interval may be intersected with
[-q min(2t,1),q min(2t,1)] and a known sign interval when the input
certifies the edit-generated bounds proved above. At t=0 return exactly
zero; when D=0 all graph differences vanish. Increasing M makes every
error contribution arbitrarily small.

If separated edit groups satisfy Section 1 with M<=2R, their retained
moments add exactly. If their total edit budget is q, the difference
between the full heat correction and the sum of the individual heat
corrections is at most

    4qt Pr[Poisson(tD)>=M].                         (5)

This follows by bounding the two omitted tails separately. A common D
must bound the baseline, fully modified, and individually modified graphs.

## 3. An explicit defect element beyond finite signed measures

Let C_n be the n-cycle and P_n the path on n vertices obtained by deleting
one edge, and set E_n=P_n-C_n, without normalization. For every r>=1 and
n>=2r+2,

    T_r E_n = 2 sum_{j=0}^{r-1}
                 delta_{(P_{r+j+1}, root j from an endpoint)}
              - 2r delta_{(P_{2r+1}, root at its center)}.         (6)

Indeed, each of the two path endpoints has r affected roots at distances
j=0,...,r-1; all remaining roots have the line r-ball. The cycle r-ball
is the centered (2r+1)-vertex path. For r=0 the difference is zero.

The right side stabilizes exactly at every radius, including every
polynomial weight. Thus E_n converges in A to an element E. It is the
local signed response to cutting the infinite line at one edge. Its
positive rooted types in (6) have sizes r+1,...,2r, so none coincides with
the negative line type. Consequently

    ||T_r E||_1 = 4r,
    p_{r,k}(E) = 2 sum_{j=0}^{r-1}(r+j+1)^k + 2r(2r+1)^k.

The unweighted marginal variations are unbounded. By the proved signed
measure characterization, E has no finite signed-measure representation.
It nevertheless has a uniform maximum-degree bound 2, vertex mass zero,
and a sequence of finite approximants with absolute edit budget q=1.
It is a concrete nonzero element beyond the finite-measure model, with
an immediately interpretable relative observable.

## 4. A precisely specified domain for relative heat

For fixed finite D>0 and q>=0, let K_(D,q) be the closure in A of the finite
combinations of paired same-vertex-set graph differences from Section 2,
with common degree bound at most D and absolute edit budget at most q.
The budget is certified additional information. A chosen finite
presentation supplies an upper bound; arbitrary completed elements do
not come with such a certificate.

Every d_j is determined by a finite-radius local observable, and hence is
continuous on this degree-bounded subspace. Taking limits in (2) gives
|d_j(X)|<=2qj/D for X in K_(D,q). Therefore the series in Section 2 converges
absolutely and defines H_t(X), with exactly the same tail bound (3).
It is independent of the approximating graph sequence, because the local
moments d_j(X) are determined by X and the remaining series tail tends
uniformly to zero. It is also independent of which admissible D and q are
reported. To compare two D values, choose their maximum; uniformization
at this larger value gives the same finite-graph heat trace and then the
same limit. Enlarging q only weakens the bound.

On each fixed K_(D,q), relative heat is uniformly continuous for the
inherited local topology. For R=ceil(M/2),

    |H_t(X)-H_t(Y)|
      <= ||T_R(X-Y)||_1 + 4qt Pr[Poisson(tD)>=M].    (7)

This follows since each retained rooted return probability lies in [0,1]
and the retained Poisson weights sum to at most one. The ordinary defining
seminorm p_(R,1) can replace the unweighted norm on the right.

The union over finite budgets is a vector space at fixed D: addition adds
budgets and scalar multiplication scales them. H_t is linear there. No
continuity on the entire unrestricted signed completion follows. Nor does
membership in this controlled domain follow merely from maximum degree
or from the existence of an arbitrary local approximation procedure.
The domain need not be assumed closed under arbitrary algebra products.

In particular E belongs to K_(2,1). Formula (4) provides finite rational
certificates for its relative heat despite the infinite supremum of its
marginal variations. Each individual marginal has finite variation.

## 5. Exact relative heat of the cut line

The path Laplacian eigenvalues are 2-2cos(pi k/n), k=0,...,n-1; the cycle
ones are 2-2cos(2pi k/n), k=0,...,n-1. The path formula follows by
substituting v_k(j)=cos[(j+1/2)pi k/n], j=0,...,n-1, including both
endpoint equations; the cycle uses the Fourier vectors. With
f(theta)=exp[-2t(1-cos(theta))], the C_(2n) eigenvalues pair at theta and
2pi-theta. Therefore the exact finite identity is

    tr exp(-t Delta_Pn)
       = [tr exp(-t Delta_C(2n)) + 1-exp(-4t)]/2.

Subtracting the C_n trace gives

    H_t(P_n-C_n)
       = [1-exp(-4t)]/2 + H_t(C_(2n)/2-C_n).

The residual X_n=C_(2n)/2-C_n has edit budget at most two: two disjoint
n-cycles can be joined into one 2n-cycle by removing one edge from each
and inserting two cross edges, a total of four edits, followed by the
coefficient 1/2. All graphs have maximum degree two. Also T_R X_n=0
whenever n>=2R+2. The retained relative heat moments consequently vanish
through M whenever n>=2ceil(M/2)+2, and (3) bounds the residual by

    |H_t(X_n)| <= 4t Pr[Poisson(2t)>=M].

Let M and then n grow. This proves the residual tends to zero for every
fixed t, using only the already proved edit-tail estimate. Hence

    H_t(E) = [1-exp(-4t)]/2.                        (8)

Comparing power-series coefficients in exp(Dt)H_t(E) gives a useful
independent moment identity, for any D>=2:

    d_j(E) = [1-(1-4/D)^j]/2.

At D=2 this is one for odd j and zero for even j, including d_0=0.
In particular the heat moments stay bounded even though the full local
marginal variations grow without bound. This provides an exact test
oracle and permits a stronger specialized tail certificate; the generic
edit-budget argument does not require this special formula.

The small-t derivative is 2, matching the Laplacian trace decrease from
a single removed edge. The formula is nonnegative and approaches 1/2 as
t grows. For every fixed finite n the correction instead approaches zero
as t tends to infinity, since P_n and C_n are both connected. Thus the
infinite-volume and long-time limits do not commute. Finite-volume
validity at a specified time is controlled by (3), with the radius
required to retain the relevant moments.

## 6. Combining a defect with Cartesian structure

Let H be a fixed finite graph on h>0 vertices. The pair
P_n square H and C_n square H differs in exactly h edges. Hence

    (P_n-C_n) U(H),     U(H)=H/h,

has budget q=h/h=1 and degree bound 2+maxdeg(H). It converges locally to
E U(H), and finite Cartesian heat factorization plus the controlled
limit gives

    H_t(E U(H)) = [1-exp(-4t)]/2 * H_t(U(H)).       (9)

For d>=0 use L, the normalized infinite-line element, and approximate
L^d by U(C_N)^d. Before normalization the cut is replicated N^d times;
the coefficient 1/N^d leaves the edit budget exactly one. Thus

    E L^d belongs to K_(2d+2,1),
    H_t(E L^d) = [1-exp(-4t)]/2 * h_line(t)^d,
    h_line(t) = (1/(2pi)) integral_0^(2pi)
                   exp[-2t(1-cos(theta))] dtheta.                 (10)

Both n and N may tend to infinity through any sequence for which each
fixed rooted radius stabilizes. This is a relative heat response per
transverse volume for a planar cut in a Cartesian lattice. The graph
number records the defect locally without constructing the full lattice.

More generally multiplying a finite controlled difference of budget q
by a finite combination Z=sum_H c_H H produces budget at most
q sum_H |c_H||V(H)|, with degrees added. This follows by replicating each
edited edge once per vertex of H.

There is also a direct moment argument for multiplication by a
finite-signed-measure element Y of degree bound D2 and variation C,
without assuming a bounded-variation finite-graph approximation to Y.
Let X have degree bound D1>0 and |d_i(X)|<=2qi/D1. With D=D1+D2,
a=D1/D and b=D2/D, Cartesian Laplacian addition gives

    d_j(XY) = sum_{i=0}^j binom(j,i) a^i b^(j-i)
                      d_i(X) m_(j-i)(Y),

where m_l(Y) is the integrated l-step return for I-Delta_Y/D2 and
|m_l(Y)|<=C. Thus

    |d_j(XY)| <= 2qCj/D.

When D2=0 the formula reduces directly to scalar multiplication. The
binomial identity extends as follows. Each fixed-order return moment is
a continuous local polynomial observable: closed walks of that length
are determined by a finite ball and bounded by a polynomial in its size.
Approximate X and Y by the degree-filtered finite signed combinations
from the representation theorem. Their products converge by continuity
of multiplication, so every term in the finite binomial sum converges.
No variation bound on those approximants is required for this identity.
Absolute convergence of the
uniformization series permits rearrangement and proves

    H_t(XY)=H_t(X) H_t(Y).

Consequently the relative-heat certificate propagates with q'=qC and
D'=D1+D2. The additional cap |H_t(XY)|<=qC also propagates if it holds
for X, since |H_t(Y)|<=C. Such a product need not be asserted to lie in
the particular finite-edit closure K_(D,qC); its local moment bounds and
heat cap supply the required certificates directly.

This suggests an effective class generated from controlled finite edits
and their limits by sums, scalar multiplication, and multiplication by
bounded-degree finite-variation elements. Each constructor proves its
metadata. Merely knowing |d_j|<=2qj/D guarantees the tail bound and
|H_t|<=2qt; it does not by itself prove the extra cap |H_t|<=q. The cap
comes from the edit construction or from a separate certificate. A
universal multiplicative heat map on all of A is not asserted.

## 7. An equally volume-independent Krylov bound

The correction-tail estimate is also available to a conventional matrix
method. Let U be an exact orthonormal basis containing the block Krylov
space K_m(Delta_G,B), where the columns of B are edited-edge incidence
vectors. Set A=U^T Delta_G U and A'=U^T Delta_G' U. The symmetric trace
exactness theorem of Cortinovis, Kressner and Massei,
[Theorem 2](https://arxiv.org/pdf/2107.04337), gives

    tr p(A') - tr p(A) = tr p(Delta_G') - tr p(Delta_G)

for every polynomial p of degree at most 2m. Both compressed Laplacians
have spectrum in [0,2D], so I-A/D and I-A'/D are symmetric contractions.
They need not be entrywise stochastic. Compression does not increase
trace norm, hence their difference has trace norm at most 2q/D.
Their j-th relative trace moment therefore also has magnitude at most
2qj/D. The full and compressed moments cancel through order 2m;
bounding the two remaining tails gives

    |H_t(G'-G) - tr(exp(-tA')-exp(-tA))|
       <= 4qt Pr[Poisson(tD)>=2m].                 (11)

This dimension-independent estimate follows by combining the established
polynomial exactness theorem with Section 2. It is a bound on the
conventional method too, not a computational advantage exclusive to the
graph-number representation. A shared degree/tolerance comparison can
take m=ceil(M/2) and use twice the rational bound B in Section 2.
Our choice of M with (4)<=epsilon/2 then gives 2B<=epsilon.

Equation (11) assumes the exact Krylov subspace and exact compressed
evaluation. Exact rank deflation preserves it; discarding a small but
nonzero direction numerically may not. Floating point orthogonality,
projection and exponential evaluation also contribute errors. The
benchmark reports (11) as an exact-arithmetic truncation bound and checks
observed values against rational intervals; it does not present the
floating point output as a fully validated enclosure.

## 8. Discriminating checks and comparisons

1. Verify (6) exactly for several n and radii, including n=2r+2 and a
   smaller n where stabilization fails. Check the variation 4r directly.
2. Compare the rational relative-heat interval with an independent
   high-precision evaluation of (8), including t=0, small positive t,
   t comparable to one, and larger t requiring more moments.
3. For a finite cycle cut, compare to the independent full Laplacian
   eigenspectrum. Benchmark local correction cost as n grows, holding t,
   accuracy, and number of defects fixed. Report input construction and
   neighborhood extraction costs separately from moment evaluation.
4. Test (9) and (10), comparing a factorized correction with explicit
   finite products. Preserve the transverse normalization when reporting
   the edit budget; an unnormalized product has h or N^d edits.
5. Test additivity for well-separated and overlapping edit clusters.
   Exact local additivity should be claimed only at an explicitly chosen
   radius. A nearby pair supplies a necessary interacting counterexample.
6. Use a conventional implementation of the same local polynomial trace
   correction as a serious baseline. The sparse trace bound and locality
   are available without graph-number terminology; reproducing their
   speedup is not evidence of mathematical novelty.

The discriminating mathematical gain is that E and E L^d are genuine
completed algebra elements with coherent signed local data, and their
relative heat is computable on an explicit controlled domain extending
beyond finite signed measures. Practical advantages require measuring
reuse of local types, factorization, repeated queries, and parameterized
or interacting defects against established sparse and perturbative
methods. A generic dense eigensolve alone is too weak a comparison.

## 9. Recorded outcome

The [benchmark record](../../python/results/defect_benchmark.json) contains
14 finite and four infinite comparisons at fixed absolute tolerance 1e-8.
The [library README](../../python/README.md#measured-defect-application)
summarizes timings and qualifications. Shared block Krylov is faster in
all finite cases. Every numerical reference lies inside its rational
interval, and the largest Krylov/dense disagreement is 1.98e-13. The
current exact histogram implementation therefore establishes no speed
advantage for this scalar heat task.

The infinite calculations demonstrate that the controlled relative-heat
domain does reach elements beyond finite signed measures, including E
and E L. Certified scalar factorization is substantially faster than
constructing product local histograms, while being equally available
from the classical Cartesian Laplacian identity. The mathematical gain
for this construction is the reusable geometric element and its proved
analytic contract. Priority and an advantage over conventional methods
for applications remain unestablished.

A useful next comparison would reuse one defect's local data across
several independent observables and many background products, and then
study interacting defects. That tests the value of retaining geometry
rather than using a representation dedicated to one scalar heat query.
