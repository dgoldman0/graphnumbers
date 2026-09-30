# Interacting cuts and reuse of local data

30 September 2026. Interacting defects and reusable observable calculations.
Throughout P_n is the n-vertex path, C_n the simple n-cycle (n>=3), L is
the normalized infinite-line element, and E=lim(P_n-C_n) is the cut-line
relative element developed in [the sparse-defect note](SPARSE_DEFECTS_AND_RELATIVE_HEAT.md). Laplacians have
unit rate on each incident edge.

## 1. Two cuts with fixed separation

Fix an integer ell>=1. On C_n, n>ell, delete the edges joining n-1 to 0
and ell-1 to ell. The remaining graph is P_ell disjoint union P_(n-ell).
Its combined correction and its two-cut interaction are

    D_(ell,n) = P_ell + P_(n-ell) - C_n,
    I_(ell,n) = D_(ell,n) - 2(P_n-C_n)
              = P_ell + P_(n-ell) + C_n - 2P_n.

The second expression subtracts the response of each cut separately. The
finite edit budgets are two for D and four for I; all graphs have maximum
degree two. Coefficients are unnormalized. Normalizing by n would make
these fixed defects disappear in the local limit.

For every fixed radius r, sufficiently large n gives

    T_r P_(n-ell) = (n-ell) T_r L + T_r E,
    T_r C_n = n T_r L,
    T_r P_n = n T_r L + T_r E.

For r>=1, n>=max(ell+2r,2r+2) suffices. Hence the limits in A exist and
are the exact algebra identities

    D_ell = E + P_ell - ell L,
    I_ell = P_ell - ell L - E.                    (1)

They have degree bound two and certified edit budgets at most two and
four respectively. Their vertex masses are zero. Formula (1) alone
would hide the uniform edit bound because its individual terms have
different masses; the finite common-background construction proves it.

For completeness, F_ell=P_ell-ell L also has an edit certificate of at
most 2ell-1. Approximate L by U(C_N). A finite zero-mass combination
sum c_G G has edit budget at most sum |c_G||E(G)|, obtained by pairing each
G with |V(G)| isolated vertices. For P_ell-ell U(C_N) this gives
(ell-1)+ell=2ell-1 independently of N. Thus the identities also hold in
the controlled domain on which relative heat is linear, consistently
with ordinary heat integration for F_ell, a finite-variation element.
For ell=1 this gives the improved interaction budget two; four remains
a valid uniform bound for all ell.

## 2. Exact radius of first interaction

If ell>=2r, the two endpoint layers of P_ell each contain r roots and
have the same rooted balls as the separate cut endpoints. Even at
ell=2r they match: the opposite endpoint can be at distance r, where
its missing exterior edge is outside the induced r-ball. Therefore

    T_r I_ell = 0  for 0<=r<=floor(ell/2).          (2)

This threshold is exact. At r=floor(ell/2)+1, a central root of P_ell
has its entire component in the r-ball, with rooted eccentricity r-1.
Every rooted type in T_r L and T_r E instead has eccentricity exactly r.
The central rooted type therefore has a positive uncancelled coefficient
in T_r I_ell. The first nonzero marginal occurs at precisely

    r_first = floor(ell/2)+1.

A local observable requiring only radius floor(ell/2) cannot detect the
interaction. This is a statement about the whole signed local array,
and consequently holds simultaneously for every observable at that
radius, including observables unrelated to heat.

For r>=ell, every rooted P_ell type has size ell. The negative ray-ball
types from -E have sizes r+1,...,2r, and the positive line type has size
2r+1 and coefficient 2r-ell. These supports are disjoint. It follows that

    ||T_r I_ell||_1 = ell + 2r + (2r-ell) = 4r,
    ||T_r D_ell||_1 = ell + 2r + (2r+ell) = 4r+2ell.

Both limits therefore lie beyond finite signed measures on rooted
graphs. Their relative observables are meaningful because of the
additional uniform edit certificates, not because these marginal
variations stay bounded.

## 3. Exact heat interaction and its sign

Let

    h(t) = H_t(L) = (1/pi) integral_0^pi exp[-2t(1-cos theta)] dtheta,
    e(t) = H_t(E) = [1-exp(-4t)]/2,
    p_ell(t) = tr exp(-t Delta_Pell)
             = sum_{a=0}^{ell-1} exp[-t(2-2cos(pi a/ell))].

The relative heat interaction is

    J_ell(t) = H_t(I_ell) = p_ell(t)-ell h(t)-e(t),                 (3)

and the combined response is e(t)+p_ell(t)-ell h(t). This follows from
(1), the controlled linearity justified in Section 1, and the usual
finite-variation heat functional for F_ell.

For ell>=2 the exact path/doubled-cycle identity gives

    J_ell(t) = tr exp(-t Delta_C(2ell))/2 - ell h(t)
             = ell [H_t(U(C_(2ell)))-h(t)].         (4)

The condition ell>=2 matters: C_2 is not a simple degree-two cycle.
At ell=1 use J_1(t)=1-h(t)-e(t) directly.

To express the result without cancellation, define for integers a>=0

    b_a(t) = sum_{j>=0} t^(a+2j)/(j!(a+j)!).

This is the usual modified Bessel function I_a(2t), but the displayed
nonnegative series is all that is needed. Expansion by walks on the
line gives h(t)=exp(-2t)b_0(t). Lifting a closed cycle walk to the line
shows its displacement is a multiple of the cycle length, so (4) gives

    J_ell(t) = 2ell exp(-2t) sum_{k>=1} b_(2ell k)(t).              (5)

For ell=1 the same formula follows from
cosh(2t)=b_0(t)+2 sum_{k>=1}b_(2k)(t) and the direct expression above.
Thus (5) is valid for every ell>=1, including the one-vertex segment.

Every term is nonnegative, with a strictly positive first term when
t>0. Therefore the two deleted edges have a strictly positive heat
interaction in this family. This sign is derived for cuts of the line;
it is not asserted for arbitrary edits on arbitrary graphs.

## 4. Moment order, separation, and time limits

Set P=I-Delta/2. Let d_j(I_ell) be the relative trace of P^j, defined by
the finite local moment. Comparing (5) with uniformization yields

    d_j(I_ell) = 2ell * 2^(-j) *
                  sum_{k>=1} binom(j,j/2+ell k)                 (6)

when j is even, with out-of-range binomial coefficients interpreted as
zero. For odd j the moment is zero. In particular

    d_j(I_ell)=0 for j<2ell,
    d_(2ell)(I_ell)=2ell/2^(2ell).

The heat interaction has the expansion

    J_ell(t) = t^(2ell)/(2ell-1)! + O(t^(2ell+1))                 (7)

as t tends to zero with ell fixed. More explicitly the next coefficient
is -2/(2ell-1)!. The whole local array becomes distinguishable at radius
floor(ell/2)+1, while this particular spectral observable first changes
at polynomial order 2ell. Observable choice therefore matters.

There is a useful stronger bound on these particular moments. For even
j, the simple-walk probabilities

    p_j(2a)=2^(-j) binom(j,j/2+a)

are nonincreasing as a>=0 grows. Partitioning positive a into consecutive
blocks of length ell gives

    ell sum_{k>=1} p_j(2ell k) <= sum_{a>=1}p_j(2a).

Consequently

    0<=d_j(I_ell)<=1-p_j(0)<1

for even j, while the odd moments are zero. The upper bound is exactly
the ell=1 moment. Summing the nonnegative Poisson-weighted series proves

    0<J_ell(t)<=J_1(t)<1/2  for t>0,
    J_ell(t)<=Pr[Poisson(2t)>=2ell].                (8)

The last inequality uses vanishing moments below order 2ell and the
uniform moment bound one. It quantifies decay with cut separation at a
fixed time. It is sharper for this family than the general edit-budget
bound. It also permits a specialized uniformization truncation tail of
at most Pr[Poisson(2t)>M], despite ||T_r I_ell||_1 growing without bound.

For the strict upper bound in (8), observe
J_1(t)=1/2+exp(-4t)/2-h(t), and h(t)>=exp(-2t)>exp(-4t)/2.

For fixed ell, p_ell(t) tends to one, h(t) tends to zero, and e(t) tends
to one half. Thus

    lim_(t->infinity) J_ell(t)=1/2,
    lim_(t->infinity) H_t(D_ell)=3/2.              (9)

The leading correction comes from the line. Expanding
2(1-cos theta)=theta^2+O(theta^4) in its integral gives
h(t)=1/sqrt(4pi t)+O(t^(-3/2)). For ell>=2, with
 gamma_ell=2-2cos(pi/ell), this yields

    J_ell(t)=1/2-ell/sqrt(4pi t)
               +O_ell(t^(-3/2))+O_ell(exp(-gamma_ell t)).

For ell=1 the finite-path remainder is absent and exp(-4t) gives the
remaining exponential term. For every fixed finite n, however,
H_t(I_(ell,n)) tends to one: the two-cut graph has two components and
each singly cut graph and the original cycle have one. The finite
combined response also tends to one. The large-volume and long-time
limits therefore do not commute.

## 5. Reusing the same element for other spectral observables

The coherent local array determines every trace polynomial in Delta:
a polynomial of degree at most 2R is evaluated from T_R. One cached local
histogram therefore supplies a whole moment vector and many heat times,
with the chosen maximum time and accuracy determining the required
truncation. Rooted motif and ordinary walk observables can use the same
array; their dependence on cut separation need not match (7).

This example also has a finite signed spectral measure even though its
measure on rooted graphs cannot have finite variation. Let

    rho(dx) = dx/[pi sqrt(x(4-x))],  0<x<4.

This is the line's Laplacian spectral law, obtained by changing variables
x=2-2cos theta in its Fourier integral. The cut and interaction spectral
measures are

    nu_E = (delta_0-delta_4)/2,
    nu_I = delta_0/2
             + sum_{a=1}^{ell-1}delta_(2-2cos(pi a/ell))
             + delta_4/2 - ell rho.               (10)

These identities follow by matching the polynomial moments, or directly
from the previously proved heat identities and their power series.
The positive atomic part of nu_I has mass ell and its negative
absolutely continuous part has mass ell, so ||nu_I||_TV=2ell.
For every continuous scalar f on [0,4], the relative spectral observable

    R_f(I_ell)=integral f(x) dnu_I(x)

is consequently defined. Polynomial approximation identifies it from
the same local moments with error at most
2ell * ||f-p||_infinity. These claims concern this explicit family and
its proved spectral representation; an arbitrary completed graph
number need not supply such a measure or bound.

For example, the resolvent interaction at s>0 is

    R_s = sum_{a=0}^{ell-1} 1/[s+2-2cos(pi a/ell)]
            -ell/sqrt(s(s+4)) -2/[s(s+4)].         (11)

Put r_s=(s+2-sqrt(s(s+4)))/2, so 0<r_s<1. The line resolvent kernel at
integer displacement a is r_s^|a|/sqrt(s(s+4)): substitution into
(s+2)g_a-g_(a-1)-g_(a+1)=delta_(a,0) verifies the formula. Summing its
images at multiples of 2ell, or integrating (5) against exp(-st), gives

    R_s = [2ell/sqrt(s(s+4))]
                  r_s^(2ell)/(1-r_s^(2ell)).      (12)

It is positive and decreases exponentially in separation for fixed s.
The expression holds at ell=1 by direct substitution into (11), avoiding
any appeal to a nonexistent simple C_2.

Multiplication by a finite normalized graph or a normalized lattice
factor preserves the earlier Cartesian heat factorization. For more
general spectral tests the Laplacian spectral laws convolve under
Cartesian multiplication. Thus (10) can also support several different
observables of the same defect with a retained product structure.

## 6. Higher inclusion-exclusion on the line

There is a special simplification for any k>=2 cuts at ordered positions
x_1<...<x_k on the line, where x_i denotes the vertex immediately to the
right of the cut and adjacent cuts have positive integer gaps. For a
nonempty selected subset S, the combined completed response is

    D_S = E + sum_(adjacent a<b in S)
                    [P_(x_b-x_a)-(x_b-x_a)L].

The empty response is zero. Its irreducible k-cut interaction is the
alternating subset sum sum_S (-1)^(k-|S|) D_S. All terms for adjacent
selected pairs cancel unless the pair consists of the two extreme
cuts. To see this, fix a pair a<b: its contribution requires a,b selected,
all indices between them omitted, and any subset of the indices outside
[a,b]. The alternating sum over an exterior index is zero; only a=1,
b=k survives. The E coefficient is (-1)^(k+1). Hence the exact graph
identity is

    I_(x_1,...,x_k)=(-1)^k I_(x_k-x_1).            (13)

The positions may be arbitrary distinct integers, including negative
ones. Translating all of them by a common integer changes no completed
element. For a finite-cycle representative, subtract min(x_i), place the
first cut at the wrap edge {n-1,0}, and place each later cut x at {x-1,x};
n>=span+2r+2 is a sufficient size for radius-r stabilization. Empty cut
sets give zero and singleton cut sets give E. Duplicate coordinates are
excluded: treating coincident labelled defects as distinct changes their
inclusion-exclusion and does not satisfy (13).

This holds before choosing an observable. It reduces 2^k formal subset
terms to one interaction element and gives a budget at most four via
that representation. The direct finite subset construction has the
larger budget k*2^(k-1). This simplification is specific to ordered cuts
of the line and their completed limits. It proves the corresponding
alternating heat sign and outer-separation dependence in this family.

## 7. Provenance and discriminating checks

Inclusion-exclusion of heat traces and the interpretation in terms of
paths that encounter every object have substantial prior literature.
For example M. Schaden, "Irreducible Many-Body Casimir Energies of
Intersecting Objects", https://arxiv.org/pdf/1011.2475, Eq. (5), uses the
alternating subset heat trace for continuum objects. Its positivity and
boundary assumptions differ from graph edge removal; its sign results
are not invoked here. Fourier cycle spectra, image sums, resolvents,
and finite-rank perturbation techniques are also classical. The present
work specializes and integrates them with coherent signed graph-number
limits, quantitative metadata, and reuse across local observables. It
does not claim invention of defect subtraction or matrix-function updates.

Useful checks are:

- Compare the two-cut finite expression with (1) exactly at several
  radii, including the threshold n=max(ell+2r,2r+2).
- Check every zero marginal in (2), the first nonzero marginal, and
  variation 4r for r>=ell. Include ell=1 and ell=2 explicitly.
- Check the moment sequence (6), its first nonzero order, and the generic
  rational relative-heat interval against independent high-precision
  evaluations of (3) or the positive series (5).
- At large ell and small t use (5) rather than subtracting nearly equal
  quantities in (3); apparent numerical zero is not proof of zero.
- Compare resolvent (11) with the independently derived positive closed
  form (12). Both use the same graph element as the heat calculation.
- Test (13) against explicit subset sums for k=3 and k=4, including
  unequal gaps. Do this at the graph-histogram level, not only for heat.
- Keep full local data reusable across observables and time parameters;
  compare timings with conventional cached moments and low-rank matrix
  methods that enjoy the same reuse. Reuse alone does not establish an
  advantage specific to graph numbers.

The implementation is `graphnumbers-local` 0.3.0. `TwoCutLineDefect`,
`CutInteraction`, and `LineCutDefect` supply the elements above with edit
metadata. `connected_cut_interaction` applies (13) without enumerating
subsets. `PreparedLocal` caches exact geometry at a declared radius;
`PreparedRelativeHeat` caches a single return-moment vector and uses
prefixes for every requested time up to its preparation horizon. Each
prepared heat query retains a rational interval and the original local
error bound. A larger prepared radius supplies all shorter moment orders
by the same locality proof; it need not be rebuilt for each time.

The generic step budget selected for a maximum time also suffices for
smaller positive times. For fixed M write the rational radius bound as
beta*(1+3a/2)/(s+a), where s=S/term_M>=1 and a=T/term_M use the notation
of the sparse-defect note. As time increases, beta and a increase while
s decreases; the displayed ratio is increasing in a and decreasing in s.
The condition M+2>tD is also preserved when time decreases. Zero time
is handled exactly. Queries may therefore choose smaller prefixes of the
prepared vector while keeping the same local approximation error bound.

The [interaction tests](../../python/tests/test_prepared_interactions.py)
check the formulas against explicit finite edits and subset sums through
five cuts, including negative translations. They also check the exact
binomial moments, positive high-precision heat series, growing variation,
and prepared queries with deliberately inexact source data. The
[geometry note](GEOMETRY_VERSUS_SPECTRUM.md) supplies an independent
example of information invisible to scalar spectral traces.
