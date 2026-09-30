# Planar defects, crossing cuts, and controlled moment profiles

30 September 2026. All Laplacians are the combinatorial Laplacians with
unit rate on each present edge. Finite graph coefficients and relative
traces are unnormalized. Write A for the signed local completion, T_r
for its radius-r marginal, and E=lim(P_n-C_n) for the cut-line element
from [the sparse-defect note](SPARSE_DEFECTS_AND_RELATIVE_HEAT.md).

There are two different planar constructions below. Finitely many
deleted edges in the square lattice give ordinary finite-rank relative
operators. Complete coordinate cuts remove infinitely many edges, but
their mixed correction admits Cartesian factorization. The latter
provides a compositional certificate unavailable from the direct
first-order edit counts of its finite approximants.

## 1. Finite edge defects on the square lattice

Let L be the Laplacian on Z^2. For distinct edges with oriented incidence
vectors b,c, put B=bb^T and C=cc^T. Deleting both edges gives L-B-C.
The relative graph interaction and its polynomial moments are

\[
 X_{b,c}=[G\setminus\{b,c\}]-[G\setminus b]
          -[G\setminus c]+[G],
\qquad
 I_j=\operatorname{Tr}\big((L-B-C)^j-(L-B)^j-(L-C)^j+L^j\big).
\]

The graph expression on the infinite lattice means the locally stable
limit of the same expression on finite exhaustions. A radius-r change
can occur only near the four edited endpoints, so sufficiently large
exhaustions give exactly the same marginal. Degree is at most four.
The direct decomposition into relative edge edits gives edit budget
at most 2+1+1=4. Each operator difference is finite rank for polynomial
functions and trace class for heat, by the usual telescoping and
Duhamel formulas. Thus these moments and relative heat agree with the
local construction and its first-order certificate.

Expanding the noncommuting powers gives, with s=b^Tc and u=b^TLc,

\[
 I_0=I_1=0,\qquad I_2=2s^2,\qquad I_3=6su-12s^2.                 \tag{1}
\]

For a useful fourth-order check, set v=b^TL^2c, alpha=b^TLb and
beta=c^TLc. Since b^Tb=c^Tc=2,

\[
 I_4=8sv+4u^2-32su-4s^2(\mathrm{alpha}+\mathrm{beta})
                  +48s^2+2s^4.                                \tag{2}
\]

These formulas do not assume that B,C or L commute. For example the
terms with one B, one C and two L's give 8sv+4u^2; the remaining mixed
words give the other terms in (2).

Fix b=delta_(0,0)-delta_(1,0). The following table gives exact integer
moments for five choices of c. The endpoint order specifies its
orientation; changing that order leaves the interaction unchanged.

| Second edge, in endpoint order | I_2 | I_3 | I_4 | I_5 | I_6 | First heat term |
|---|---:|---:|---:|---:|---:|---|
| (0,0)--(0,1), perpendicular adjacent | 2 | 24 | 226 | 1980 | 16826 | t^2 |
| (1,0)--(2,0), collinear adjacent | 2 | 24 | 218 | 1840 | 15212 | t^2 |
| (0,1)--(1,1), opposite sides of one square | 0 | 0 | 16 | 320 | 4176 | (2/3)t^4 |
| (0,2)--(1,2), parallel at distance two | 0 | 0 | 0 | 0 | 24 | t^6/30 |
| (3,0)--(4,0), separated collinear | 0 | 0 | 0 | 0 | 6 | t^6/120 |

Here heat means sum_j (-t)^j I_j/j!. For both adjacent cases alpha=beta=10
and su=6. The values sv are respectively 38 and 37, proving their
fourth-order distinction directly from (2). Their heat interactions
differ by t^4/3+O(t^5). The displayed moments were also checked by two
independent exact integer calculations: sparse recurrence on a 12 by
12 torus and dense matrix powers on a 10 by 10 torus, agreeing through
degree eight. These sizes cannot introduce a mixed closed walk of
length at most eight that winds around the torus.

There is a general explanation of the first nonzero term. Suppose

\[
 m=\min\{a\ge0:b^TL^ac\ne0\}.
\]

Then

\[
 I_j=0\quad(j<2m+2),\qquad
 I_{2m+2}=(2m+2)(b^TL^mc)^2.                                  \tag{3}
\]

Indeed, inclusion-exclusion removes every word missing B or C. In a
cyclic trace of a remaining word, there are at least two transitions
between B and C. Rank-one multiplication makes each such transition
contain a factor b^TL^ac. A nonzero transition therefore needs at least
m intervening L's. Minimal total length is 2m+2, attained only by the
cyclic rotations of B L^m C L^m. There are 2m+2 such words and their
traces are all (b^TL^mc)^2. The two negative perturbations give a
positive sign. Consequently the leading heat term is

\[
 \frac{(b^TL^mc)^2}{(2m+1)!}\,t^{2m+2}.                       \tag{4}
\]

For the five rows, (m,b^TL^mc) is respectively
(0,1), (0,-1), (1,-2), (2,2), (2,-1). If every cross moment is zero,
the same word argument makes every mixed polynomial trace zero.
Equation (4) establishes the small-time sign when a first cross moment
exists; it does not assert a sign for every time or every higher-order
interaction.

The bridge reduction in [the branching note](BRANCHING_DEFECTS.md)
does not apply to these lattice edges: they lie on cycles. Neither
their heat traces nor their full local interactions generally factor
into independent one-dimensional cuts.

## 2. Complete crossing cuts are the product E squared

On a finite rectangular torus C_n square C_m, remove every seam edge
in either coordinate. The four-term mixed correction is exactly

\[
 (P_n-C_n)(P_m-C_m)
 =P_n\square P_m-P_n\square C_m-C_n\square P_m+C_n\square C_m.
                                                                    \tag{5}
\]

One seam removes m edges and the other removes n. The direct edit
budgets therefore diverge as the rectangle grows. Nevertheless the
product converges in A to E^2. At fixed radius, cancellation removes
roots that see only one seam, leaving the crossing region. More
generally independent coordinate cuts in k dimensions give E^k as
the limit of a 2^k-term inclusion-exclusion expression.

For every r>=1, the precise marginal variation is

\[
 \|T_r(E^k)\|_1=(4r)^k,\qquad
 |\operatorname{supp}T_r(E^k)|=\binom{r+k}{k}.                  \tag{6}
\]

In particular E^2 has variation 16r^2 and is not representable by a
finite signed measure on rooted graphs. The exact variation for k=2
was checked at r=1,2,3,4, giving 16,64,144,256 and respectively
3,6,10,15 rooted types.

To prove (6), let B_(r,a) be the r-ball in a half-line whose root is
at distance a from its endpoint, and let B_(r,infinity) be the line
ball. The one-dimensional marginal is

\[
 T_rE=2\sum_{a=0}^{r-1}\delta_{B_{r,a}}
                      -2r\delta_{B_{r,\infty}}.               \tag{7}
\]

We must check that multiplying (7) cannot merge opposite signs. The
untruncated sphere generating series of these rooted factors are

\[
 F_\infty(z)=\frac{1+z}{1-z},\qquad
 F_a(z)=\frac{1+z-z^{a+1}}{1-z}.
\]

Cartesian distance adds. If n_a factors have finite endpoint distance
a and n_infinity factors are lines, the sphere series, divided by
F_infinity(z)^k and read modulo z^(r+1), is

\[
 \prod_{a=0}^{r-1}
       \left(1-\frac{z^{a+1}}{1+z}\right)^{n_a}.               \tag{8}
\]

In its formal logarithm, the coefficient of z^(a+1) is -n_a plus an
expression in n_0,...,n_(a-1). Thus the rooted ball's sphere counts
recover every n_a, and then n_infinity=k-sum_a n_a. Different
multisets cannot be isomorphic; permutations of factors are isomorphic.
The coefficient of the type belonging to a multiplicity vector is

\[
 2^k(-r)^{n_\infty}
       \frac{k!}{n_\infty!\prod_a n_a!}.                     \tag{9}
\]

Every such vector has a nonzero coefficient, and its sign is fixed.
Counting the vectors and summing the absolute coefficients proves
(6). At r=1 this gives the particularly visible planar identity

\[
 T_1(E^2)=4\delta_{K_{1,2}}-8\delta_{K_{1,3}}
                                      +4\delta_{K_{1,4}},       \tag{10}
\]

with all stars rooted at their centers. For a seminorm weighted by
the s-th power of ball size, the elementary size bound (2r+1)^k also
gives p_(r,s)(E^k)<=(4r)^k(2r+1)^(ks). Radius zero gives zero for all
k>=1.

## 3. The mixed infinite-volume heat operator is trace class

On l^2(Z), let A_0 be the line Laplacian and A_1 its one-edge-cut
Laplacian. For t>=0 put K_t=exp(-tA_1)-exp(-tA_0). Duhamel gives

\[
 K_t=\int_0^t e^{-(t-s)A_1}(A_0-A_1)e^{-sA_0}\,ds,
 \qquad \|K_t\|_1\le 2t.                                    \tag{11}
\]

The integrand is trace class because A_0-A_1 is one edge Laplacian,
with trace norm two, and both semigroups are contractions. The earlier
cut-line calculation gives Tr K_t=(1-exp(-4t))/2.

On l^2(Z^k), the Laplacian after a set S of complete coordinate cuts
is the Kronecker sum of A_1 in coordinates in S and A_0 elsewhere.
Factorization of the commuting coordinate semigroups yields

\[
 \sum_{S\subseteq[k]}(-1)^{k-|S|}e^{-tL_S}=K_t^{\otimes k}.
                                                                    \tag{12}
\]

The sum is trace class, with trace norm at most (2t)^k, and hence

\[
 H_t(E^k)=\left(\frac{1-e^{-4t}}2\right)^k.                    \tag{13}
\]

Individual single-coordinate relative heat operators in dimension
at least two need not be trace class: they contain an unchanged
infinite-lattice heat factor. In (12) the alternating operator sum is
formed before taking its trace. There is no subtraction of separately
defined infinite scalar traces.

For k=2, (13) starts 4t^2-16t^3+O(t^4); for general k its leading
term is 2^k t^k. Its long-time limit is 2^(-k). On each finite torus
approximant the corresponding mixed heat trace instead tends to zero,
so large-volume and long-time limits do not commute. These exact heat
factorizations are standard consequences of Cartesian products; the
additional graph information is the full signed geometry in (6)-(10).

## 4. Finite moment profiles give a compositional controlled domain

Fix a degree cap D>0 and set P_D=I-Delta/D. A finite nonnegative
profile (A_0,...,A_s) certifies X when its local trace moments obey

\[
 |d_j(X;D)|\le \sum_{r=0}^s A_r\frac{(j)_r}{D^r}
 \quad(j\ge0),                                                 \tag{14}
\]

where (j)_r=j(j-1)...(j-r+1), (j)_0=1, and (j)_r=0 for r>j.
The profile is required to hold for every degree parameter at least
the certified cap. A bound proved at one positive cap d already has
this property: P_D=(1-d/D)I+(d/D)P_d, and binomial averaging sends
(j)_r/d^r to (j)_r/D^r.

A finite-variation element of variation C has profile (C). A relative
edit element with budget q has profile (0,2q), by the earlier trace
norm telescoping bound. Absolute scalar multiples scale profiles;
sums add them. Products convolve their coefficients:

\[
 (A*B)_r=\sum_{a+b=r}A_aB_b.                                  \tag{15}
\]

Indeed, for caps D_1,D_2>0, Cartesian uniformization has total cap
D=D_1+D_2 and P_D=(D_1/D)P_1+(D_2/D)P_2 in the two coordinates.
Expand its j-th power and use the binomial factorial-moment identity

\[
 \sum_m\binom jm w_1^m w_2^{j-m}(m)_a(j-m)_b
                 =(j)_{a+b}w_1^aw_2^b.
\]

It cancels the factor caps in (14), proving (15). Zero-degree factors
are scalar multiples of the isolated-vertex unit and are handled
directly, without division by a zero cap.

In particular the product of k edit elements of budgets q_i has pure
order-k profile A_k=2^k product_i q_i. Its moments satisfy

\[
 |d_j|\le\frac{2^k\prod_iq_i}{D^k}(j)_k.                      \tag{16}
\]

This bound also holds for an irreducible interaction of k finite
groups of deleted edges on a single graph, without any Cartesian
assumption. Require the edge groups to be disjoint, and let B_i be
their edge Laplacian sums, with q_i edges. Interpolate by
P(s)=I-[L-sum_i s_i B_i]/D for s in [0,1]^k. These are weighted-graph
contractions at the same cap. The mixed finite difference of P^j is
the integral of its mixed derivative. That derivative has (j)_k
ordered terms, each containing one B_i/D. The trace norm of each
term is at most 2^k product_i q_i/D^k, using
||B_i||_1=2q_i and ||B_i||<=2q_i. This proves (16), including for
infinite bounded-degree graphs with finitely supported edge groups.
This argument cannot directly certify complete coordinate planes,
whose q_i are infinite; their product representation supplies the
finite profile instead.

The same derivative proof permits a valid mixture of edge insertions
and deletions. Each group then has a signed B_i, with
||B_i||_1<=2q_i. Require every subset edit to give a valid simple graph
and use a cap covering the graph with all insertions and no deletions.
The multilinear interpolation has nonnegative edge weights and degree
at most that cap throughout its parameter cube, so every P(s) remains
a contraction. No positivity of B_i is needed in the trace-norm bound.

With lambda=tD, (14) makes the uniformized heat series absolutely
convergent and bounds its truncation after degree M by

\[
 \left|e^{-\lambda}\sum_{j>M}\frac{\lambda^j}{j!}d_j\right|
 \le\sum_r A_rt^r
        \Pr\{\operatorname{Poisson}(\lambda)\ge M+1-r\}.      \tag{17}
\]

A Poisson threshold at most zero means probability one. For a pure
order-k profile this is 2^k(product q_i)t^k times the Poisson tail
at M+1-k. At k=2 for crossing cuts, D=4, so (16) reads
|d_j|<=j(j-1)/4 and the remainder is at most
4t^2 Pr{Poisson(4t)>=M-1}.

Uniform bounds of this kind justify taking locally convergent limits
of controlled approximants: every retained moment converges by local
continuity, and (17) controls the omitted terms uniformly. They also
define heat directly from a certified element's local moments.
Linearity and Cartesian multiplicativity follow from absolutely
convergent series. Consequently the union over finite degree caps and
finite profiles is a unital subalgebra of A containing every finite
graph combination and all the products certified above. Heat is a
linear, multiplicative functional on this subalgebra. Changing to a
larger degree parameter leaves its value unchanged by the binomial
uniformization identity and absolute convergence.

For a fixed cap D, fixed profile and fixed t, heat is uniformly
continuous in the inherited local topology. Choose a common M to
make the two profile tails small; the retained difference is bounded
by ||T_R(X-Y)||_1, where R=ceil(M/2), because rooted uniformized return
probabilities lie in [0,1] and the retained Poisson weights sum to at
most one. The defining weighted seminorm p_(R,1) also bounds it.
Allowing profiles to grow without bound loses this uniform estimate.
For t>0 there is no continuous linear extension agreeing with finite
graph heat on all of A, as proved in
[the effective-analysis note](EFFECTIVE_ANALYSIS_AND_HEAT.md).

### Rational enclosure without evaluating an exponential

Let S=S_M(lambda)=sum_(j=0)^M lambda^j/j!, and let T be any rational
upper bound for exp(lambda)-S. Put S_a=0 for a<0, and

\[
 G(t)=\sum_r A_rt^r,\quad
 C_M=\sum_r A_rt^rS_{M-r},\quad
 B_M=\sum_r A_rt^r\frac{S-S_{M-r}+T}{S+T}.                    \tag{18}
\]

The retained numerator N_M has absolute value at most C_M<=G(t)S.
Each tail probability in (17) is 1-S_(M-r)/exp(lambda), which grows
with exp(lambda). Substituting S+T therefore proves the tail bound
B_M, also when r>M. The reciprocal exponential belongs to
[1/(S+T),1/S]. If local data give numerator error at most S epsilon,
intersect its numerator interval with [-C_M,C_M], multiply intervals
by that reciprocal interval, and enlarge by [-B_M,B_M]. The resulting
enclosure has radius at most

\[
 B_M+\frac{G(t)T}{2(S+T)}+\mathrm{epsilon}.                    \tag{19}
\]

The second term bounds denominator uncertainty; the last bounds
local approximation. Empty numerator intersections flag an
inconsistent supplied certificate. Degree zero is treated separately:
Delta=0, so heat is exactly the vertex-mass functional. At time zero
the same observation avoids any truncation. Profiles are supplied
analytic guarantees, not a property inferred merely from finitely
many computed moments.

### Sharper special information for E powers

For the cut line at cap two, exp(2t)H_t(E)=sinh(2t), so d_j(E) is one
at odd j and zero at even j. Cartesian multiplication shows that
d_j(E^k;2k) is the probability that every category has odd occupancy
in j independent uniform draws from k categories:

\[
 d_j(E^k;2k)=k^{-j}
   \sum_{\substack{m_1+\cdots+m_k=j\\m_i\text{ odd}}}
      \binom{j}{m_1,\ldots,m_k}.                              \tag{20}
\]

Thus these moments lie in [0,1], vanish below k and when j and k have
different parity. For k=2 they are one half at positive even j and
zero otherwise. This gives a stronger special bound than (16).
Equivalently the signed Laplacian spectral measure is

\[
 \nu_{E^k}=2^{-k}\sum_{a=0}^k(-1)^a\binom ka\delta_{4a},
 \qquad\|\nu_{E^k}\|_{\rm TV}=1.                             \tag{21}
\]

It provides the alternative scalar profile (1). Finite spectral
variation does not imply finite variation of the rooted graph measure,
as (6) demonstrates. Nor do the diverging direct seam-edit budgets
prove that E^k has no other uniformly bounded edit representation.
The proved gain of (15)-(17) is compositional certification from the
chosen factors, without requiring either such a representation or
the special spectral formula (21).

## 5. Implementation and reproducible checks

Library version 0.4.0 exposes `moment_profile(X)` and
`controlled_heat(X, time, epsilon)`. Built-in sums, absolute scalar
multiples and Cartesian products propagate profiles by (15).
`PreparedLocal` retains them. `EdgeInteraction` supplies the pure-order
bound (16), using the reduced number of selected edges when the bridge
identity applies. The older `relative_heat` retains its first-order
contract; `controlled_heat` now evaluates E squared directly from its
signed local geometry and profile (0,0,4).

The implementation follows (18)-(19) using rational arithmetic throughout.
Its certificate records the profile, truncation, exponential enclosure,
moment-tail bound, source approximation error and retained moments.
Zero time, zero-degree scalars and zero profiles are handled explicitly.
Custom profiles remain caller-supplied mathematical contracts.
If an inexact local approximation contains types exceeding the certified
degree cap, the implementation projects them out. The true marginal has
zero coefficients there, so this projection cannot increase the weighted
local error. This makes the enclosure valid for such noisy approximants
without imposing an extra support requirement on their oracle.

The [certificate example](../../python/examples/controlled_heat_examples.py)
records crossing cuts and finite star/triangle responses, with independent
high-precision closed-form checks. The
[exact geometry verifier](../../python/examples/branch_planar_verification.py)
checks the five planar pair rows on two buffered open square grids,
including (3), and records stabilized three- and four-edge interactions.
Tests also cover E squared and E cubed marginal variation/type counts,
exact crossing moments, profile propagation, deliberately inexact local
oracles, mixed edits, and the existing heat contracts.

## 6. Prior work and the comparison to make

Inclusion-exclusion of heat traces and relative matrix functions has
substantial prior literature. The following are primary research
sources; none is a claim that this particular local completion was
already constructed there.

- M. Schaden, *Irreducible Many-Body Casimir Energies of Intersecting
  Objects*, [arXiv:1011.2475](https://arxiv.org/pdf/1011.2475), equations
  (4)-(5), (14)-(16). This defines alternating heat-trace contributions
  and explains cancellation of paths missing any object. The setting
  uses continuum domains and positive local potentials or specified
  boundary conditions. Its sign and ultraviolet conclusions cannot
  simply be transferred to degree-adjusted graph edge deletion.
- K. V. Shajesh and M. Schaden, *Many-Body Contributions to Green's
  Functions and Casimir Energies*, Physical Review D 83, 125032 (2011),
  [doi:10.1103/PhysRevD.83.125032](https://doi.org/10.1103/PhysRevD.83.125032),
  [primary preprint](https://arxiv.org/pdf/1103.3048), equations (24)-(27).
  Irreducible Green-function terms and multiple scattering provide a
  conventional way to organize interactions among localized objects.
- B. Beckermann, D. Kressner and M. Schweitzer, *Low-rank updates of
  matrix functions*, [arXiv:1707.03045](https://arxiv.org/pdf/1707.03045).
  Finite-edge lattice approximants are exactly within the low-rank
  matrix-function-update setting. This is an appropriate computational
  baseline for their relative spectral observables.
- A. Cortinovis, D. Kressner and S. Massei, *Divide and conquer methods
  for functions of matrices with banded or hierarchical low-rank
  structure*, [arXiv:2107.04337](https://arxiv.org/pdf/2107.04337),
  Theorem 2. For symmetric matrices and a low-rank symmetric update,
  the shared block Krylov compression gives trace exactness for
  polynomials of degree at most twice the block Krylov step count.
  The existing comparator implements this strong conventional option.

For complete Cartesian cuts, a fair baseline must also use direct
tensor factorization: formula (13) is already an exact scalar answer.
For finite non-Cartesian defects, compare matched finite volumes,
unnormalized absolute errors, setup costs, and reuse across times or
observables against low-rank update methods. A direct local moment
or motif calculation is another ordinary baseline when only one
geometric test is needed.

The substantive conclusions here are structural: the completion
contains a nonzero crossing geometry with exactly quantified local
variation; mixed operator traces can survive infinite individual
cut sizes; and finite moment profiles carry a proof of heat control
through sums, products and locally convergent approximants. These
facts identify a usable domain and concrete planar tests. They do
not by themselves establish a computational speed advantage or a
new general principle of irreducible interactions.
