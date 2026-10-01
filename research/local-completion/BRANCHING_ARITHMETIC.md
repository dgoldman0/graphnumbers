# Single cuts in regular trees: geometry and completed arithmetic

30 September 2026. This note compares single-edge cuts in infinite
regular trees inside the established **local Cartesian completion**
$A=\mathcal A_{\mathrm{loc}}$. The degree-two case is the previous
cut-line element $E$. Complex spectra and holomorphic function algebras
refer to $A_{\mathbb C}$; real Taylor coefficients give the real versions.

Every regular-tree cut has the same one-generator entire-function
arithmetic as $E$. Branching changes its exact radius-dependent norms
and the radius at which an inverse obstruction becomes visible. Any
finite set of distinct regular-tree degrees supplies independent entire
variables. These statements concern actual elements and their embedding
in the local Cartesian completion; they do not identify its whole
algebra with a classical function algebra.

## 1. Constructing the branching defect

Fix an integer $d\ge2$, put $q=d-1$, and write

$$S_r(d)=\sum_{j=0}^{r-1}q^j
=\begin{cases}r,&d=2,\\(q^r-1)/(q-1),&d\ge3.
\end{cases}\tag{1}$$

Take a central edge and grow $q$ children from each of its endpoints,
then $q$ children from each subsequent vertex, through depth $L$ on
each side. Denote this finite tree by $G_{d,L}$ and its central edge by
$e$. Define

$$B_{d,L}=(G_{d,L}\setminus e)-G_{d,L}.\tag{2}$$

This is a same-vertex-set one-edge deletion, with maximum degree $d$
and absolute edit budget one. For a fixed radius $r$, roots at distance
$j<r$ from their nearest endpoint of $e$ are affected; there are
$2q^j$ of them. Every other rooted $r$-ball is unchanged. If $L\ge2r$,
the affected neighborhoods lie far enough from the artificial outer
boundary to have their infinite-tree types. Thus the signed local
histogram stabilizes exactly, including every polynomial weight, and

$$B_d=\lim_{L\to\infty}B_{d,L}\in A\tag{3}$$

exists. It has vertex mass zero, degree bound $d$, and edit budget one.
For $d=2$ its local histograms equal those of $E$, hence $B_2=E$.

Let $R_{d,r}$ be the rooted radius-$r$ ball of the $d$-regular tree.
Let $A^{(d)}_{r,j}$ be that rooted ball with one outward edge from depth
$j$ to depth $j+1$ deleted, retaining the root's component. For $r\ge1$,

$$\boxed{T_rB_d=2\sum_{j=0}^{r-1}q^j\delta_{A^{(d)}_{r,j}}
-2S_r(d)\delta_{R_{d,r}}.\tag{4}$$

The intact ball has size $b_{d,r}=1+dS_r(d)$, and the deleted branch
contains $S_{r-j}(d)$ vertices. Therefore

$$|A^{(d)}_{r,j}|=b_{d,r}-S_{r-j}(d),\qquad
\boxed{\|T_rB_d\|_1=4S_r(d)},\tag{5}$$
$$p_{r,k}(B_d)=2\sum_{j=0}^{r-1}q^j
\bigl(b_{d,r}-S_{r-j}(d)\bigr)^k
+2S_r(d)b_{d,r}^k.\tag{6}$$

The types in (4) are distinct: the intact type has no missing branch,
while the defect type has a uniquely deficient interior vertex at
distance $j$. Consequently none of these signed terms cancels.
The variation grows linearly with $r$ for $d=2$ and exponentially for
$d\ge3$, so every $B_d$ lies outside the finite signed graph-measure
model.

## 2. Additive coordinates for a fixed branching degree

Fix $r\ge2$. For an arbitrary rooted radius-$r$ ball $C$, write
$\delta(C)$ for root degree and $Q(C)$ for the number of simple
four-cycles through the root, including cycles with chords. As in the
[cut-line inversion proof](UNBOUNDED_VARIATION_INVERSION.md),

$$Q(C\star_r D)=Q(C)+Q(D)+\delta(C)\delta(D).\tag{7}$$

Hence

$$N_d(C)=\frac{(2d-1)\delta(C)-\delta(C)^2+2Q(C)}{d(d-1)}\tag{8}$$

is additive on the whole local monoid. On each tree atom in (4),
$Q=0$ and the root degree is either $d-1$ or $d$; thus $N_d=1$.
For arbitrary balls the values of $N_d$ may be negative or fractional.

Let $S_C(z)$ be the root-sphere polynomial through degree $r$. It
multiplies under the local Cartesian product modulo $z^{r+1}$.
Put

$$F_d(z)=\frac{1+z}{1-(d-1)z},\qquad
\ell_j(z)=\log\left(1-\frac{z^{j+1}}{1+z}\right)
\pmod {z^{r+1}}.\tag{9}$$

The intact and cut tree balls have sphere polynomials

$$S_{R_{d,r}}=F_d,\qquad
S_{A^{(d)}_{r,j}}=F_d\left(1-\frac{z^{j+1}}{1+z}\right)
\pmod {z^{r+1}}.\tag{10}$$

Indeed, at distance $k\ge j+1$ the deleted branch removes
$(d-1)^{k-j-1}$ vertices. The defect correction factor in (10) is
independent of $d$.

Since $\ell_j$ starts with $-z^{j+1}$, the triangular identity

$$\log S_C-N_d(C)\log F_d
=\sum_{j=0}^{r-1}c_j(C)\ell_j\pmod {z^{r+1}},
\qquad c_R=N_d-\sum_jc_j\tag{11}$$

defines additive rational-valued functions on the whole local monoid.
They take the values

$$c_i(A^{(d)}_{r,j})=\delta_{ij},\quad c_R(A^{(d)}_{r,j})=0,
\qquad c_i(R_{d,r})=0,\quad c_R(R_{d,r})=1.\tag{12}$$

Thus they recover the exponent tuple of every product of these atoms.
The phases $\exp(i\sum\theta_jc_j+i\theta_Rc_R)$ are bounded
semicharacters, regardless of the signs or denominators of their
coordinate values on other balls.

## 3. Exact local spectra and inverse obstructions

Assign every positive atom in (4) the same unit-modulus phase $u$ and
the intact atom a phase $v$. The associated continuous character takes
the value

$$\chi(B_d)=2S_r(d)(u-v).\tag{13}$$

The difference of two unit-circle points fills the disk of radius two.
The upper bound is the local variation in (5), so

$$\boxed{\sigma_{\ell^1(M_r)}(T_rB_d)
=\{\lambda:|\lambda|\le4S_r(d)\}}\qquad(r\ge2).\tag{14}$$

As $r$ grows these disks cover the plane. Hence

$$\sigma_{A_{\mathbb C}}(B_d)=\mathbb C,\qquad
1-tB_d\text{ is a nonunit whenever }t\ne0.\tag{15}$$

There is also an exact local reciprocal criterion. Distinct powers
have disjoint supports because $N_d$ equals the power index; within
each power the coordinates (12) prevent coefficient cancellation.
Therefore

$$\left\|T_r\left(\sum_na_nB_d^n\right)\right\|_1
=\sum_n|a_n|\bigl(4S_r(d)\bigr)^n\tag{16}$$

for every finite coefficient sequence. The unique formal reciprocal
of $1-tT_rB_d$ is the geometric series in $tT_rB_d$. Its local
absolute coefficient sum converges exactly when
$4S_r(d)|t|<1$, and then equals

$$\frac1{1-4S_r(d)|t|}.\tag{17}$$

For $d\ge3$, a sufficient and sharp integer-radius obstruction is
$r\ge2$ with

$$q^r\ge1+\frac{q-1}{4|t|}.\tag{18}$$

Thus the global affine inverse always fails, while the resolution
needed to expose that failure grows logarithmically in $1/|t|$ for a
branching tree. For the line the corresponding threshold grows
linearly in $1/|t|$.

## 4. Entire functions and units of the branching defect

A product of $n$ atoms in (4) has degree at most $dn$. Therefore

$$p_{r,k}(B_d^n)\le(r+1)^k(1+dn)^{rk}
\bigl(4S_r(d)\bigr)^n.\tag{19}$$

Equations (16) and (19) identify the closed complex algebra generated
by $B_d$ with $\mathcal O(\mathbb C)$, exactly as for $E$: polynomial
factors in $n$ are absorbed by increasing the exponential coefficient
radius, and $4S_r(d)$ tends to infinity. For every entire
$f(z)=\sum_na_nz^n$,

$$\|T_rf(B_d)\|_1=\sum_n|a_n|\bigl(4S_r(d)\bigr)^n
\qquad(r\ge2).\tag{20}$$

The explicit characters and the entire reciprocal give

$$f(B_d)\in A_{\mathbb C}^{\times}
\quad\Longleftrightarrow\quad f\text{ is zero-free on }\mathbb C,
\qquad \sigma_{A_{\mathbb C}}(f(B_d))=f(\mathbb C).\tag{21}$$

Ambient divisibility is equally exact: for entire $f\not\equiv0,g$,
$f(B_d)$ divides $g(B_d)$ in $A_{\mathbb C}$ precisely when $g/f$
extends to an entire function. To see the necessity, evaluate a proposed
quotient at the characters filling the disk in (14). The quotient
$g/f$ is uniformly bounded there by one local total variation norm,
so its apparent poles are removable. Domain uniqueness identifies the
ambient quotient with $(g/f)(B_d)$.

In particular,

$$\exp(tB_d)^{-1}=\exp(-tB_d),\qquad
\boxed{\|T_r\exp(tB_d)\|_1=
\exp\bigl(4|t|S_r(d)\bigr)}.\tag{22}$$

For $t\ne0$, both inverse partners have unbounded local variation.
This growth is exponential in the radius for the line and exponential
in $q^r$ for a branching tree. Every nonconstant entire function of
$B_d$ has unbounded local variation by (20); the subalgebra intersects
the finite signed graph-measure model only in scalars.

## 5. Recovering rooted tree factors from a product ball

The preceding sphere calculation treats one fixed degree. It does not
by itself prove joint independence across degrees: sphere products can
lose which branching degree belongs to which missing-branch distance.
The full rooted geometry supplies that association.

**Tree-factor lemma.** Fix $r\ge2$. A rooted radius-$r$ ball of a
Cartesian product of nontrivial rooted trees determines the multiset
of the factors' rooted radius-$r$ balls. Consequently distinct
multisets of nontrivial rooted tree balls have distinct local products.

**Proof.** At the product root, two different incident edges belong to
the same tree factor exactly when they lie in no common four-cycle.
Edges in different coordinates form a Cartesian square visible inside
radius two; a single tree factor contains no four-cycle. Thus the
neighbor set is partitioned into its coordinate blocks.

Starting with one such block, follow a root path outward. At a vertex
of depth $k<r$, a candidate next edge continues in the same tree
coordinate exactly when it and the preceding edge lie in no common
four-cycle. If it changes another coordinate, the Cartesian square has
its fourth vertex at depth $k$, while the candidate vertex has depth
$k+1\le r$, so the whole square remains visible. In the same tree
coordinate no such square exists. The preceding vertex is excluded
from the next step, so these are nonbacktracking tree paths and their
distance from the root increases at each step.

The union of these paths through depth $r$, for all starting neighbors
in the block, is precisely the factor's rooted $r$-ball. Repeat for
each root block. This construction is invariant under rooted
isomorphism and recovers the required multiset.

For the atoms of (4), the degree $d$ is itself recoverable at radius
$r\ge2$: it is the maximum degree among vertices at distance less
than $r$. The intact atom has no deficient interior vertex; a cut atom
has exactly one, at its specified distance $j$. Thus atoms belonging
to different degrees, as well as different $j$ within a degree, are
pairwise distinct.

## 6. Joint arithmetic of several branching degrees

Fix distinct $d_1,\ldots,d_m\ge2$. They may include $d_1=2$, so the
previous line defect is part of this joint result. Write
$B^\alpha=\prod_iB_{d_i}^{\alpha_i}$ and
$R_i(r)=4S_r(d_i)$.

By the tree-factor lemma every atom exponent in every mixed product
is recoverable. Different $\alpha$ have disjoint supports, and signs
within one expansion cannot cancel. Hence

$$\boxed{\left\|T_r\left(\sum_\alpha a_\alpha B^\alpha\right)
\right\|_1=\sum_\alpha|a_\alpha|\prod_iR_i(r)^{\alpha_i}}
\qquad(r\ge2).\tag{23}$$

Products in the $\alpha$ term have degree at most
$\sum_i d_i\alpha_i\le d_{\max}|\alpha|$, giving the weighted upper
bound

$$p_{r,k}\left(\sum_\alpha a_\alpha B^\alpha\right)
\le(r+1)^k\sum_\alpha|a_\alpha|
(1+d_{\max}|\alpha|)^{rk}\prod_iR_i(r)^{\alpha_i}.\tag{24}$$

### Independent characters on the full ambient algebra

To make the joint unit conclusion ambient, independent phases must be
defined on the whole local monoid. The established separating additive
invariants $K_F$ from the
[multiplication theorem](MULTIPLICATION_AND_UNITS.md) provide them.

For the finite collection of atoms at this fixed radius, their vectors
$(K_F(C))_F$ are linearly independent over $\mathbb Q$. Otherwise,
clearing denominators and separating positive and negative coefficients
would give two different atom products with equal values of every
$K_F$. Separation would identify their rooted balls, contradicting the
tree-factor lemma. Finitely many $K_F$ coordinates therefore already
give a matrix of full column rank. Rational linear algebra supplies
additive functions $h_C$ taking value one on the selected atom $C$ and
zero on every other selected atom.

This is an effective finite-coordinate construction in principle:
enumerate the known rooted patterns until the evaluation matrix reaches
the required rank, then solve the rational linear system. Termination
follows from the proved independence; no bound on its computational
cost is asserted.

The phases $\exp(i\sum_C\theta_Ch_C)$ are bounded semicharacters of
the full local monoid. Assigning common phases to each degree's positive
atoms and separate phases to its intact atom shows that the joint
character image of $(B_{d_1},\ldots,B_{d_m})$ at radius $r$ is exactly

$$\prod_{i=1}^m\{z_i:|z_i|\le R_i(r)\}.\tag{25}$$

The opposite containment follows from each coordinate's total variation
bound. As $r$ increases these polydisks cover $\mathbb C^m$.

### The resulting function algebra and its units

The coefficient norms (23), their weighted estimates (24), and the
growth of every $R_i(r)$ give

$$\boxed{\overline{\mathbb C[B_{d_1},\ldots,B_{d_m}]}
\cong\mathcal O(\mathbb C^m)}\tag{26}$$

with the usual compact-open topology. To check the topology directly,
each compact polydisk is dominated by some box in (25). Conversely,
Cauchy's multivariate coefficient bound on a larger polydisk controls
the weighted coefficient sum on a smaller one. Polynomial factors in
$|\alpha|$ are absorbed by enlarging all radii. Completeness and
coefficientwise convergence then identify the closed generated algebra.

For every entire $f$ of $m$ variables,

$$f(B_{d_1},\ldots,B_{d_m})\in A_{\mathbb C}^{\times}
\quad\Longleftrightarrow\quad f(z)\ne0\text{ for every }z\in\mathbb C^m,
\tag{27}$$
$$\sigma_{A_{\mathbb C}}(f(B_{d_1},\ldots,B_{d_m}))
=f(\mathbb C^m).\tag{28}$$

A zero is detected by (25); a zero-free entire function has an entire
reciprocal, which gives its inverse in (26). Applying that equivalence
to $\lambda-f$ proves (28). These arguments work over the real algebra
for real-coefficient functions and inverses.

For example, mixed exponential units have the exact variation

$$\left\|T_r\exp\left(\sum_it_iB_{d_i}\right)\right\|_1
=\exp\left(\sum_i4|t_i|S_r(d_i)\right),\tag{29}$$

and inverse obtained by negating all $t_i$. The mixed coordinates are
algebraically independent, not merely distinct individual elements.

## 7. A degree-matched comparison with the square lattice

Let $P$ denote the single-edge-cut defect of the square lattice, as
defined in [the planar comparison](PLANAR_ARITHMETIC.md). Both $B_4$
and $P$ have degree bound four, vertex mass zero, and edit budget one.
Their first three relative Laplacian trace moments agree, while the
fourth detects the squares through the deleted edge.

More generally, consider a deleted edge $e$ in a triangle-free
$d$-regular medium, and let $c_e$ count the four-cycles containing that
edge. Write $L$ for the original combinatorial Laplacian and
$b=e_u-e_v$ for its incidence vector. The new Laplacian is $L-bb^T$.
All computations can be made in a sufficiently buffered finite graph;
the resulting local moments stabilize in the completion.

Set $a_k=b^TL^kb$. The first four values are

$$a_0=2,\quad a_1=2d+2,\quad a_2=2d^2+6d,\quad
a_3=2d^3+12d^2+4d-2+2c_e.\tag{30}$$

To obtain the last value, the number of length-three adjacency walks
from $u$ to $v$ is $2d-1+c_e$: $2d-1$ walks backtrack, and each square
supplies its other three-edge route. Triangle-freeness gives zero
diagonal third adjacency moments. Expanding $(dI-A)^k$ now gives
(30).

The formal determinant identity and trace logarithm give

$$\log\frac{\det(I-z(L-bb^T))}{\det(I-zL)}
=\log\left(1+\sum_{k\ge0}a_kz^{k+1}\right)
=-\sum_{m\ge1}\frac{\Delta_m}{m}z^m,\tag{31}$$

where $\Delta_m=\operatorname{tr}((L-bb^T)^m-L^m)$. Thus

$$\begin{aligned}
\Delta_0&=0,&\Delta_1&=-2,&\Delta_2&=-4d,\\
\Delta_3&=-6d^2-6d+4,&
\Delta_4&=-8d^3-24d^2+16d-8c_e.
\end{aligned}\tag{32}$$

For example, the fourth coefficient follows directly from
$\Delta_4=-4(a_3-a_0a_2-a_1^2/2+a_0^2a_1-a_0^4/4)$.
A tree edge has $c_e=0$, whereas a square-lattice edge belongs to two
squares. Consequently the moment lists through order four are

$$B_4:\ (0,-2,-16,-116,-832),\qquad
P:\ (0,-2,-16,-116,-848).\tag{33}$$

The controlled relative heat series therefore satisfies

$$H_t(B_4)-H_t(P)=\frac23t^4+O(t^5)\qquad(t\to0).\tag{34}$$

This difference follows from local cycle geometry while the two
defects share degree and edit controls. The common entire-function
arithmetic alone does not identify their heat behavior.

## 8. What the comparison establishes

The abstract one-generator topological algebra is the same for every
$d$: $f(B_d)\mapsto f(B_e)$ is an isomorphism of those closed
subalgebras. Its embedding retains different local growth laws, degree
bounds, and inverse-detection radii. This subalgebra isomorphism does
not establish an ambient automorphism or preservation of the original
radius labels and graph-positive cones.

The single-cut entire-function algebra is therefore not peculiar to a
line. Branching preserves that arithmetic structure while changing its
quantitative relation to observation radius. The exact joint theorem
also shows that different regular-tree backgrounds remain independent
coordinates when studied together. These facts do not settle arbitrary
defects or the intrinsic recovery of graph structure from the bare
topological algebra.

The function-algebra identification uses the same classical
entire-function coefficient norms described in Bhatt–Patel,
[*On Fréchet algebras of power
series*](https://repository.ias.ac.in/59672/1/9_PUB.pdf), Example 1.4.
The graph-specific statements are the stabilized marginal formula,
additive coordinates, rooted tree-factor recovery, and their application
to the ambient arithmetic. No originality or numerical speed claim is
made.
