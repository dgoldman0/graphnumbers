# Joint arithmetic of branching and planar cuts

30 September 2026. Work in the same local Cartesian graph completion
\(A=\mathcal A_{\mathrm{loc}}\). The complexification is used for
characters, entire functions and spectra. Real statements follow by
restricting to real Taylor coefficients.

Let \(B_d\), for \(d\ge3\), be the signed local limit obtained by
deleting one edge of the infinite \(d\)-regular tree: the graph after
deletion minus the graph before deletion. Let \(P\) be the analogous
single-edge deletion in the square lattice \(\mathbb Z^2\). These
elements can be constructed using finite tree or square-grid neighborhoods
whose boundaries recede from the fixed deleted edge. At each observation
radius the signed local differences stabilize, so the limits exist in
every defining weighted seminorm. They have vertex mass zero.

For any finite list of distinct integers \(d_1,\ldots,d_s\ge3\), the
closed subalgebra generated jointly by

\[
X=(B_{d_1},\ldots,B_{d_s},P)
\]

is a copy of \(\mathcal O(\mathbb C^{s+1})\). Its ambient units are
exactly the zero-free entire functions of these variables, and
\(\sigma_A(f(X))=f(\mathbb C^{s+1})\). Every nonconstant element of
this subalgebra has unbounded local total variation.

The proof uses a multiplicative face of local graph types and independent
coordinates on the regular background graphs. It does not require unique
factorization of arbitrary rooted balls, or freeness of all the positive
cut atoms.

## 1. Exact local variation of the two defect families

Fix \(r\ge2\). Let \(T_{d,r}\) be the rooted radius-\(r\) ball in
the intact \(d\)-regular tree, and let \(Z_r\) be the corresponding
ball in the intact square lattice. Write

\[
c_d(r)=2\sum_{j=0}^{r-1}(d-1)^j
=\frac{2((d-1)^r-1)}{d-2},\qquad c_P(r)=2r^2.
\tag{1}
\]

The local differences have the form

\[
T_rB_d=A_{d,r}-c_d(r)\delta_{T_{d,r}},\qquad
T_rP=A_{P,r}-c_P(r)\delta_{Z_r},
\tag{2}
\]

where the arrays \(A_{d,r},A_{P,r}\) are nonnegative and have masses
\(c_d(r),c_P(r)\), respectively. Their atoms are the balls rooted at
vertices affected by the deleted edge, observed in the graph after
deletion.

Both backgrounds are bipartite. A vertex is affected exactly when its
distance to the nearer endpoint of the deleted edge is at most \(r-1\).
Indeed, the distances to the two endpoints differ by one. The edge is
visible in the original ball precisely in that case. Deleting it either
removes that edge from the ball or also moves some vertices beyond the
radius; the resulting rooted ball has changed. Outside that set, no
path of length at most \(r\) uses the deleted edge and the ball is
unchanged.

In the tree, there are \(2(d-1)^j\) roots at distance \(j\) from the
nearer endpoint. In the lattice, place the deleted edge between
\((0,0)\) and \((1,0)\). At height \(y\), where \(|y|\le r-1\),
the union of the two endpoint diamonds contains \(2r-2|y|\) roots.
Its total size is

\[
2r+2\sum_{j=1}^{r-1}(2r-2j)=2r^2.
\]

The next section separates every positive atom in (2) from the intact
background. It follows that there is no cancellation between those parts,
and therefore

\[
\boxed{\|T_rB_d\|_1=2c_d(r),\qquad
\|T_rP\|_1=2c_P(r)=4r^2.}
\tag{3}
\]

Branching gives exponential growth in \(r\); the square-lattice cut
gives quadratic growth. Both differ quantitatively from the linear
variation growth of the cut-line defect \(E\).

## 2. A multiplicative face of locally regular balls

Call a rooted radius-\(r\) graph type \(B\) **interior-regular** if
every vertex at distance strictly less than \(r\) from its root has
the same degree in \(B\) as the root. Degrees at those interior
vertices are fully visible in the ball. Boundary degrees are unrestricted.
Let \(F_r\) be this subset of the local Cartesian monoid \(M_r\).

**Face lemma.** The one-vertex type belongs to \(F_r\), and

\[
B\star_r C\in F_r
\quad\Longleftrightarrow\quad B\in F_r\text{ and }C\in F_r.
\tag{4}
\]

**Proof.** At an interior product vertex \((u,v)\), its degree is
\(\deg_B(u)+\deg_C(v)\): all adjacent product vertices remain in the
radius-\(r\) ball. If both factors are interior-regular this equals
the product root degree. Conversely, test the interior product vertices
\((u,o_C)\) and \((o_B,v)\). Equality with the product root degree
forces \(\deg_B(u)=\deg_B(o_B)\) and
\(\deg_C(v)=\deg_C(o_C)\) for every interior factor vertex.

Consequently the indicator \(\rho_r=\mathbf1_{F_r}\) is a bounded
monoid character, allowing zero values. Projection of an absolutely
summable local array onto \(F_r\) is a contractive algebra homomorphism.

The intact backgrounds in (2) belong to \(F_r\). Every positive cut
atom lies outside it. To check the latter, take the nearer cut endpoint,
whose distance from the root is at most \(r-1\) and is unchanged by
the deletion. Its degree is deficient by one. If the root is not an
endpoint, its degree remains the original regular degree, so the two
interior degrees differ. If the root is an endpoint, it has a remaining
ordinary neighbor at distance one. Since \(r\ge2\), that neighbor is
interior and its degree is the original regular degree. Thus

\[
\rho_r A_{d,r}=0,\qquad \rho_r A_{P,r}=0.
\tag{5}
\]

Here \(\rho_r\) acts coefficientwise. This proves the separation
used in (3). It also ensures that a product containing any positive cut
atom contributes nothing to the interior-regular part.

## 3. Root-edge components distinguish the backgrounds

For any rooted ball \(B\) of radius \(r\ge2\), define an auxiliary
graph \(\Gamma(B)\). Its vertices are the neighbors of the root of
\(B\). Join two distinct such neighbors \(u,v\) when they have no
common neighbor other than the root. Equivalently, their two incident
root edges lie in no simple four-cycle together. Four-cycles need not
be induced. The definition uses only the radius-two ball.

**Root-edge product lemma.**

\[
\Gamma(B\star_r C)\cong\Gamma(B)\sqcup\Gamma(C).
\tag{6}
\]

**Proof.** Root neighbors in the product divide into the two factor
sets. Two neighbors from different factors have the additional common
neighbor \((u,v)\), visible at distance two, so they are not adjacent
in \(\Gamma\). For two neighbors in the same factor, all their common
neighbors other than the root lie in that factor: a vertex changing the
other coordinate cannot be adjacent to both distinct neighbors. Their
adjacency condition in \(\Gamma\) is therefore unchanged. This gives
exactly the disjoint union in (6).

For a nonempty connected finite graph \(H\), let

\[
\nu_H(B)=\#\{\text{components of }\Gamma(B)\text{ isomorphic to }H\}.
\tag{7}
\]

These are nonnegative integer-valued additive coordinates on the whole
local monoid. In the regular tree, any two root neighbors have only the
root as a common neighbor, whereas in the grid the opposite pairs have
only the root and each perpendicular pair has a second common neighbor.
Thus

\[
\Gamma(T_{d,r})=K_d,\qquad \Gamma(Z_r)=K_2\sqcup K_2.
\tag{8}
\]

For the chosen distinct \(d_i\ge3\), the coordinates
\(\nu_{K_{d_1}},\ldots,\nu_{K_{d_s}},\nu_{K_2}\) distinguish the
exponents of every product of the intact backgrounds: their values are
\((n_1,\ldots,n_s,2n_P)\). This proves the required independence
without a factorization theorem for the full local monoid.

## 4. Independent joint character disks

Choose \(|z_i|\le1\) and \(|w|\le1\). The function

\[
s_{r,z,w}(B)=\rho_r(B)
\prod_{i=1}^s z_i^{\nu_{K_{d_i}}(B)}
\,w^{\nu_{K_2}(B)},\qquad 0^0=1,
\tag{9}
\]

is a bounded unital semicharacter on \(M_r\), by (4) and (6).
It gives the continuous graph-algebra character

\[
\chi_{r,z,w}(Y)=\sum_B(T_rY)(B)s_{r,z,w}(B),\qquad
|\chi_{r,z,w}(Y)|\le\|T_rY\|_1.
\tag{10}
\]

Equations (2), (5) and (8) give

\[
\boxed{\chi_{r,z,w}(B_{d_i})=-c_{d_i}(r)z_i,\qquad
\chi_{r,z,w}(P)=-c_P(r)w^2.}
\tag{11}
\]

Since the squaring map sends the closed unit disk onto itself, these
characters realize the entire product of closed disks

\[
\prod_{i=1}^s\overline D(0,c_{d_i}(r))
\times\overline D(0,c_P(r)).
\tag{12}
\]

Every radius in (12) tends to infinity with \(r\). In particular,
every tuple in \(\mathbb C^{s+1}\) is obtained as \(\chi(X)\) by
one of these characters, and every compact polydisk is covered at one
common finite observation radius.

## 5. The joint generated algebra

Use a multiindex \(\alpha=(\alpha_1,\ldots,\alpha_s,\alpha_P)\),
and abbreviate \(c(r)=(c_{d_1}(r),\ldots,c_{d_s}(r),c_P(r))\).
For a polynomial \(f(z)=\sum_\alpha a_\alpha z^\alpha\), the
interior-regular part of \(T_rf(X)\) contains exactly the all-background
choice from each monomial. Its coefficient is
\(a_\alpha(-1)^{|\alpha|}c(r)^\alpha\). Distinct multiindices give
different background types by (8), so

\[
\boxed{\sum_\alpha|a_\alpha|c(r)^\alpha
\le\|T_rf(X)\|_1
\le\sum_\alpha|a_\alpha|(2c(r))^\alpha.}
\tag{13}
\]

The lower expression is exactly the norm of the interior-regular
projection. The upper bound follows from (3) and submultiplicativity.

Let \(D=\max(4,d_1,\ldots,d_s)\), using \(D=4\) when the list
is empty. Every monomial of total degree \(n\) is supported on balls
of maximum degree at most \(nD\). A radius-\(r\) ball there has at
most \((r+1)(1+nD)^r\) vertices. Therefore

\[
p_{r,k}(f(X))\le(r+1)^k
\sum_\alpha|a_\alpha|(1+D|\alpha|)^{rk}(2c(r))^\alpha.
\tag{14}
\]

Entire coefficients are summable with all such polynomially weighted
polyradii. Thus an entire \(f\) defines a convergent series \(f(X)\)
in every defining seminorm, and the evaluation map is a continuous
algebra homomorphism. Smaller observation radii are dominated by radius
two.

The coefficient norms \(\sum|a_\alpha|R^\alpha\), for positive
polyradii \(R\), give the usual compact-open topology of
\(\mathcal O(\mathbb C^{s+1})\): they bound disk suprema, and
multivariable Cauchy estimates on a larger polydisk bound the coefficient
norms in return. Formula (13), with all \(c_i(r)\) growing, bounds
each such coefficient norm by a defining graph seminorm at some radius.
Formula (14) gives continuity in the other direction. It follows that

\[
\boxed{\overline{\mathbb C[B_{d_1},\ldots,B_{d_s},P]}
\cong\mathcal O(\mathbb C^{s+1})}
\tag{15}
\]

as topological algebras. More explicitly, a convergent sequence of
polynomial expressions is Cauchy in every coefficient norm by (13),
its coefficients converge to those of an entire function, and (14)
identifies its graph limit with that entire function of \(X\).
This proves injectivity and closedness of the image as well as density
of the polynomials in that image.

The inequalities extend to entire \(f\) by convergence. If \(f\)
has any nonzero coefficient with \(|\alpha|>0\), the single lower
term \(|a_\alpha|c(r)^\alpha\) tends to infinity. Consequently

\[
f(X)\text{ has bounded local total variation}
\quad\Longleftrightarrow\quad f\text{ is constant}.
\tag{16}
\]

By the existing signed graph-measure characterization, only scalars in
this subalgebra have a finite signed graph-measure representation.

## 6. Ambient units, spectra and mixed examples

For every entire \(f\), continuity of (10) gives
\(\chi(f(X))=f(\chi(X))\). If \(f\) vanishes at any tuple,
Section 4 supplies an ambient character annihilating \(f(X)\), so
\(f(X)\) cannot be a unit of the whole graph algebra. Conversely,
if \(f\) is zero-free, \(1/f\) is entire on \(\mathbb C^{s+1}\)
and its evaluation supplies an inverse. Hence

\[
\boxed{f(X)\in A_{\mathbb C}^{\times}
\Longleftrightarrow f\text{ has no zero in }\mathbb C^{s+1}.}
\tag{17}
\]

Applying this to \(\lambda-f\) also proves

\[
\boxed{\sigma_{A_{\mathbb C}}(f(X))=f(\mathbb C^{s+1}).}
\tag{18}
\]

These are ambient statements: allowing inverses anywhere in the full
completion does not enlarge the unit set of the displayed subalgebra.
For real coefficients, the reciprocal is real whenever it exists.

For example, \(1-B_3P\) is a nonunit because \(1-z_1z_2\) has
zeros. Nevertheless

\[
\exp(B_3P)\exp(-B_3P)=1.
\tag{19}
\]

Both factors in (19) have unbounded local variation. Formula (13) gives
the quantitative bounds

\[
\exp(c_3(r)c_P(r))
\le\|T_r\exp(B_3P)\|_1
\le\exp(4c_3(r)c_P(r)),
\tag{20}
\]

where \(c_3(r)c_P(r)=4r^2(2^r-1)\). Thus the mixed arithmetic keeps
both the branching and planar contributions in its analytic growth.
The spectrum of this exponential is \(\mathbb C\setminus\{0\}\),
by (18).

## 7. A reusable criterion and the limits of the comparison

The mechanism requires three ingredients at arbitrarily large radii:

1. Each signed generator consists of a distinguished negative background
   atom of weight \(c_i(r)>0\), together with positive atoms outside a
   multiplicative face containing those backgrounds.
2. Additive nonnegative local coordinates distinguish the background
   exponents and allow their semicharacter values to vary independently
   over disks, as in (9)–(12).
3. Every background weight \(c_i(r)\) grows without bound, while the
   fixed-radius weighted norms of the generators are finite.

The face isolates independent background coefficients; this yields the
lower bound needed for an entire-function embedding. The semicharacters
give the ambient unit obstruction. Ordinary local submultiplicativity
already suffices for convergence of entire series; the degree estimate
(14) adds an explicit geometric upper bound.

The present families satisfy these hypotheses by direct graph
calculations. The same conclusion for a different medium requires
checking them again. In particular, the restriction \(d_i\ge3\) is
substantive in this proof: a line background gives \(\Gamma=K_2\),
whereas the square-lattice background gives \(\Gamma=2K_2\). These
specific background coordinates therefore do not establish independence
of the line-cut defect \(E\) and the planar cut \(P\).

The entire-function arithmetic found for the line is consequently part
of a broader phenomenon involving branching media, planar media, and
their joint Cartesian arithmetic. Their embeddings retain different
growth rates even though their generated abstract function algebras have
the same familiar form. No retraction of the full completion onto (15)
or intrinsic recognition theorem for these generators is asserted here.

The dependencies are the existing local convolution algebra, the
signed graph-measure characterization, and elementary entire-function
coefficient estimates. The graph-specific ingredients proved here are
the affected-root counts, the interior-regular face, and the root-edge
component coordinates. No general local prime-factorization theorem is
used.

The entire-function coefficient topology is classical; see
Bhatt–Patel, [*On Fréchet algebras of power series*](https://repository.ias.ac.in/59672/1/9_PUB.pdf),
Example 1.4 for its one-variable form. The multivariable coefficient
comparison above follows directly from Cauchy's formula on polydisks.
No novelty claim is made for the resulting function algebra.
