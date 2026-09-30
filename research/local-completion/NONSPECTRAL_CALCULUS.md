# Joint local statistics, nonspectral jets, and graph arithmetic

30 September 2026. Work in the same real local Cartesian graph completion
\(A=\mathcal A_{\mathrm{loc}}\). Cartesian product is multiplication,
\(1=K_1\), and \(V\) is the vertex-mass character. Its complexification is
used for Fourier transforms. This note develops a scalar and distribution
calculus; it does not assert that reweighting roots defines an endomorphism
or derivation of the graph algebra.

The [multiplication note](MULTIPLICATION_AND_UNITS.md) supplies the local
convolution algebras and separating additive rooted invariants. The
[character note](CHARACTERS_AND_INVERSION.md) supplies their faithful
Fourier family. The present specialization makes joint motif statistics,
their error bounds, and their behavior under reciprocals and exponentials
explicit. Cumulants and convolution identities are classical [1]; no
priority claim is made.

## 1. A continuous joint-distribution homomorphism

Fix a radius \(r\ge1\) and functions
\(a_1,\ldots,a_m:\mathcal B_r\to\mathbb Z\) such that

\[
a_j(B\star_r C)=a_j(B)+a_j(C),\qquad
|a_j(B)|\le C_j|B|^{d_j}.
\tag{1}
\]

The unit has \(a_j(e)=0\). The functions can be integer rescalings of any
finite collection of the rational additive invariants \(K_F\) constructed
in the multiplication note. A particularly accessible nonnegative family
is

\[
c_q(G,o)=\#\{q\text{-cliques of }G\text{ containing }o\},\qquad q\ge2.
\tag{2}
\]

Here \(c_2\) is degree, \(c_3\) counts root triangles, and \(c_4\) counts
root four-cliques, without dividing by four. A clique containing the root
of a Cartesian product is contained in one coordinate fiber: two neighbors
changing different coordinates are not adjacent. Thus (1) holds for all
these counts at radius one. Also
\(c_q(B)\le |B|^{q-1}/(q-1)!\).

For \(T_rX=(x_B)_B\), define the signed joint distribution

\[
\mu_a(X)(n)=\sum_{B:a(B)=n}x_B,\qquad n\in\mathbb Z^m.
\tag{3}
\]

Its coefficients need not be positive. They are absolutely summable with
every polynomial moment. Indeed, with \(d=\max_jd_j\),

\[
\sum_n(1+\|n\|_1)^k|\mu_a(X)(n)|
\le (1+\textstyle\sum_j C_j)^k p_{r,dk}(X),
\tag{4}
\]

with a weight exponent of at least one understood if necessary. The
triangle inequality also gives
\(\|\mu_a(X)\|_1\le\|T_rX\|_1\).

**Joint-distribution theorem.** Formula (3) is a continuous unital algebra
homomorphism from \(A\) into rapidly decaying signed arrays on
\(\mathbb Z^m\), equipped with convolution. In particular,

\[
\mu_a(XY)=\mu_a(X)*\mu_a(Y),\quad
\mu_a(1)=\delta_0,\quad \sum_n\mu_a(X)(n)=V(X).
\tag{5}
\]

**Proof.** Push the absolutely convergent local convolution for
\(T_r(XY)\) through the additive map \(a\). Reordering is justified by
the product of the two local total variations. The same argument with
polynomial weights proves continuity and (4).

For positive mass-one elements, (3) is a probability distribution. Equation
(5) then describes the joint statistics at an independently selected pair
of roots in a Cartesian product. The theorem also applies to signed
elements and to elements outside the finite signed graph-measure model.

## 2. Fourier transforms and motif generating functions

The joint characteristic transform

\[
\widehat\mu_a(X)(\theta)=
\sum_n\mu_a(X)(n)e^{i\langle n,\theta\rangle}
\tag{6}
\]

is a smooth function on the \(m\)-torus. Every derivative is obtained
termwise because every polynomial moment is absolutely summable. Each
fixed \(\theta\) gives a continuous character on \(A\). Fourier inversion
recovers the entire joint array, including its correlations.

When all \(a_j\ge0\), the generating function

\[
F_X(z)=\sum_{n\in\mathbb N^m}\mu_a(X)(n)z^n,
\qquad |z_j|\le1,
\tag{7}
\]

is holomorphic in the open polydisk and has continuous partial derivatives
of every order on its closure. It satisfies
\(F_{XY}=F_XF_Y\) and \(F_X(1,\ldots,1)=V(X)\).
Consequently every zero of \(F_X\) in the closed polydisk obstructs
invertibility of \(X\). A chosen finite family of motif statistics need
not detect every obstruction.

The transform is quantitatively stable:

\[
\sup_{|z_j|\le1}|F_X(z)-F_Y(z)|
\le\|T_r(X-Y)\|_1.
\tag{8}
\]

If \(X,Y\) are actual units, and both transforms have modulus at least
\(\delta>0\) on a chosen set, then on that set

\[
|F_{X^{-1}}-F_{Y^{-1}}|
\le\delta^{-2}\|T_r(X-Y)\|_1.
\tag{9}
\]

This follows by subtracting scalar reciprocals. It is a condition on
the chosen transforms, and supplies no openness claim for the full unit
group.

## 3. Joint moments and higher Leibniz jets

For a multiindex \(\alpha\in\mathbb N^m\), put

\[
M_\alpha(X)=\sum_B x_B\prod_j a_j(B)^{\alpha_j},\qquad M_0=V.
\tag{10}
\]

Every \(M_\alpha\) is a continuous linear observable, with

\[
|M_\alpha(X-Y)|\le
\left(\prod_j C_j^{\alpha_j}\right)
p_{r,\max(1,\sum_jd_j\alpha_j)}(X-Y).
\tag{11}
\]

Additivity of \(a\) and the multiindex binomial theorem give the exact
higher Leibniz rule

\[
M_\alpha(XY)=
\sum_{\beta\le\alpha}\binom\alpha\beta
M_\beta(X)M_{\alpha-\beta}(Y).
\tag{12}
\]

In particular each \(M_{e_j}\) is a continuous point derivation at \(V\):
\(M_{e_j}(XY)=M_{e_j}(X)V(Y)+V(X)M_{e_j}(Y)\).
For every finite order \(N\), the map

\[
J_N(X)=\sum_{|\alpha|\le N}\frac{M_\alpha(X)}{\alpha!}t^\alpha
\quad\bmod (t_1,\ldots,t_m)^{N+1}
\tag{13}
\]

is a continuous unital algebra homomorphism into a finite-dimensional
truncated polynomial algebra. The factor \(\alpha!\) is essential.

These are ordinary moments. Differentiating (7) directly at \(z=1\)
instead gives falling-factorial moments. Formally substituting
\(z_j=e^{t_j}\) gives (13). We only use this as a formal power-series
identity: the original topology does not guarantee a moment-generating
function on any real neighborhood of \(t=0\).

## 4. Reciprocals, exponentials, and cumulants

If \(X\) is a unit and \(v=V(X)\), then \(v\ne0\), and (12) determines
the inverse moments recursively:

\[
M_0(X^{-1})=v^{-1},\qquad
M_\alpha(X^{-1})=-v^{-1}
\sum_{0<\beta\le\alpha}\binom\alpha\beta
M_\beta(X)M_{\alpha-\beta}(X^{-1}).
\tag{14}
\]

For one statistic this begins

\[
M_1(X^{-1})=-M_1(X)/v^2,\qquad
M_2(X^{-1})=2M_1(X)^2/v^3-M_2(X)/v^2.
\tag{15}
\]

Every element has an algebra exponential. Indeed, the defining seminorms
are submultiplicative, so \(\sum X^n/n!\) converges in every seminorm and
in the complete algebra. Continuity of (3) and (13) therefore gives

\[
\mu_a(e^X)=\exp_*(\mu_a(X)),\qquad
J_N(e^X)=\exp(J_N(X)),\qquad F_{e^X}=e^{F_X}.
\tag{16}
\]

The exponential on the truncated polynomial algebra includes its scalar
constant term; its positive-degree part is a finite nilpotent sum.

For every \(X\) with \(V(X)\ne0\), define joint cumulants by the formal
series

\[
\log\left(\frac{1}{V(X)}
\sum_\alpha \frac{M_\alpha(X)}{\alpha!}t^\alpha\right)
=\sum_{|\alpha|>0}\frac{\kappa_\alpha(X)}{\alpha!}t^\alpha.
\tag{17}
\]

The logarithm exists formally because its argument has constant term one.
The first cumulants are normalized means and covariances:

\[
\kappa_{e_i}=M_{e_i}/V,\qquad
\kappa_{e_i+e_j}=M_{e_i+e_j}/V-M_{e_i}M_{e_j}/V^2.
\tag{18}
\]

For signed elements these are algebraic cumulants and need not obey
probabilistic positivity inequalities. Formal logarithms and (12) prove

\[
\boxed{\kappa_\alpha(XY)=\kappa_\alpha(X)+\kappa_\alpha(Y),\quad
\kappa_\alpha(X^{-1})=-\kappa_\alpha(X),\quad
\kappa_\alpha(e^{sX})=sM_\alpha(X).}
\tag{19}
\]

The first identity assumes nonzero masses; the second assumes an actual
inverse. Moment and cumulant formulas alone cannot establish that an
inverse exists: every jet (13) is invertible as soon as \(V(X)\ne0\),
whereas graph-algebra invertibility has stronger requirements.

The functions (17) at each finite order are rational expressions in
continuous moments and \(V\), hence continuous wherever \(V\ne0\).
For example, if \(|V(X)|,|V(Y)|\ge b>0\),

\[
|\kappa_{e_i}(X)-\kappa_{e_i}(Y)|
\le |M_{e_i}(X-Y)|/b+
|M_{e_i}(Y)|\,|V(X-Y)|/b^2.
\tag{20}
\]

Thus the calculus has explicit local-error control as well as identities.

## 5. Separate marginal distributions lose geometric correlations

Let \(G=K_3\sqcup K_{1,3}\). Let \(P\) be the paw graph, a triangle
with one pendant leaf, and put \(J=P\sqcup P_3\).
Both ordinary graphs have seven vertices. Their degree multisets are
\(1^3,2^3,3^1\), and their root-triangle multisets are \(0^4,1^3\).
Thus even the full individual distributions of degree and triangle count
agree. Their joint distributions differ:

| Root degree, root triangles | Multiplicity in \(G\) | Multiplicity in \(J\) |
|---|---:|---:|
| \((1,0)\) | 3 | 3 |
| \((2,0)\) | 0 | 1 |
| \((2,1)\) | 3 | 2 |
| \((3,0)\) | 1 | 0 |
| \((3,1)\) | 0 | 1 |

Consequently

\[
F_{G-J}(z,w)=z^2(w-1)(1-z),\qquad
M_{(1,1)}(G)=6,\quad M_{(1,1)}(J)=7.
\tag{21}
\]

The normalized covariance difference is \(-1/7\). Under multiplication
by any mass-one background \(Y\), (12) gives

\[
M_{(1,1)}((U(G)-U(J))Y)=-1/7,
\tag{22}
\]

because the mass and both first moments of the difference vanish. The
correlation witness survives every such Cartesian background, including
completed ones. Correlation supplies information beyond adding another
separate marginal observable.

## 6. A nonspectral reciprocal and exponential response

Use the [rook and Shrikhande graphs](GEOMETRY_VERSUS_SPECTRUM.md), with
\(R=U(K_4\square K_4)\), \(S=U(\mathrm{Shrikhande})\), and \(X=R-S\).
Every scalar adjacency or Laplacian spectral trace annihilates \(X\).
The root four-clique count is constantly two on \(R\) and zero on \(S\).
Thus

\[
\mu_{c_4}(X)=\delta_2-\delta_0,\quad
F_X(w)=w^2-1,\quad V(X)=0,\quad M_q(X)=2^q\ (q\ge1).
\tag{23}
\]

For \(U_t=1+tX\), whenever the actual inverse exists and
\(|t|<1/2\),

\[
F_{U_t^{-1}}(w)=\frac{1}{1-t+tw^2},\qquad
\mu_{c_4}(U_t^{-1})=
\frac{1}{1-t}\sum_{n\ge0}
\left(\frac{-t}{1-t}\right)^n\delta_{2n}.
\tag{24}
\]

The series has all polynomial moments in this parameter range. The
companion arithmetic investigation determines the actual invertibility
range using a richer transform; (24) is its projected motif response.
The first three raw moments are

\[
M_1=-2t,\qquad M_2=8t^2-4t,\qquad
M_3=-48t^3+48t^2-8t.
\tag{25}
\]

The algebra exponential has the particularly transparent response

\[
F_{e^{tX}}(w)=\exp(t(w^2-1)),\qquad
\kappa_q(e^{tX})=2^q t\quad(q\ge1).
\tag{26}
\]

For \(t\ge0\), its projected motif distribution is the law of twice a
Poisson random variable with mean \(t\). This does not imply positivity
of the underlying graph element: a positive pushforward can result from
a signed input. The exponential series here has global finite-graph
variation at most \(e^{2|t|}\). Its signed spectral measure is the
convolution exponential of the zero signed spectral measure of \(tX\),
hence \(\delta_0\). Every bounded scalar spectral trace therefore treats
the same exponential as the scalar unit, while these motif responses
vary with \(t\). This argument supplies a concrete spectral domain for
this exponential; it does not extend heat continuously to all of \(A\).

The one-statistic transform in (24) detects the obstruction at positive
\(t=1/2\), by evaluating \(w=i\). It does not detect the negative
threshold. For \(t<0\), its denominator is nonzero throughout the closed
unit disk. A projected inverse or a finite jet is not a full graph inverse.

## 7. Full distributions contain more than all moment jets

Finite collections of clique statistics do not separate all graph
elements. Even *all polynomial moments* of those statistics can vanish
on a nonzero completed element. To see the second obstruction, use the
hypercube retract \(S:A^\infty(\overline{\mathbb D})\to A\), where
\(H=K_2/2\), and set

\[
f(z)=\exp\bigl(-(1-z)^{-1/2}\bigr),\qquad f(1)=0.
\tag{27}
\]

Choose the branch positive for real \(z<1\). On the closed disk near one,
\(|\arg(1-z)|\le\pi/2\), so
\(\Re(1-z)^{-1/2}\ge |1-z|^{-1/2}/\sqrt2\).
Every derivative of (27) is bounded by a power of \(|1-z|^{-1}\) times
this decaying exponential. Thus \(f\) and all its derivatives extend
continuously and vanish at one. Its Taylor coefficients are real and
rapidly decreasing, so \(Z=S(f)\in A\).

Since \(f(0)=e^{-1}\), \(Z\ne0\). On a hypercube every root has degree
equal to its dimension and has zero clique counts of size at least three.
All degree moments of \(Z\) are values at one of
\((z\,d/dz)^qf\), hence zero; every mixed clique moment is zero as well.
The joint distribution and full transform still distinguish \(Z\).
This is an exact signed example; no positive moment-indeterminacy claim
is needed.

For a faithful distribution framework one may instead use *all finite
joint vectors* of the additive invariants \(K_F\) at every radius. Their
rooted values separate balls. For a fixed ball, the sets matching its
first \(m\) invariant values decrease to that singleton. Absolute local
summability and dominated convergence recover its coefficient from the
joint distributions. This is the distribution form of the established
separating-character theorem, rather than a new classification of all
characters.

### Finite reconstruction under a degree bound

The failure of moments to determine (27) depends on the unbounded range
of the retained statistic. Suppose instead that every radius-one
coefficient of \(X\) outside root degree at most \(D\) is zero. For
clique counts \(c_{q_1},\ldots,c_{q_m}\), let
\(N_j=\binom D{q_j-1}\). On this class each statistic lies in
\(\{0,\ldots,N_j\}\). Define the Lagrange polynomials

\[
L_{j,n}(u)=\prod_{\substack{0\le h\le N_j\\h\ne n}}
\frac{u-h}{n-h}.
\tag{28}
\]

Then the joint coefficient at \(n=(n_1,\ldots,n_m)\) is exactly the
observable obtained by integrating \(\prod_jL_{j,n_j}(a_j)\) against
\(T_1X\). Expanding that polynomial expresses it in mixed moments of
total order at most \(\sum_jN_j\). Empty products cover \(N_j=0\).
Thus finitely many moments recover the full selected joint law under
the stated bound.

There is also a stronger local reconstruction statement. Fix a global
degree bound \(D\) and radius \(r\). Only finitely many rooted balls
are possible, since their sizes are bounded by
\(1+D+\cdots+D^r\). The established additive \(K_F\) invariants
separate this finite set. Choose finitely many separating invariants,
clear denominators, and take an integer linear combination whose
values \(b_1,\ldots,b_N\) on those balls are pairwise distinct. Such
a combination exists because only finitely many proper hyperplanes
of coefficient choices are forbidden. Equivalently, a sufficiently
large positional base encodes their bounded integer value vectors.
Write the resulting additive statistic as \(a\).

For the \(i\)-th ball its coefficient in \(T_rX\) is

\[
\sum_B x_B\prod_{h\ne i}\frac{a(B)-b_h}{b_i-b_h},
\tag{29}
\]

whenever \(T_rX\) is supported on the allowed finite set. Hence raw
moments of this statistic through order \(N-1\) recover the entire
radius-\(r\) histogram. A more economical vector can use multivariate
interpolation instead. These degree-controlled classes are not claimed
to be subalgebras at a fixed bound: Cartesian multiplication adds degree
bounds. The reconstruction assertion itself is exact and finite.

## 8. Why scalar point derivations do not automatically lift to graph derivations

A tempting formula multiplies each rooted coefficient by \(c_3(B)\),
or by \(e^{s c_3(B)}\), while keeping the underlying graph. It respects
the additive-statistic convolution rule in the ambient array algebra,
but generally violates mass-transport balance.

On the paw graph, transport one unit from its unique leaf to the triangle
vertex adjacent to that leaf. After weighting roots by \(c_3\), the total
outgoing mass is zero and the total incoming mass is one. After weighting
by \(e^{s c_3}\), the corresponding totals are one and \(e^s\), unequal
for real \(s\ne0\). Therefore these weighted rooted arrays fail the
representation theorem's balance condition and are not elements of \(A\).

Equations (12)–(19) are valid as scalar observables, distribution maps,
and finite-dimensional jets. Constructing an \(A\)-valued derivation
requires a separate balanced construction. No such construction is
inferred from the point-derivation identity alone.

## 9. Scope and precedent

The [Python implementation](../../python/README.md) exposes a narrower
effective interface: statistic axes take nonnegative integer values with
growth constant one, exact joint distributions have finite support, and
certified jets use supplied weighted local approximation oracles. Its
rational formal exponential and logarithm require constant coefficient
zero and one respectively. Nonlinear operations on the center of a jet
certificate do not propagate that certificate's error intervals. The
distribution theorem above also covers signed integer statistics, other
growth constants, and infinite rapidly summable distributions.

This provides a nonspectral calculus on the entire completed algebra:
joint statistics have stable distribution transforms, exact higher
Leibniz rules, and controlled inverse and exponential responses. It keeps
correlations that separate marginals lose, and it can measure changes
invisible to all scalar spectral traces. The distinction between full
distributions, jets, and balanced graph elements is part of the result.

The convolution, cumulant, and formal logarithm rules are classical.
Their applicability follows here from the proven local additivity and
polynomial growth of the chosen graph statistics. The rook–Shrikhande
cospectrality and motif discrepancy are classical and sourced in the
earlier geometry note. The correlation pair (21) is supplied explicitly,
so its values are directly checkable from seven-vertex graphs.

[1] T. P. Speed, *Cumulants and Partition Lattices*, Australian Journal of
Statistics 25 (1983), 378–388.
https://doi.org/10.1111/j.1467-842X.1983.tb00391.x . This primary paper
develops joint cumulants through the partition-lattice Möbius function;
the formal-logarithm identities used here are proved directly above.
