# Local geometry, analytic retracts, and graph arithmetic

30 September 2026. Write $A=\mathcal A_{\mathrm{loc}}$ for the established
real local Cartesian graph completion. Complex spectra below are taken in its
complexification $A_{\mathbb C}$. The operations and topology are unchanged.

The results connect neighbor geometry to completed power series, ambient
invertibility, spectra, and divisibility. In particular, a family that all
scalar adjacency and Laplacian spectral observables identify with the unit
has an exact geometric threshold for invertibility.

## 1. Neighbor links give several independent coordinates

For a rooted graph $(G,o)$ let $\operatorname{lk}(G,o)$ be the induced
graph on the neighbors of $o$. It is determined by the induced radius-one
ball. A Cartesian product has

$$
\operatorname{lk}(G\square F,(o,v))
=\operatorname{lk}(G,o)\sqcup\operatorname{lk}(F,v).
\tag{1}
$$

Indeed, neighbors obtained by changing different coordinates have distance
two and are not adjacent; within either coordinate the original induced
adjacency is retained.

Fix pairwise nonisomorphic nonempty connected finite graphs
$J_1,\ldots,J_m$. Let $c_i(B)$ count connected components isomorphic to
$J_i$ in the neighbor link of the radius-one ball $B$. Equation (1) gives

$$c_i(B\star_1 C)=c_i(B)+c_i(C),\qquad
\sum_i c_i(B)\le\deg(o_B)=|B|-1.\tag{2}$$

Define the rapidly decreasing power-series algebra

$$
\mathscr S_m=
\left\{f(z)=\sum_{n\in\mathbb N^m}a_nz^n:
q_k(f):=\sum_n|a_n|(1+|n|)^k<\infty\quad(k\ge1)\right\}.
\tag{3}
$$

Here $|n|=\sum_i n_i$, and real coefficients are used unless stated
otherwise. For every $X\in A$ put

$$
\mathcal L_J(X)(z)=
\sum_{B\in\mathcal B_1}a_1^X(B)
z_1^{c_1(B)}\cdots z_m^{c_m(B)}.
\tag{4}
$$

Grouping equal exponents is absolutely legitimate, and

$$q_k(\mathcal L_J(X))\le p_{1,k}(X).\tag{5}$$

Equation (2), absolute summability, and local convolution show that
$\mathcal L_J:A\to\mathscr S_m$ is a continuous unital algebra
homomorphism. It is a joint generating function of local geometric
statistics. Components of the link having other isomorphism types are
ignored by this particular map.

## 2. A geometric retract theorem

Assume finite connected graphs $G_1,\ldots,G_m$ satisfy

$$\operatorname{lk}(G_i,v)\cong J_i\quad
\text{for every }v\in V(G_i).\tag{6}$$

Thus $G_i$ is regular of degree $d_i=|V(J_i)|\ge1$. Vertex transitivity
makes the links isomorphic, but they must still have the prescribed
connected types $J_i$, distinct across $i$. Transitivity alone is not
sufficient for these hypotheses. Write $U_i=G_i/|V(G_i)|$.

**Theorem.** The map

$$S_G(f)=\sum_{n\in\mathbb N^m}a_n U_1^{n_1}\cdots U_m^{n_m}\tag{7}$$

is a topological algebra isomorphism from $\mathscr S_m$ onto
$B_G=\overline{\mathbb R[U_1,\ldots,U_m]}\subset A$.
Moreover,

$$\mathcal L_J S_G=\operatorname{id},\qquad
P_G=S_G\mathcal L_J:A\to B_G\tag{8}$$

is a continuous unital algebra retraction. In particular, $B_G$ is a
closed complemented subalgebra.

**Proof.** Every root of the product indexed by $n$ has neighbor link
$\bigsqcup_i n_i J_i$, so distinct $n$ have distinct radius-one ball
types. Their supports remain disjoint at every larger radius, since
truncation recovers those distinct one-balls. All normalized product
histograms are positive of mass one. Therefore, for finite sums,

$$
p_{1,k}(S_G(f))=\sum_n|a_n|(1+d\cdot n)^k.
\tag{9}
$$

Each product is regular of degree $d\cdot n$. Its $r$-balls have at most
$\sum_{j=0}^r(d\cdot n)^j\le(r+1)(1+d\cdot n)^r$ vertices. Hence

$$
p_{r,k}(S_G(f))
\le(r+1)^k\sum_n|a_n|(1+d\cdot n)^{rk}\quad(r\ge1).
\tag{10}
$$

Since $|n|\le d\cdot n\le(\max d_i)|n|$, these estimates identify the
induced topology with (3). The series (7) converges in every defining
seminorm, multiplication extends the Cauchy product, and finite
polynomials are dense. Equation (6) gives $\mathcal L_J(U_i)=z_i$,
which proves (8). The image of the continuous idempotent $P_G$ is closed,
finishing the proof. The same proof applies over $\mathbb C$.

### The classical analytic algebra appearing here

The complex algebra $\mathscr S_m$ is $A^\infty(\overline{\mathbb D}^m)$:
holomorphic functions on the open polydisc whose derivatives of every
order extend continuously to its closure. Rapid coefficient decay gives
uniform convergence of all differentiated series. Conversely, the
restriction to the distinguished boundary $\mathbb T^m$ is smooth;
integration by parts, or repeated application of its torus Laplacian,
gives $|a_n|\le C_N(1+|n|)^{-N}$ for every $N$. Taking $N>k+m$ gives
(3). These arguments also identify the Fréchet topologies.

No new function algebra is being proposed. The content of the theorem is
its realization and continuous retraction through specified graph
geometry. The one-variable case with $G_1=K_2$ has the same image as the
earlier [normalized-edge retract](MULTIPLICATION_AND_UNITS.md), but a
different retraction: it counts isolated vertices of the link, whereas
the earlier map counts root degree. A triangle has zero of the former
and two of the latter.

## 3. Ambient units, spectra, and divisibility

**Unit and spectrum theorem.** For $f\in\mathscr S_m$,

$$
S_G(f)\in A^\times
\quad\Longleftrightarrow\quad
f(z)\ne0\quad\text{for every }z\in\overline{\mathbb D}^m.
\tag{11}
$$

When this holds, its inverse is $S_G(1/f)$. For complex coefficients,

$$\sigma_{A_{\mathbb C}}(S_G(f))
=f(\overline{\mathbb D}^m).\tag{12}$$

**Proof.** Evaluation at any point of the closed polydisc, composed with
$\mathcal L_J$, is a continuous complex-valued character on $A$. A zero
of $f$ consequently obstructs even an inverse lying outside $B_G$.
If there are no zeros, compactness bounds $|f|$ away from zero; the usual
derivative formulas show that $1/f$ belongs to $A^\infty$, and (7)
supplies an inverse. Real coefficients are preserved by reciprocal.
Apply the same equivalence to $\lambda-f$ to obtain (12).

For instance, real or complex numbers $b_1,\ldots,b_m$ satisfy

$$1-\sum_i b_iU_i\in A^\times
\quad\Longleftrightarrow\quad\sum_i|b_i|<1.\tag{13}$$

The linear form $\sum b_i z_i$ maps the closed polydisc onto the closed
disk of radius $\sum|b_i|$. In the unit range the reciprocal is the
convergent multinomial series. Its terms are normalized Cartesian
products with the explicitly prescribed constituent geometries.

**Divisibility theorem.** For $x,y\in B_G$,

$$y\in xA\quad\Longleftrightarrow\quad y\in xB_G.\tag{14}$$

Indeed, if $y=xv$, applying $P_G$ gives $y=xP_G(v)$; the converse is
immediate. Thus leaving this closed subalgebra cannot create new
divisibility relations between its elements. In function language,
$S_G(f)$ divides $S_G(g)$ in the full completion exactly when
$g=fh$ for some $h\in A^\infty(\overline{\mathbb D}^m)$.
Zeros alone do not provide a general multivariate divisibility test;
the requirement on $h$ includes boundary regularity and multiplicity.

**Relative algebraic-closure theorem.** If an element $x\in A$ satisfies
a nonzero polynomial with coefficients in $B_G$, then $x\in B_G$.
The same statement holds for the complexified algebras.

**Proof.** Both algebras are integral domains by the established domain
theorem. Localize $A$ at $B_G\setminus\{0\}$; the retraction extends to
a homomorphism from this localization to $K=\operatorname{Frac}(B_G)$,
fixing $K$. If $x$ is algebraic over $K$, let $m(T)\in K[T]$ be its
monic irreducible minimal polynomial. Applying the extended retraction
to $m(x)=0$ gives $m(P_G(x))=0$ in $K$. Thus $m$ has a root in its
coefficient field, forcing its degree to be one. It follows that
$x=P_G(x)\in B_G$.

In particular, every ambient solution of an algebraic equation over
these geometric coordinates is already a smooth polydisc function of
them. Divisibility is its degree-one instance. For $H=K_2/2$ and
$n\ge2$, neither $H$ nor $1+H$ has an $n$th root even in $A_{\mathbb C}$.
For $H$, a holomorphic root would give a zero of order $1/n$ at the
origin. For $1+H$, any smooth disk root $g$ would satisfy
$g(-1)=0$ and $ng^{n-1}g'=1$ up to the boundary, a contradiction at
$z=-1$. Thus allowing arbitrary completed elements does not repair
either algebraic or boundary-regularity obstruction.

## 4. Cliques and the arithmetic information discarded by degree

Take $H_d=K_{d+1}/(d+1)$. Its neighbor link is $K_d$, so every finite
collection of distinct $H_d$ satisfies the retract theorem. These are
independent geometric coordinates even where total root degree agrees.

The two-generator case $H=H_1$, $T=H_2$ already makes that distinction
explicit. Write $B=\overline{\mathbb R[H,T]}$, identify its variables
with $(z,w)$, and let $D$ be the degree generating function. Then

$$D(S_G(f))(u)=f(u,u^2).\tag{15}$$

**Principal-kernel theorem.**

$$\ker(D|_B)=(T-H^2)B.\tag{16}$$

The ideal is closed, and division by $T-H^2$ is continuous from this
kernel to $B$. In particular, this is also an exact ambient divisibility
criterion for members of $B$.

**Proof.** If $f(z,z^2)=0$, the fundamental theorem of calculus gives

$$
f(z,w)=(w-z^2)g(z,w),\qquad
g(z,w)=\int_0^1
\partial_w f\bigl(z,z^2+s(w-z^2)\bigr)\,ds.
\tag{17}
$$

The intermediate argument lies in the closed disk, since it is a convex
combination of $z^2$ and $w$. Differentiating under this integral shows
that $g$ is holomorphic inside and smooth on the closed polydisc, with
every derivative bounded by finitely many derivatives of $f$.
This proves membership and continuity. The reverse implication follows
by substituting $w=z^2$. The kernel of the continuous map $D|_B$ is
closed; (14) gives the ambient assertion. Uniqueness of the quotient
also follows from the integral-domain theorem.

Geometrically, $T$ is the normalized triangle, while $H^2$ is the
normalized four-cycle. Every element of this subalgebra whose full
degree generating function vanishes is divisible by their difference.

For a concrete unit distinction, put

$$x=1+\tfrac14H^2,\qquad
y=1-\tfrac12H^2+\tfrac34T.\tag{18}$$

They have the identical, zero-free degree generating function
$D(x)(u)=D(y)(u)=1+u^2/4$. Nevertheless $x$ is a unit, whereas $y$ is
a nonunit: its two-variable polynomial vanishes at $(z,w)=(1,-2/3)$.
Their difference $y-x$ is $3(T-H^2)/4$, precisely in the principal kernel.

More generally, adjoining $H_{d_2},\ldots,H_{d_m}$ to $H_1$ gives

$$\ker D=
\sum_{i=2}^m(H_{d_i}-H_1^{d_i})B_G.\tag{19}$$

Successively replace $z_i$ by $z_1^{d_i}$ and apply (17) in that
coordinate to prove this finite sum identity and continuous division
operators. This identifies a finitely generated closed ideal, rather
than merely exhibiting a few equal-degree examples.

## 5. A spectral-blind family with an exact unit threshold

Let $Q=K_4/4$, let $R=Q^2$ be the normalized $4\times4$ rook graph,
and let $S$ be the normalized Shrikhande graph. The neighbor link of
$K_4$ is $K_3$, and the neighbor link of the Shrikhande graph is $C_6$.
These distinct connected links make $Q,S$ independent coordinates of
another retract. Set

$$X=R-S=Q^2-S,\qquad Y_t=1+tX\quad(t\in\mathbb R).\tag{20}$$

The two unnormalized graphs each have 16 vertices, degree six, adjacency
spectrum $6^1,2^6,(-2)^9$, and Laplacian spectrum $0^1,4^6,8^9$.
These finite spectral facts can be verified without approximation.
For the rook graph they follow by adding the two $K_4$ spectra.
For the Shrikhande graph use the Cayley graph on $\mathbb Z_4^2$ with
steps $\pm(1,0),\pm(0,1),\pm(1,1)$; its Fourier eigenvalues are

$$2\cos(\pi a/2)+2\cos(\pi b/2)+2\cos(\pi(a+b)/2),
\qquad 0\le a,b<4.\tag{21}$$

Enumeration gives the displayed multiplicities; the six neighbors of
zero induce a six-cycle. Translation handles every other root.

Consequently every scalar function of either finite spectrum has equal
normalized traces on $R,S$. In particular,

$$D(X)=0,\qquad H_\tau(X)=0\quad(\tau\ge0),\qquad
D(Y_t)=1,\quad H_\tau(Y_t)=1.\tag{22}$$

All these equalities hold for every $t$, yet the full algebra gives

$$
\boxed{\sigma_{A_{\mathbb C}}(X)=\{\lambda:|\lambda|\le2\},\qquad
Y_t\in A^\times\ \Longleftrightarrow\ |t|<\tfrac12.}
\tag{23}
$$

Indeed, $X$ corresponds to $z^2-w$, whose image on the closed bidisc is
the entire closed disk of radius two. Apply (11) and (12). Thus the
local geometric distinction between the rook and Shrikhande graphs
controls ambient invertibility even though their entire scalar
adjacency and Laplacian spectral data agree.

### The inverse and its geometric size

For $|t|<1/2$,

$$
Y_t^{-1}=
\sum_{a,b\ge0}\binom{a+b}{a}(-t)^a t^b R^aS^b.
\tag{24}
$$

Products indexed by $(a,b)$ have neighbor links
$2aK_3\sqcup bC_6$, so their local supports never cancel against one
another. Their degree is $6(a+b)$. It follows exactly that, for $r\ge1$,

$$\|T_r(Y_t^{-1})\|_1=\frac1{1-2|t|},\qquad
p_{1,k}(Y_t^{-1})=\sum_{n\ge0}(2|t|)^n(6n+1)^k.\tag{25}$$

The first statement uses mass one for each normalized Cartesian product
and the binomial identity $\sum_{a+b=n}\binom n a=2^n$. The second also
uses its fixed degree. Thus this inverse belongs even to the finite
signed graph-measure model, while its local variation diverges at the
exact algebraic unit boundary.

Truncating (24) at total index $a+b\le N$ gives the explicit certificate

$$
p_{r,k}\left(Y_t^{-1}-\sum_{a+b\le N}
\binom{a+b}{a}(-t)^a t^bR^aS^b\right)
\le(r+1)^k\sum_{n>N}(2|t|)^n(6n+1)^{rk}.
\tag{26}
$$

For a fixed $r,k,t$ in the unit range this is an exponentially decaying
tail times a polynomial, and can be bounded by elementary rational
series estimates when $t$ is rational. At radius one (25) gives the
exact weighted tail by restricting its sum to $n>N$.

## 6. A constructive inverse beyond the selected subalgebras

There is a useful general existence statement from the established
[local unit criterion](CHARACTERS_AND_INVERSION.md). If

$$\|T_r(Y)\|_1<1\qquad\text{for every }r\ge1,\tag{27}$$

then $1-Y$ is a unit of $A$: each marginal has its ordinary $\ell^1$
Neumann inverse, and the unit criterion lifts these compatible local
inverses to $A$. The bound can depend on $r$; a uniform bound $q<1$ is
sufficient. No degree restriction is needed for this existence result.

A degree cap supplies an effective certificate in the original weighted
topology. Suppose every local marginal of $Y$ is supported on graphs
of maximum degree at most $D$, and

$$\|T_r(Y)\|_1\le q<1\qquad(r\ge1).\tag{28}$$

The $n$-fold local product is supported on balls of degree at most $nD$
and has total variation at most $q^n$. Thus

$$p_{r,k}(Y^n)\le(r+1)^k(1+nD)^{rk}q^n\quad(r,k\ge1).\tag{29}$$

Consequently $Z=\sum_{n\ge0}Y^n$ converges in $A$, and multiplication
of its partial sums by $1-Y$ proves $Z=(1-Y)^{-1}$. Furthermore,

$$\|T_r(Z)\|_1\le\frac1{1-q},\qquad
p_{r,k}\left(Z-\sum_{n=0}^NY^n\right)
\le(r+1)^k\sum_{n>N}(1+nD)^{rk}q^n.\tag{30}$$

The output can have unbounded degree even though its input has a finite
degree cap. A bounded-degree annotation must therefore not be inherited
by the inverse in general.

### Exact rational tail bounds

For $m\ge0$ let

$$F_m(q)=\sum_{n\ge0}n^mq^n
=\sum_{j=0}^m\left\{\!\begin{matrix}m\\j\end{matrix}\!\right\}
\frac{j!q^j}{(1-q)^{j+1}},\tag{31}$$

with $0^0=1$ and the usual Stirling numbers of the second kind. Expanding
$n^m$ in falling factorials and differentiating the geometric series
proves (31). With $m=rk$, the tail in (30) is exactly

$$
(r+1)^k q^{N+1}\sum_{j=0}^m\binom mj
\bigl(1+D(N+1)\bigr)^{m-j}D^jF_j(q).
\tag{32}
$$

It is rational for rational $q,D$ and tends to zero. Formula (32)
therefore produces a finite, certified truncation without numerical
spectral calculations.

### Stability under an inexact local oracle

Fix $r,k$ and write $a=T_r(Y)$. Suppose a finite array $\widetilde a$
approximates $a$ with $\|a-\widetilde a\|_1\le\delta$. Project it onto
the types of maximum degree at most $D$; this cannot increase its
distance from $a$. Put $q'=(1+q)/2$ and require
$\delta\le(1-q)/2$, so $\|\widetilde a\|_1\le q'$.
The telescoping product identity gives

$$\|a^{*n}-\widetilde a^{*n}\|_1
\le n(q')^{n-1}\delta.\tag{33}$$

Both sides are supported on balls of degree at most $nD$. Therefore
the propagated weighted error of the entire reciprocal series is at
most

$$
\delta(r+1)^k\sum_{n\ge1}n(q')^{n-1}(1+nD)^{rk}.
\tag{34}
$$

The sum is the derivative with respect to $q'$ of the polynomially
weighted geometric series and is again exactly rational for rational
data. Choose $\delta$ so that (34) is at most half the requested error;
then choose $N$ so that (30), using $q'$ in place of $q$, is at most the
other half. The finite array $\sum_{n=0}^N\widetilde a^{*n}$ has the
claimed weighted error. An oracle controlling the stronger weighted
norm also controls the total variation required here.

These are certified sufficient conditions for a constructive inverse.
They do not make the full unit group open: imposing (27) or (28) at
every radius is an infinite collection of conditions, and the earlier
nonunits converging to the identity remain counterexamples to openness.

## 7. Scope and attribution

These are explicit geometric subalgebras and arithmetic criteria in the
existing completion. The retract does not identify the whole algebra
with a polydisc algebra, characterize all units by finitely many link
statistics, or determine arbitrary factorization. The selected graph
generators and link coordinates remain specified extrinsically; the
full intrinsic-structure question is unchanged.

Every element of $B_G$ has a finite signed graph-measure representation,
since (3) implies $\sum_n|a_n|<\infty$ and the monomials in (7) are
normalized finite graphs. The uniform-variation Neumann domain in
Section 6 also retains finite global variation. These results establish
arithmetic and geometric structure within that part of the completion;
they do not supply inverses for the previously constructed elements of
unbounded local variation.

The analytical ingredients are classical power-series and smooth
function-algebra arguments. Relevant primary sources include:

- S. J. Bhatt and S. R. Patel, [*On Fréchet algebras of power
  series*](https://repository.ias.ac.in/59672/1/9_PUB.pdf), Bulletin of
  the Australian Mathematical Society 66 (2002), 135–148, especially
  the one-variable smooth disk algebra in Example 1.5.
- H. G. Dales, S. R. Patel, and C. J. Read, [*Fréchet algebras of power
  series*](https://www.impan.pl/en/publishing-house/banach-center-publications/all/91/0/86359/frechet-algebras-of-power-series),
  Banach Center Publications 91 (2010), 123–158, for the general
  finite-variable power-series setting. No classification theorem from
  that paper is required for the direct estimates here.
- S. S. Shrikhande, [*The uniqueness of the L2 association
  scheme*](https://doi.org/10.1214/aoms/1177706207), Annals of Mathematical
  Statistics 30 (1959), 781–798, for the classical graph underlying the
  last example. Formula (21) supplies the particular spectral calculation
  used here directly.

No priority claim is made for these constructions or deductions.
