# Separating characters and unweighted local inversion

30 September 2026. Continuation of items 1 and 2 in the research plan.
The two-generator and derivation questions are deferred.
This note uses the same algebra, topology, and finite-graph embedding as
[the multiplication note](MULTIPLICATION_AND_UNITS.md).

The complexification is denoted by $A=\mathcal A_{\mathrm{loc},\mathbb C}$.
Complex arrays satisfy the same summability, compatibility, and balance
conditions, by applying the real representation theorem to real and
imaginary parts. Real inverses obtained below remain real by uniqueness.

## Results

1. An explicit family of continuous characters separates every element
   of $A$. Thus the Gelfand transform is injective and the Jacobson radical
   is zero, including on elements outside the finite signed-measure model.
2. At each radius, all polynomially weighted convolution Banach algebras
   have the same characters and spectra as unweighted $\ell^1$.
3. An element of $A$ is invertible exactly when its marginal at every
   positive radius is invertible in unweighted $\ell^1$. Equivalently,
   its formal local reciprocals need only be absolutely summable;
   their polynomial moments then follow automatically.
4. Bounded local semicharacters give an exact spectral and unit test.
   The separating phase characters alone do not suffice for that test.

## 1. Explicit characters and coefficient recovery

Fix $r\ge1$ and enumerate the rooted multigraph patterns of root
eccentricity at most $r$ as $F_1,F_2,\ldots$. The additive invariants
$K_{F_j}$ constructed in the multiplication note separate the monoid
$M_r=(\mathcal B_r,\star_r)$.

For $m\ge1$ and $t\in\mathbb R^m$, put

$$
s_{r,m,t}(B)=\exp\left(i\sum_{j=1}^m t_jK_{F_j}(B)\right),\qquad
\chi_{r,m,t}(X)=\sum_B a_r^X(B)s_{r,m,t}(B).
\tag{1}
$$

Additivity gives $s(B\star_r D)=s(B)s(D)$ and $s(e)=1$.
Moreover $|s(B)|=1$. Absolute convergence justifies rearranging the
convolution sums, proving that $\chi$ is multiplicative and unital.
It is continuous, since

$$|\chi_{r,m,t}(X)|\le\|a_r^X\|_1\le p_{r,1}(X).$$

**Separation theorem.** If all the characters (1) vanish on $X$, then
$X=0$.

**Proof.** Fix a radius and a target ball $B_0$. Write
$v_m(B)=(K_{F_1}(B),\ldots,K_{F_m}(B))$. Average the character after
removing the phase of $B_0$:

$$
\frac1{(2T)^m}\int_{[-T,T]^m}
 e^{-i\langle t,v_m(B_0)\rangle}\chi_{r,m,t}(X)\,dt
=\sum_B a_r^X(B)\prod_{j=1}^m
 \operatorname{sinc}\big(T(K_{F_j}(B)-K_{F_j}(B_0))\big),
\tag{2}
$$

where $\operatorname{sinc}(u)=\sin(u)/u$ and its value at zero is one.
Fubini is justified by $\sum_B|a_r^X(B)|<\infty$.
As $T\to\infty$, dominated convergence gives

$$\sum_{B:v_m(B)=v_m(B_0)}a_r^X(B).$$

As $m\to\infty$, the matching sets decrease to $\{B_0\}$ because the
invariants separate balls. A second dominated-convergence argument gives
$a_r^X(B_0)$. If all characters vanish, this coefficient is zero.
Every positive-radius marginal is therefore zero, and faithfulness gives
$X=0$. This also supplies an iterated-limit coefficient recovery formula.

The kernels of these characters are maximal ideals, since the characters
are unital complex-linear maps onto $\mathbb C$. Their intersection is
zero, so the intersection of all maximal ideals is zero as well.
This is algebraic semisimplicity, proved by continuous characters.

Only one absolutely summable local array was used at a time. A global
measure and a uniform bound on total variation across radii are unnecessary.
The result gives an injective algebraic transform into functions on the
displayed character family. It does not identify the original Fréchet
topology with pointwise convergence or classify all continuous characters.

## 2. Polynomial weights do not change the local spectrum

Put

$$E_{r,k}=\ell^1(M_r,w_k),\qquad w_k(B)=|B|^k,\qquad k\ge0,$$

where $E_{r,0}=\ell^1(M_r)$. Each is a commutative unital complex Banach
algebra. Recall $L_r=\bigcap_{k\ge1}E_{r,k}$.

**Growth lemma.** For fixed $B\in M_r$,

$$\lim_{n\to\infty}w_k(B^{\star_r n})^{1/n}=1.\tag{3}$$

Indeed, if $\Delta$ is the maximum degree of $B$, its $n$-fold Cartesian
power has maximum degree at most $n\Delta$. Truncation commutes with the
product, so

$$
|B^{\star_r n}|\le\sum_{j=0}^r(n\Delta)^j
\le(r+1)(1+n\Delta)^r.
\tag{4}
$$

The lower bound is one; taking $n$th roots proves (3). This is the
subexponential power-growth condition underlying the following argument.

Let

$$
\Sigma_r=\{s:M_r\to\overline{\mathbb D}:
s(e)=1,\ s(B\star_r D)=s(B)s(D)\}.
$$

These are bounded semicharacters; zero values away from $e$ are allowed.

**Local character theorem.** For every $k\ge0$, the characters on $E_{r,k}$
are exactly

$$\varphi_s(a)=\sum_B a(B)s(B),\qquad s\in\Sigma_r.\tag{5}$$

**Proof.** Such a sum is absolutely convergent, continuous, and
multiplicative. Conversely, a character $\varphi$ on a unital Banach
algebra has $|\varphi(a)|\le\|a\|$. Set $s(B)=\varphi(\delta_B)$.
Multiplicativity and (3) give

$$
|s(B)|^n=|\varphi(\delta_{B^{\star_r n}})|
\le w_k(B^{\star_r n}),\qquad |s(B)|\le1.
$$

Density of finite arrays in $E_{r,k}$ now gives (5).

The set $\Sigma_r$ is compact in the product topology: it is the closed
set defined by the displayed multiplicative equations in
$\overline{\mathbb D}^{M_r}$. For $a\in\ell^1(M_r)$, its transform
$\widehat a(s)=\sum_Ba(B)s(B)$ is continuous by uniform convergence.
The usual commutative Banach-algebra character criterion consequently gives

$$
\sigma_{E_{r,k}}(a)=\{\widehat a(s):s\in\Sigma_r\}
=\sigma_{E_{r,0}}(a),\qquad a\in E_{r,k}.
\tag{6}
$$

In particular, $E_{r,k}$ is inverse-closed in $E_{r,0}$. If $a\in L_r$
has an inverse in $E_{r,0}$, it has an inverse in every $E_{r,k}$.
Uniqueness in $E_{r,0}$ makes all those inverses the same array, which
therefore belongs to $L_r$.

## 3. The sharpened global unit theorem

For $X\in A$, the following conditions are equivalent:

1. $X$ is a unit of $A$.
2. $a_r^X$ is a unit of $\ell^1(M_r)$ for every $r\ge1$.
3. $I(X)\ne0$ and the recursively defined formal reciprocals $b_r$ obey
   $\sum_B|b_r(B)|<\infty$ for every $r\ge1$.
4. $\widehat{a_r^X}(s)\ne0$ for every $r\ge1$ and $s\in\Sigma_r$.

**Proof.** A global inverse maps to a local inverse. Conversely, (2)
and local inverse-closedness put every reciprocal in all the weighted
spaces. The original unit theorem then supplies compatibility and
mass-transport balance, and hence a global inverse. Evaluation at the
local unit $e$ forces $I(X)\ne0$; uniqueness of formal inversion proves
the equivalence with (3). Formula (6) proves equivalence with (4).

Thus the additional weighted tests in the original reciprocal criterion
are automatic once an element already belongs to $A$ and its reciprocals
are locally in unweighted $\ell^1$. Membership of $X$ still requires all
the original polynomial moments.

Each radius has its own positive minimum
$\min_{s\in\Sigma_r}|\widehat{a_r^X}(s)|$ when $X$ is a unit.
No radius-uniform lower bound is required. The theorem remains an
infinite criterion and gives no finite decision procedure or effective
bound for the weighted norm of the reciprocal.

There is also an exact spectral formula:

$$
\boxed{\sigma_A(X)=
\bigcup_{r\ge1}\{\widehat{a_r^X}(s):s\in\Sigma_r\}.}
\tag{7}
$$

Apply the unit equivalence to $\lambda1-X$. Truncation makes the compact
sets on the right increasing with $r$. Their union need not be treated
as a single compact Banach-algebra spectrum.

**Separation versus detection of units.** For $H=K_2/2$, every phase
character (1) has $|\chi(H)|=1$. Thus none vanishes on $1-2H$, since
$|1-2\chi(H)|\ge1$. Nevertheless $1-2H$ is a nonunit: the degree
character with parameter $z=1/2$ annihilates it. Bounded semicharacters
with values inside the disk are essential to the complete unit test.

## 4. Attribution and scope

The argument uses standard Fourier averaging, Banach-algebra character
theory, and the inverse-closedness principle associated with subexponential
weights. The local growth estimate (4) verifies the needed condition for
this particular graph monoid; Sections 2–3 include the application in full.

- Abolghasemi–Rejali–Vishki,
  [*Weighted Semigroup Algebras as Dual Banach Algebras*, Section 2](https://arxiv.org/abs/0808.1404),
  supplies the weighted convolution setting.
- Gröchenig–Leinert,
  [*Symmetry and inverse-closedness of matrix algebras and functional
  calculus for infinite matrices*, Section 2.1](https://homepage.univie.ac.at/karlheinz.groechenig/preprints/inverselast.pdf),
  states the Gelfand–Raikov–Shilov condition in its group/matrix setting.
  Its matrix theorem is not being applied without its hypotheses here.
- Pedersen,
  [*A class of weighted convolution Fréchet algebras*, Theorem 2.3 and
  Corollary 2.4](https://arxiv.org/abs/0909.2749),
  illustrates character classification and semisimplicity in a different
  convolution Fréchet algebra.

No classification of every character or maximal ideal of $A$ is claimed.
The displayed local family does separate elements and detect all units,
which suffices for the stated results. Priority remains unestablished.

The [combined verifier](verify_spectral_approximation.py) checks exact
finite Fourier inversion and multiplicativity over fourth roots of unity,
the polynomial power-growth estimate on direct Cartesian products, and
the distinction between separating phases and interior spectral zeros.
Its [335 recorded checks](spectral_approximation_results.json) also cover
the separate quantitative approximation note. Infinite separation and
inverse-closedness follow from the proofs above, not from those fixtures.
