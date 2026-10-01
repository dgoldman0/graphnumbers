# Spectral measures on a proper subalgebra of the local completion

1 October 2026. Step 5 of the [development path](../../reviews/local-completion-2026-09-30/DEVELOPMENT_PATH.md).
All operators are combinatorial Laplacians with unit edge rates. Finite
graphs and relative traces are unnormalized unless divided explicitly by
vertex count. These are written proofs with targeted exact computational
fixtures; independent proof review remains deferred.

The local Cartesian graph completion has a canonical spectral-measure
homomorphism on the degree-bounded elements whose polynomial spectral
functionals are bounded in the uniform norm. A signed Hausdorff criterion
characterizes this domain exactly. It contains finite signed rooted-graph
measures, the cut-line defect, and their sums and Cartesian products.
Uniform spectral total variation permits local limits. Uniform edit
budgets alone do not: Section 5 gives an explicit counterexample, already
inside the previously constructed cycle sequence space.

This corrects D4's identification of a relative trace measure with the
Krein spectral shift. The latter pairs with a derivative; its distributional
derivative is a finite measure precisely under an additional bounded-
variation condition. The moment criterion and operator trace formulas used
here are classical. The domain identification and counterexample apply
them to this particular completion; no novelty claim is made.

## 1. Local polynomial moments

Let A_D denote the elements whose every local marginal is supported on
balls of maximum degree at most D. For a real polynomial p define

\[
 \Phi_X(p)=\Lambda_X\big((p(\Delta_G))_{oo}\big),\qquad
 m_j(X)=\Phi_X(\lambda^j).
\]

The diagonal of a fixed polynomial is a local observable of polynomial
growth in ball size. Thus these quantities are defined and continuous on
the full completion, before asserting existence of any spectral measure.
In particular m_0(X)=V(X). The finite Cartesian Laplacian identity and
continuity of each moment and multiplication give

\[
 m_j(XY)=\sum_{i=0}^j\binom ji m_i(X)m_{j-i}(Y).                 \tag{1}
\]

For completeness, approximate X and Y by finite signed graph combinations,
apply the identity to each pair, and pass to the limit in this finite sum.
Degree-filtered approximation is available from the representation theorem
when needed. No uniform bound on the approximants' coefficient mass enters
this argument. Products of A_D and A_E lie in A_{D+E}.

For D>0 put K_D=[0,2D] and a_j=m_j(X)/(2D)^j. This interval contains the
Laplacian spectrum of every graph with degree cap D. Define

\[
 b_{n,k}(X;D)=\binom nk\sum_{\ell=0}^{n-k}(-1)^\ell
                      \binom{n-k}{\ell}a_{k+\ell},
 \qquad C_n(X;D)=\sum_{k=0}^n|b_{n,k}(X;D)|.                  \tag{2}
\]

These are evaluations of the Bernstein basis
\(B_{n,k}(u)=\binom nk u^k(1-u)^{n-k}\) at u=Delta/(2D).
Use C_0=|a_0|. Each finite row is determined by finitely many local
moments; a finite row provides a lower bound, not a certificate that all
future rows stay bounded.

## 2. Exact existence and variation criterion

**Theorem 1 (signed Hausdorff criterion in local coordinates).** For
X in A_D, D>0, the following are equivalent:

1. There is a finite signed Borel measure nu_X on K_D with
   \(m_j(X)=\int\lambda^j\,d\nu_X\) for every j.
2. There is C<infinity such that
   \(|\Phi_X(p)|\le C\|p\|_{\infty,K_D}\) for every polynomial p.
3. \(\sup_n C_n(X;D)<\infty\).

The measure is unique and

\[
 \|\nu_X\|_{\rm TV}=\sup_n C_n(X;D)=\lim_{n\to\infty}C_n(X;D).
                                                                    \tag{3}
\]

The sequence C_n is nondecreasing. Moreover nu_X is positive exactly when
all the b_{n,k} are nonnegative.

**Proof.** Scale to [0,1]. A measure with variation C bounds its polynomial
functional by C. A uniformly bounded polynomial functional extends to
C[0,1] by density and is represented by a unique signed measure by the
Riesz representation theorem. For a measure, the nonnegative Bernstein
basis sums to one, so C_n<=||nu||_TV.

Conversely assume C_n<=C. The atomic measures
\(\rho_n=\sum_{k=0}^n b_{n,k}\delta_{k/n}\), n>=1, have variation at
most C. They have a weak-* convergent subsequence. For a fixed monomial
u^j, its integral against rho_n is Phi_X applied to its Bernstein
approximant. Explicitly, for n>=j,

\[
 B_n(u^j)=\sum_{\ell=0}^j
   {j\brace\ell}\frac{(n)_\ell}{n^j}u^\ell.
\]

Here the braces are Stirling numbers of the second kind. Its coefficients
converge to those of u^j. Therefore the subsequential limit has exactly
the prescribed moments. Uniqueness follows from polynomial density, and
the whole sequence rho_n converges weak-* to that measure. Weak-* lower
semicontinuity gives ||nu||_TV<=liminf C_n; the reverse inequality was
already proved.

Degree elevation gives
\(b_{n,k}=\frac{n+1-k}{n+1}b_{n+1,k}
+\frac{k+1}{n+1}b_{n+1,k+1}\).
Summing absolute values proves monotonicity. If all b_{n,k} are
nonnegative, the approximating measures are positive, with mass a_0;
their limit is positive. Necessity is immediate. QED.

This is the one-dimensional classical signed Hausdorff theorem, written
in the completion's local moments; see [1], Theorem 2.2 and Corollary 2.3.
The proof is included to specify normalization and the variation bound.

Let S_D be this domain and S=union_{D>=0} S_D. At D=0 the only elements
are c K1 and nu_{c K1}=c delta_0. Increasing a cap preserves membership
and the measure. If the same element admits measures for two caps, both
can be placed in their common larger interval, where their polynomial
moments imply equality. No assertion that every element of A_D belongs
to S_D is being made.

## 3. Convolution, heat, resolvents, and controlled limits

**Theorem 2.** S is a unital subalgebra and

\[
 \nu_{aX+bY}=a\nu_X+b\nu_Y,\qquad
 \nu_{XY}=\nu_X*\nu_Y,\qquad \nu_{K1}=\delta_0.                \tag{4}
\]

The target is the algebra of compactly supported finite signed measures
on [0,infinity), with additive convolution. For X in S_D and Y in S_E,
the product is in S_{D+E}, and

\[
 \|\nu_{XY}\|_{\rm TV}\le\|\nu_X\|_{\rm TV}\|\nu_Y\|_{\rm TV}.
\]

**Proof.** Sums have the required signed measure in a common cap. The
convolution is supported on [0,2(D+E)], has the stated variation bound,
and its moments equal (1). Apply uniqueness. QED.

In particular

\[
 H_t(X)=\int e^{-t\lambda}\,d\nu_X(\lambda),\qquad
 R_s(X)=\int\frac{d\nu_X(\lambda)}{s+\lambda}\quad(s>0),       \tag{5}
\]

and

\[
 H_t(XY)=H_t(X)H_t(Y),\qquad
 R_s(XY)=\iint\frac{d\nu_X(\lambda)d\nu_Y(\eta)}{s+\lambda+\eta}
        =\int_0^\infty e^{-st}H_t(X)H_t(Y)\,dt.              \tag{6}
\]

The resolvent itself is generally not multiplicative. Its magnitude is
bounded by ||nu_X||_TV/s. Heat is entire in complex t and is bounded by
||nu_X||_TV on Re(t)>=0. For any cap at least D,
\(|d_j(X;D)|=|\Phi_X((1-\lambda/D)^j)|\le||\nu_X||_{\rm TV}\).
Thus S lies in the finite-moment-profile heat domain, with scalar profile
(||nu_X||_TV), and (5) agrees with that domain's uniformization series.

The map is not injective: the known normalized rook–Shrikhande difference
has zero spectral measure while its local geometry is nonzero. The
spectral kernel is an ideal in S, exactly the elements with all m_j=0,
equivalently all H_t=0 for t>=0. These claims follow from compact support,
analyticity, and uniqueness of polynomial moments. A spectral response
does not determine the completed graph element or its ambient units.

**Theorem 3 (local limits with spectral variation control).** Suppose
X_n in S_D converge locally to X and ||nu_{X_n}||_TV<=C. Then X belongs
to S_D, ||nu_X||_TV<=C, and nu_{X_n} converges weak-* to nu_X.

**Proof.** Every finite expression in (2) converges, so C_k(X;D)<=C for
each k. Theorem 1 gives the measure. Polynomial convergence plus uniform
variation and uniform polynomial approximation prove convergence against
every continuous function on K_D. QED.

Heat convergence is uniform on compact complex time sets, by the uniform
exponential-series tail and moment convergence. Resolvents converge
uniformly on compact subsets of s>0, for example by polynomial
approximation and equicontinuity. Total-variation convergence need not
hold: normalized cycles tend locally to the line, their spectral measures
are atomic, and the line's spectral measure is continuous. Their total-
variation distance is two at every finite stage.

The map has no continuous extension in the unrestricted local topology:
such an extension into weak-* measures would make evaluation against
e^{-t lambda} continuous on a fixed degree cap, contradicting the
degree-two obstruction in EFFECTIVE_ANALYSIS_AND_HEAT, Section 5.
The bounded-variation hypothesis in Theorem 3 is substantive.

## 4. Included examples and the finite-edit qualification

For a finite graph G,
\(\nu_G=\sum_{\lambda\in\operatorname{spec}\Delta_G}\delta_\lambda\),
counting multiplicities. Dividing by |V(G)| gives its probability spectral
measure. More generally let X be represented by a finite signed measure
mu_X on rooted graphs of degree at most D. The root spectral theorem gives
probability measures sigma_{G,o} on K_D. Their integrals of continuous
functions are measurable, by polynomial approximation and locality. Set

\[
 \nu_X=\int\sigma_{G,o}\,d\mu_X(G,o),\qquad
 \|\nu_X\|_{\rm TV}\le\|\mu_X\|_{\rm TV}.                    \tag{7}
\]

Its moments are the local moments, proving inclusion without any assumption
about positive finite-graph approximation. Balance is required for mu_X
to represent an element of the completion; the spectral mixture itself
does not need a soficity assumption.

For the normalized line L, Fourier diagonalization gives

\[
 d\nu_L(\lambda)=\frac{\mathbf1_{(0,4)}(\lambda)}
                       {\pi\sqrt{\lambda(4-\lambda)}}\,d\lambda,
 \qquad m_j(L)=\binom{2j}{j}.                                \tag{8}
\]

For E=lim(P_n-C_n), the established local moments give

\[
 \nu_E=\tfrac12(\delta_0-\delta_4),\qquad
 \nu_{E^k}=2^{-k}\sum_{a=0}^k(-1)^a\binom ka\delta_{4a},
 \qquad\|\nu_{E^k}\|_{\rm TV}=1.                            \tag{9}
\]

Consequently E^k L^d and E^k times any degree-bounded finite rooted-measure
element have spectral measures by convolution. For example

\[
 \nu_{E\,U(P_3)}=\tfrac16(\delta_0+\delta_1+\delta_3
                              -\delta_4-\delta_5-\delta_7),
 \qquad
 R_s(E^k)=\frac{2^k k!}{\prod_{a=0}^k(s+4a)}.                 \tag{10}
\]

For k>=1, the rooted-graph variation of E^k is (4r)^k at radius r,
although its spectral variation is one. These are different measures.
Formula (9) follows from the moments of the limit, not from applying
Theorem 3 to the unnormalized P_n-C_n spectra with an unproved uniform
variation bound.

Now consider an actual finite edit on an infinite bounded-degree graph.
Let A=Delta_G, B=Delta_{G'}, V=B-A finite rank, and X its locally stabilized
relative graph element. The Krein trace formula [2], Theorem 7 and
Lemma 10, with smooth cutoffs outside the spectral interval, gives

\[
 \Phi_X(p)=\operatorname{Tr}(p(B)-p(A))
          =\int p'(\lambda)\xi(\lambda;B,A)\,d\lambda.       \tag{11}
\]

The normalized shift xi is integrable and supported in a common compact
spectral interval. The relative functional is the compactly supported
distribution -Dxi, where D is distributional differentiation. It follows
that

\[
 X\in S_D\quad\Longleftrightarrow\quad
 \xi\text{ (extended by zero) has a representative in }BV(\mathbb R),
 \qquad \nu_X=-D\xi.                                      \tag{12}
\]

Here D bounds both graph degrees. To prove necessity, if a finite measure
has the same polynomial moments as -Dxi, approximate any C^1 function
and its derivative uniformly on the common interval by polynomials
(approximate the derivative and integrate). Equality extends to smooth
test functions, so -Dxi is that measure. The elementary distributional
characterization of BV supplies (12). Sufficiency is integration by parts.
Endpoint jumps are included by extending xi by zero.

For E the convention is xi=-1/2 on (0,4), zero outside. Then -Dxi is
exactly (9), and integral xi=-2 is the deleted edge's Laplacian trace.
In general integrability of xi, finite rank, or an edit-budget moment
bound does not establish its variation. This note does not classify BV
for arbitrary single finite edits on infinite graphs. For finite linear
combinations of such edits the same criterion applies to the summed shift;
individual nonmeasure contributions could cancel. Products of measures
certified by (12) fall under Theorem 2.

## 5. A controlled edit limit with no finite spectral measure

Use the sequence-space family from REPRESENTATION_THEOREM, Section 5,
with the opposite sign. Put N_j=3^j and

\[
 X_M=\sum_{j=1}^M\left(U(C_{N_j})-U(C_{2N_j})\right),
 \qquad X=\lim_M X_M.                                      \tag{13}
\]

At fixed radius r, every summand with N_j>2r+1 vanishes exactly. Hence
the limit exists in A_2, including every weighted local seminorm.

**Uniform edit certificate.** On 2N vertices, turn C_{2N} into two copies
of C_N by deleting (N-1,N),(2N-1,0), and inserting (N-1,0),(2N-1,N).
For N>=3 these are four distinct valid edits; both endpoint graphs have
degree two. Dividing their difference by 2N gives the summand in (13),
with budget 2/N. Therefore

\[
 q_M=\sum_{j=1}^M2/3^j=1-3^{-M}<1,\qquad
 X\in K_{2,1},\qquad |d_k(X;2)|\le k.                       \tag{14}
\]

This is a finite-profile element with profile (0,2). Its relative heat
is well defined and is the limit of the finite partial sums. Indeed
each summand has nonnegative heat by the cycle double-cover argument,
and the existing edit bound gives 0<=H_t(X)<=min(2t,1).

**No spectral measure.** Let T_l denote the Chebyshev polynomial and
put p_l(lambda)=T_l(1-lambda/2). Its uniform norm on [0,4] is one.
The cycle eigenvalues give, for l>=1,

\[
 \Phi_{U(C_n)}(p_l)=\frac1n\sum_{a=0}^{n-1}
                         \cos(2\pi al/n)=\mathbf1_{n\mid l}.
\]

At l=3^m the first m positive cycle terms each contribute one, all
negative terms contribute zero, and all later terms contribute zero.
Thus

\[
 \Phi_X(p_{3^m})=m,\qquad\|p_{3^m}\|_{\infty,[0,4]}=1.       \tag{15}
\]

Theorem 1 excludes a finite signed measure on [0,4]. This is not repaired
by increasing the cap: each summand defines a functional bounded by
(4/N_j)||f'||_infinity on C^1[0,4]. One way to see that bound is to
interpolate its two finite Laplacians and use cyclicity of trace for
polynomials; the trace norm of the scaled difference is at most 4/N_j.
The sum defines a distribution of order at most one supported on [0,4].
If a finite measure on any larger compact interval had its moments,
C^1 polynomial approximation would identify that measure with this
distribution. Its support would then be in [0,4], contradicting (15).

Consequently S is a **proper** subalgebra of the finite-profile heat
domain. Even the uniform-edit closure K_{2,1} is not contained in S.
The partial sums' spectral variations satisfy ||nu_{X_M}||_TV>=M by the
same polynomial p_{3^M}, explaining the failed hypothesis in Theorem 3.
This example is a controlled limit of finite edit combinations; it is
not asserted to be a single finite-edge perturbation on one infinite
graph. It also does not decide whether the moment-profile domain strictly
contains the edit-controlled domain, a different inclusion question.

## 6. The interaction's eventual sign from its lowest atom

For the five-vertex fixture in HIGHER_INTERACTION_GEOMETRY, Section 6,
take edges 01,02,03,04,12 and selected deletions 01,03,04. Let I be the
three-fold alternating deletion sum, with coefficient (-1)^{3-|S|}.
Exact characteristic-polynomial factorization of its eight graphs gives

\[
 \nu_I=\sum_{q(\alpha)=0}\delta_\alpha
       -2\delta_{2-\sqrt2}-2\delta_{2+\sqrt2}
       +2\delta_1-2\delta_2+2\delta_4-\delta_5,
 \quad q(z)=z^3-7z^2+13z-5.                                \tag{16}
\]

The zero atom cancels. The three roots of q lie respectively in
(0.5188,0.5189), (2,3), and (4,5), as follows from rational endpoint
signs and the degree. The first, alpha, is below 2-sqrt(2), the next
atom. Its coefficient is +1. All atoms are distinct and ||nu_I||_TV=14.
The gap is greater than 0.066, so

\[
 H_t(I)\ge e^{-\alpha t}\left(1-13e^{-0.066t}\right)>0
 \quad(t\ge40).                                           \tag{17}
\]

The final strict inequality follows already from the rational lower
bound \(\sum_{j=0}^{12}(66/25)^j/j!>13\). Existing positive heat at
t=1 and negative heat at t=3 are independently re-enclosed below.
Together with (17) they prove at least two positive-time zeros: one
between 1 and 3 and another between 3 and 40. No claim of exactly two
zeros is made. The lowest-atom argument requires an isolated atom and
a positive gap; it is not a general sign rule for continuous measures.

## 7. Reproduction and limits of the evidence

Run from the repository root:

```sh
python -S research/local-completion/verify_spectral_measure.py
```

The standard-library-only verifier imports no graphlocal code. Its
[result record](spectral_measure_results.json) identifies fixtures and
executed horizons. It compares finite moment-derived Bernstein rows with
direct matrix-polynomial evaluation, checks explicit Cartesian products
against the binomial law, and compares stabilized path/cycle defects with
the atomic formulas. The line's Bernstein rows are compared with their
closed beta-integral formula. Exact cycle adjacency/Chebyshev recurrence
checks (15) for m=1,2,3, and explicit four-edge rewiring checks (14).
Characteristic polynomials for (16) are calculated by both permutation
determinants and Newton identities; rational root brackets and independent
matrix-based heat intervals certify the sign conclusions.

Finite prefixes do not prove the infinite Hausdorff bound, BV membership,
or a universal classification. Those claims use the proofs and explicit
formulas above. This checkpoint changes no package API or version.
Independent proof review and the unrelated audit backlog remain open.

## References checked for this checkpoint

1. Oliver Knill, *On Hausdorff's moment problem in higher dimensions*,
   author preprint dated 4 January 2000, Theorem 2.2 and Corollary 2.3.
   [Author PDF](https://people.math.harvard.edu/~knill/preprints/stability.pdf).
   These state the classical signed-measure criterion and Bernstein
   reconstruction. They are prior art for Theorem 1's moment-theoretic part.
2. Denis Potapov, Fedor Sukochev, Dmitriy Zanin, *Krein's trace theorem
   revisited*, Journal of Spectral Theory 4 (2014), 415–430,
   [doi:10.4171/JST/75](https://doi.org/10.4171/JST/75),
   [publisher PDF](https://ems.press/content/serial-article-files/33519).
   Theorem 7 states the trace formula with f'; Lemma 10 gives the bounded
   operator case. Equation (11) uses this theorem, and (12) is the
   distributional BV consequence proved here.
