# Effective inversion from local residual certificates

30 September 2026. This note removes the uniform variation and degree
hypotheses from the **computability theorem for inversion on its unit
domain**. The practical certificate accepts a supplied finite local inverse
candidate. The exhaustive search below proves computability; it is not a
claim that the library implements that search efficiently, or at all.

Write \(A=\mathcal A_{\mathrm{loc}}\). The
[global unit theorem](CHARACTERS_AND_INVERSION.md) identifies its units by
invertibility of every positive-radius marginal. Its proof, including
mass-transport balance of the inverse, is in
[Multiplication and units, Section 4](MULTIPLICATION_AND_UNITS.md).
The estimates below are direct Banach-algebra residual and Neumann-series
arguments, specialized to the effective local-array representation.

## 1. The weighted local Banach algebra

Fix a radius \(r\ge0\) and an integer weight \(k\ge1\). Let

\[
E_{r,k}=\ell^1(M_r,w_k),\qquad
w_k(B)=|B|^k,\qquad
\|a\|_{r,k}=\sum_B |a(B)|\,|B|^k.
\tag{1}
\]

Here \(M_r\) is the monoid of rooted radius-\(r\) graph types under
truncated Cartesian product, and \(e\) denotes the one-vertex rooted
type. Its point mass \(\delta_e\) is the algebra unit. At radius zero
this is just the scalar algebra. The bound

\[
|B\star_r C|\le |B|\,|C|
\]

gives submultiplicativity of (1), and \(\|\delta_e\|_{r,k}=1\).
The map \(T_r:A\to E_{r,k}\) is a continuous unital homomorphism.

A finite rational local array is a finite rational combination of rooted
types. Such arrays are dense in \(E_{r,k}\). They need not individually
be marginals of globally balanced graph elements.

All norms and operations on finite arrays are exactly computable with
rational arithmetic and decidable finite rooted-graph isomorphism.
Finite work limits in an implementation may interrupt these calculations;
the computability assertions below allow the necessary finite work.

## 2. A finite certificate of local invertibility

Let \(a=T_rX\), and suppose an approximation oracle supplies a finite
rational array \(a_0\) and a rational bound \(\delta\ge0\) satisfying

\[
\|a-a_0\|_{r,k}\le\delta.
\tag{2}
\]

Supply any finite rational local candidate \(b\), and calculate

\[
B=\|b\|_{r,k},\qquad
e_0=\delta_e-a_0*b,\qquad
q=\|e_0\|_{r,k}+\delta B.
\tag{3}
\]

**Residual certificate theorem.** If \(q<1\), then \(a\) is a unit
of \(E_{r,k}\), and

\[
\boxed{\|a^{-1}\|_{r,k}\le \frac{B}{1-q}},\qquad
\boxed{\|a^{-1}-b\|_{r,k}\le \frac{Bq}{1-q}}.
\tag{4}
\]

In particular, if the second bound is at most a requested \(\epsilon\),
the candidate itself is a certified approximation to the inverse marginal.

**Proof.** Set \(e=\delta_e-a*b\). Submultiplicativity and (2) give
\(\|e\|_{r,k}\le q<1\). The series

\[
c=\sum_{j=0}^{\infty}e^{*j}
\]

converges in \(E_{r,k}\) and inverts \(\delta_e-e=a*b\).
Commutativity then gives \(a^{-1}=b*c\). Thus
\(\|a^{-1}\|\le B/(1-q)\), and

\[
a^{-1}-b=b*\sum_{j=1}^{\infty}e^{*j}
\]

has norm at most \(Bq/(1-q)\). In particular a successful certificate
has \(B>0\). No global variation or degree bound appears in the proof.

The certificate also works for an input known only as an element of this
one local Banach algebra. Its immediate conclusion is local invertibility.
Claiming that it represents \(T_r(X^{-1})\) uses the additional assertion
that \(X\) is a unit of the whole graph algebra.

## 3. Refining a candidate to arbitrary precision

A successful certificate with \(q<1\) can be refined without another
search for candidates. Put

\[
M=\frac{B}{1-q},\qquad \alpha=\frac{1+q}{2}<1.
\tag{5}
\]

For a requested error \(\epsilon>0\), obtain a new finite approximation
\(a_1\) whose error \(\delta_1\) satisfies

\[
\delta_1\le
\min\left\{\frac{1-q}{4B},\,
\frac{\epsilon}{4M^2}\right\}.
\tag{6}
\]

Define \(e_1=\delta_e-a_1*b\). The original certificate bounded
\(\|\delta_e-a*b\|\) by \(q\), so (6) gives

\[
\|e_1\|_{r,k}\le q+\delta_1B\le\alpha.
\tag{7}
\]

The quarter-margin choice also certifies
\(\|e_1\|+\delta_1B\le\alpha<1\) for the refreshed full residual.
The second quantity in (6) equals
\(\epsilon(1-\alpha)/(2MB)\), using (5).

Choose \(N\ge0\) with \(M\alpha^{N+1}\le\epsilon/2\), and compute
the finite rational array

\[
b_N=b*\sum_{j=0}^{N}e_1^{*j}.
\tag{8}
\]

Then

\[
\boxed{\|a^{-1}-b_N\|_{r,k}\le\epsilon.}
\tag{9}
\]

Indeed, \(\|b_N\|\le B/(1-\alpha)\), and the exact finite identity
\(\delta_e-a_1*b_N=e_1^{*(N+1)}\) implies

\[
\begin{aligned}
\|a^{-1}-b_N\|
&\le\|a^{-1}\|\,\|\delta_e-a*b_N\|\\
&\le M\left(\alpha^{N+1}+\delta_1\frac{B}{1-\alpha}\right)
\le\epsilon.
\end{aligned}
\tag{10}
\]

Every factor in (8) is a finite array. The required \(N\) can be found
using rational repeated multiplication, without logarithmic rounding.
Directly evaluating the finite norms of \(e_1^{*(N+1)}\) and \(b_N\)
can sharpen the bound in (10).

The practical refinement can use the measured
\(q_1=\|e_1\|\) in place of \(\alpha\). Put
\(M_1=B/(1-q_1)\le2M\). The perturbation error between the inverse of
\(a\) and that of \(a_1\) is at most
\(\delta_1MM_1\le\epsilon/2\), while the finite Neumann tail is
\(Bq_1^{N+1}/(1-q_1)\). Choosing the tail at most
\(\epsilon/2\) is therefore sufficient. This gives the same certified
output with a potentially smaller truncation degree.

For comparison, using a fixed approximation \(a_0\), set
\(q_0=\|e_0\|<1\) and
\(y_N=b*\sum_{j=0}^N e_0^{*j}\). The alternative bound

\[
\|a^{-1}-y_N\|
\le
\frac{\delta B^2}{(1-q)(1-q_0)}
+\frac{Bq_0^{N+1}}{1-q_0}
\tag{11}
\]

separates the error in the source from the Neumann tail. It follows by
inverting \(a_0\), applying
\(a^{-1}-a_0^{-1}=a^{-1}*(a_0-a)*a_0^{-1}\), and truncating the
Neumann series for \(a_0^{-1}\). Increasing \(N\) alone does not remove
the first term; (6) supplies the needed source refinement.

## 4. Inversion is computable on the promised unit domain

An effective name for \(X\in A\) is an algorithm which, for every
\((r,k,\eta)\) with rational \(\eta>0\), returns a finite rational
array \(a_0\) and a valid bound
\(\|T_rX-a_0\|_{r,k}\le\delta\le\eta\).
The algorithm must terminate on each such request. This is the mathematical
content of the library's local-approximation protocol; a fixed work budget
may prevent a particular implementation from realizing all requests.

**Effective inverse theorem.** Given such a name and the promise that
\(X\) is a unit of \(A\), there is a uniform algorithm producing an
effective name for \(X^{-1}\). No uniform bound over radii on variation,
degree, inverse norms or spectral gaps is required.

**Proof.** Fix a requested \((r,k,\epsilon)\). Effectively enumerate
all finite rational arrays \(b_1,b_2,\ldots\) on \(M_r\). One way is
to enumerate finite labeled rooted simple graphs, retain connected graphs
of root eccentricity at most \(r\), and enumerate finite lists of these
graphs with rational coefficients. Duplicate descriptions do not matter.
Each stage of the enumeration is finite, and every finite rational array
eventually appears.

At stage \(s\ge1\), obtain an approximation \(a_s\) with error
\(\delta_s\le2^{-s}\). For the first \(s\) candidates, compute

\[
q_{s,j}=\|\delta_e-a_s*b_j\|_{r,k}
+\delta_s\|b_j\|_{r,k}.
\tag{12}
\]

Stop and return \(b_j\) as soon as

\[
q_{s,j}<1,\qquad
\frac{\|b_j\|_{r,k}q_{s,j}}{1-q_{s,j}}\le\epsilon.
\tag{13}
\]

Every returned array is valid by (4). To prove termination, the unit
promise gives \(a^{-1}\in E_{r,k}\), where \(a=T_rX\). Choose a
finite rational \(b_j\) close enough to \(a^{-1}\). Density and

\[
\|\delta_e-a*b_j\|\le\|a\|\,\|a^{-1}-b_j\|
\tag{14}
\]

make this residual arbitrarily small, while
\(\|b_j\|\le\|a^{-1}\|+\|a^{-1}-b_j\|\) stays bounded. We can
therefore arrange (13) with strict margin for the true residual.
For this fixed candidate,

\[
\|\delta_e-a*b_j\|
\le q_{s,j}
\le\|\delta_e-a*b_j\|+2\delta_s\|b_j\|.
\tag{15}
\]

Thus sufficiently large \(s\ge j\) satisfies (13). Each individual
stage terminates using finite exact arithmetic and the source oracle, so
the entire search terminates.

Alternatively, stop the search at any certificate with \(q<1\), cache
the local bound (5), and apply Section 3 to all later precision requests
at that radius and weight. This also provides a computable local bound
for the inverse norm.

This is a computability result on a promised domain. It supplies no
useful general complexity bound for the exhaustive search. A practical
implementation can accept candidates from a user, a special construction,
formal reciprocal truncation, or a numerical search, and apply the same
exact residual test. A failed candidate is not evidence of noninvertibility.

## 5. Why local candidates need no balance constraint

The finite candidates in Sections 2–4 need not be coherent across radii
or satisfy mass-transport balance. They are finite approximations in the
ambient Banach spaces, as permitted by `LocalHistogram` and
`LocalApproximation`.

For a genuine unit \(X\), uniqueness identifies every certified local
target with \(T_r(X^{-1})\). Truncation is a continuous unital
homomorphism, so it sends an inverse at a larger radius to the unique
inverse at a smaller radius. Changing the weight gives the same array by
uniqueness in unweighted \(\ell^1\). Exact inverse targets are therefore
coherent even when their finite approximants differ.

More generally, suppose inverses are established at every positive radius
in unweighted \(\ell^1\). The earlier character theorem makes each
weighted local algebra inverse-closed, and the global unit theorem then
supplies compatibility, all weighted moments, and mass-transport balance.
The latter uses the formal reciprocal and degree-truncated transports; it
does not assume that arbitrary independently specified local arrays are
globally realizable.

The local Banach-algebra certificate alone does not replace this global
theorem. In particular, imposing balance on each search candidate would
be unnecessary and could obstruct the simple density argument.

## 6. Local success is compatible with global noninvertibility

The cut-line defect gives an exact quantitative example. The
[entire defect arithmetic theorem](UNBOUNDED_VARIATION_INVERSION.md)
proves, for \(E=\lim(P_n-C_n)\) and \(r\ge2\),

\[
1-tT_rE\text{ is locally invertible}
\quad\Longleftrightarrow\quad 4r|t|<1,
\qquad
\|(1-tT_rE)^{-1}\|_1=\frac1{1-4r|t|}
\tag{16}
\]

when this inequality holds. The local inverse has every polynomial
weighted moment by inverse-closedness. At any fixed radius choose a
nonzero \(t\) below that threshold: the dense search in Section 4
terminates for that local request, with every weight and tolerance.
At sufficiently large radius the local spectrum obstructs inversion.
Consequently \(1-tE\) is a global nonunit for every \(t\ne0\).

A separate example shows exact agreement with the identity at any fixed
finite collection of observation radii.
For odd \(N\ge3\), the earlier nonunit construction is

\[
X_N=1-\tfrac12\bigl(U(C_{2N})-U(C_N)\bigr),\qquad
U(G)=G/|V(G)|.
\tag{17}
\]

For every fixed observation radius \(R\) with \(N>2R+1\), both
normalized cycles have the same \(R\)-ball histogram. Hence
\(T_RX_N=\delta_e\), and the candidate \(b=\delta_e\) passes (3)
with \(q=0\) at every weight. Nevertheless \(X_N\) is a nonunit:
the odd closed-walk cumulant character constructed in
[Multiplication and units, Section 6](MULTIPLICATION_AND_UNITS.md)
annihilates it.

Thus a successful finite collection of local residual tests does not
certify global invertibility. Without the unit promise, the exhaustive
procedure can terminate on some requested radii and fail to terminate on
another. With the promise, it yields all inverse approximations. There is
no conflict with the fact that the full unit group is not open: inversion
is continuous, and here computable, on that group with its relative
topology.

There are also explicit inputs to the effective inverse theorem beyond
uniform variation: \(\exp(tE)\) is a global unit with inverse
\(\exp(-tE)\), and both have local total variation
\(e^{4r|t|}\) for \(r\ge2\). Their specialized factorial-tail oracle
and the general residual certificates therefore concern actual units
outside the finite signed graph-measure model.

## 7. Implementation scope

Version 0.7.0 exposes `local_inverse_certificate` and
`refine_local_inverse` in the [library](../../python/README.md). They report
the finite candidate's norm, source precision, residual bound (3), local
inverse norm bound (4), and a resulting `LocalApproximation`. Refinement
uses the quarter-margin precision in (6) and the measured-residual
stability and tail bounds following (10). Such a certificate requires no
`variation_bound` or `degree_bound` metadata on the source element.

The theoretical enumeration in Section 4 is separate from that API. No
automated exhaustive local-array enumerator or finite global-unit decision
procedure is asserted. Graph-size and isomorphism work limits remain
computational limits, not mathematical evidence of noninvertibility.
