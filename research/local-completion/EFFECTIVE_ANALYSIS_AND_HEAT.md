# Effective local calculus and certified heat return

30 September 2026. The accompanying implementation is the independent Python
package [graphnumbers-local](../../python/README.md), imported as `graphlocal`.
It packages a computable part of the existing completion and develops one
application with explicit accuracy contracts.

The subsequent [sparse-defect note](SPARSE_DEFECTS_AND_RELATIVE_HEAT.md)
gives another controlled domain: an edit budget can replace global
variation for relative heat. This includes the cut-line defect beyond
finite signed measures. It does not change the obstruction to a continuous
heat functional on the unrestricted completion proved below.

## 1. What a finite computation represents

For an effective element X, `approximate(r, k, epsilon)` returns a finite
rational rooted histogram b and a rational eta with

$$\|T_rX-b\|_{1,k}\le\eta\le\varepsilon.$$

The output is a certificate about one radius and weight. A local histogram
alone does not establish coherence at all radii or mass-transport balance,
and does not uniquely identify a completed element. `Finite`, `Line`, and
their algebraic and exponential expressions specify actual elements. A
user-defined `Element` must supply correct mathematical bounds.

Finite input coefficients are rational. Integers, fractions and decimal
strings specify exact values; binary floating-point inputs are rejected.
This implementation does not represent every real coefficient or every
element of the completion. Computable limits include the infinite line,
its Cartesian products, and entire exponentials of supported expressions.

The local product is computed directly on pairs (u,v) whose distances from
the respective roots sum to at most r. The full Cartesian product need
not be constructed. Rooted isomorphism is decided exactly by refinement
and backtracking; work limits report `BudgetExceeded`.

If approximate inputs have centers b,c and errors eta,theta in one
defining seminorm, the product error is at most

$$p(b)\theta+p(c)\eta+\eta\theta.$$

Addition and scalar multiplication use eta+theta and |s|eta. Truncating
radius or decreasing the weight exponent preserves the error bound.
For a local observable f with |f(B)| <= C|B|^k, evaluation has error
at most C eta. The supplied observables include vertex, edge and isolated
vertex counts, and fixed-length closed-walk counts.

For exponentials, if p(X),p(Y) <= B, telescoping the powers gives

$$p(e^X-e^Y)\le e^B p(X-Y).$$

The implementation first obtains a rational upper bound for e^(B+1),
approximates the input sufficiently closely, then truncates the algebra
exponential. If z bounds the seminorm of the approximate input and
N+2 > z, its remaining series has bound

$$\sum_{j>N}\frac{z^j}{j!}
\le \frac{z^{N+1}}{(N+1)!}\frac1{1-z/(N+2)}.$$

Both contributions are included in the returned certificate. Polynomial
and exponential directional derivatives use the previously proved
commutative-algebra formulas. General division and effective unit
recognition remain outside the package.

Finite reconstruction uses the existing exact catalog LP, with a primal
graph combination and matching dual certificate. For example the
radius-two histogram of the line is realized by P5-P4 with minimum
coefficient mass nine on the catalog {P4,P5}; allowing C6 reduces the
minimum to one. An infeasibility certificate concerns the supplied
catalog. A work-limit exception carries no infeasibility claim.

## 2. The domain of the heat application

Let M_D be the elements represented by finite signed measures on rooted
connected graphs with maximum degree at most D. Such measures have every
required local polynomial moment, since each r-ball has uniformly bounded
size. They must also satisfy the balance condition to belong to A.
Suppose the global total variation is bounded by C. The software requires
certified D and C as input metadata. For finite combinations,

$$C=\sum_G |c_G|\,|V(G)|$$

is an available upper bound; the line has D=2 and C=1. Sums and products
propagate these bounds conservatively. A product of degree-bounded
elements has degree bound D1+D2 and variation bound C1 C2.

Write Delta_G = diag(deg)-A_G, so each edge has continuous-time jump rate
one. Define

$$\mathcal H_t(X)=\int (e^{-t\Delta_G})_{oo}\,d\mu_X(G,o),\qquad t\ge0.$$

For a normalized finite graph U(G)=G/|V(G)| this is

$$\mathcal H_t(U(G))=\frac1{|V(G)|}\operatorname{tr}(e^{-t\Delta_G}).$$

It is the mean probability of returning to the starting vertex at time t.
For an unnormalized graph it is the unnormalized trace. Signed inputs
give signed linear combinations of these quantities. This graph heat
operator and the entire algebra function `exp(X)` are separate operations.

For D>0 put P_G=I-Delta_G/D and lambda=tD. P_G is a symmetric stochastic
matrix (or a bounded operator on a degree-bounded infinite graph), and

$$e^{-t\Delta_G}=e^{-\lambda}\sum_{j\ge0}\frac{\lambda^j}{j!}P_G^j.$$

This is standard uniformization. For a primary modern treatment see
Rao and Teh, *Fast MCMC Sampling for Markov Jump Processes and Extensions*,
JMLR 14 (2013), Section 3.1,
https://jmlr.org/papers/volume14/rao13a/rao13a.pdf.
The simulation algorithm in that paper is not implemented here. We use
the uniformization identity for deterministic rational enclosures.

When D=0 the heat functional is V(X). The same identity with the zero
Poisson parameter also handles t=0.

## 3. Exact locality through order 2R

**Lemma.** For a rooted graph of maximum degree at most D, and B its
induced R-ball,

$$ (P_G^j)_{oo}=(P_B^j)_{oo}\quad(0\le j\le2R).$$

Here P_B uses the degrees in the induced ball, so its boundary holding
probabilities can differ from those in G.

**Proof.** A contributing walk starts and ends at the root. Leaving the
ball and returning requires at least 2R+2 steps. The only other possible
discrepancy is a holding step at distance R, where the induced degree
may be smaller. Reaching that vertex, holding, and returning requires
at least 2R+1 steps. Thus neither discrepancy occurs through order 2R.
All remaining transition weights agree. This also covers R=0. QED.

The boundary is sharp: on a long cycle with D=2, the radius-one induced
path creates holding probabilities at its endpoints. Its third return
probability differs from that of the original cycle, while the first
three moments (orders zero, one and two) agree.

Consequently M+1 retained Poisson terms, with indices 0 through M, require
only R=ceil(M/2). In the implementation P_B^j is evaluated by sparse
integer-vector updates for D P_B, divided by D^j.

## 4. Rational enclosure, including local input error

Let

$$S=\sum_{j=0}^M\lambda^j/j!,\qquad
T=\frac{\lambda^{M+1}}{(M+1)!}\frac1{1-\lambda/(M+2)},$$

where M+2>lambda. Then e^lambda=S+E with 0<=E<=T.
Let A be the retained numerator obtained by integrating the diagonal
powers against the true radius-R marginal.

For positive mass m, 0<=A<=mS. The omitted numerator lies between 0 and
mE. Therefore

$$\frac A{S+T}\le\mathcal H_t(X)\le\frac{A+mT}{S+T}. \tag{1}$$

For a signed measure of variation at most C, |A|<=CS and the omitted
numerator has absolute value at most CE. Monotonicity in E gives

$$\frac{A-CT}{S+T}\le\mathcal H_t(X)\le\frac{A+CT}{S+T}. \tag{2}$$

Suppose the supplied local histogram has weighted error eta. Its
unweighted error is also at most eta. Each diagonal return probability
lies in [0,1], so a computed numerator Ahat has |A-Ahat|<=S eta.
Intersect [Ahat-S eta,Ahat+S eta] with [0,mS] or [-CS,CS], respectively,
and insert its endpoints in (1) or (2). The returned interval radius is
at most

$$\eta+\frac{mT}{2(S+T)}\quad\hbox{or}\quad
\eta+\frac{CT}{S+T}.$$

`heat_return` budgets at most epsilon/2 to each contribution. All numbers
in this argument are rational when time is rational; no floating-point
exponential enters the certificate. Returned decimal displays are
approximations to the stored rational endpoints. The floating-point
eigensolver is solely an independent comparison.

## 5. Continuity with bounds, and obstruction on the full completion

For fixed D,C and t, the heat functional is uniformly continuous on the
subset with degree bound D and total variation at most C. Indeed, let
eta_M be the omitted Poisson probability. The retained local function is
bounded by one, and each discarded part has absolute value at most
C eta_M. Thus, for X,Y in this subset and R=ceil(M/2),

$$|\mathcal H_t(X)-\mathcal H_t(Y)|
\le p_{R,1}(X-Y)+2C\eta_M. \tag{3}$$

First choose M to control the tail, then the local seminorm to control
the first term. The degree and variation bounds make the rate effective.

**Proposition.** For every t>0, the heat functional on finite graph
combinations has no continuous linear extension to all of A. The
obstruction already occurs among degree-two finite combinations.

**Proof.** A continuous linear functional is bounded by a constant times
some p_(R,k), so it vanishes on ker(T_R). Choose n>=max(3,2R+2). Then

$$X_n=U(C_n)-U(C_{2n})\in\ker(T_R).$$

The double covering C_(2n) -> C_n injects rooted closed walks upstairs
into those downstairs, by unique lifting, and the injection misses the
two once-around walks of length n. Hence their rooted adjacency moments
satisfy m_j(C_n)>=m_j(C_(2n)) for every j, with strict inequality at n.
Since both graphs have degree two,

$$\mathcal H_t(X_n)
=e^{-2t}\sum_{j\ge0}\frac{t^j}{j!}
   [m_j(C_n)-m_j(C_{2n})]>0.$$

This contradicts vanishing on ker(T_R). Equivalently,
X_n/H_t(X_n) tends to zero in every local seminorm as n grows, while
its heat value stays one. Its coefficient mass is unbounded. QED.

At t=0 the functional is the continuous vertex character. For t>0,
the preceding obstruction explains why a uniform degree bound alone
does not ensure continuity on the whole signed vector space. Uniform
control of total variation is substantive, especially in view of the
previously constructed elements beyond the finite-measure model.

The heat functional is still multiplicative on its bounded-degree,
finite-variation domain. The Cartesian Laplacian is
Delta_G tensor I + I tensor Delta_H, so its diagonal heat kernel is the
product of those of the factors. Integrating against the product signed
measure proves H_t(XY)=H_t(X)H_t(Y). This is a character on that algebraic
domain, with the continuity qualifications just stated.

## 6. Validation and practical outcome

The [test suite](../../python/tests/test_graphlocal.py) checks exact
isomorphism against independent permutation enumeration, all 300 local
Cartesian products at radii zero through two for connected graphs of
at most four vertices, weighted cancellation, approximation propagation,
and finite reconstruction duals. Heat moments are checked against dense
rational full-matrix powers. Independent 70-digit complete-graph and
infinite-line formulas test the final enclosures, including signed and
deliberately perturbed local inputs.

The [benchmark](../../python/examples/heat_benchmark.py) compares six
synthetic finite graphs with dense symmetric eigensolves, adds the
specialized cycle-spectrum formula, and checks factorized and limiting
inputs where available. All resulting numerical comparisons fall in
the certified intervals. See the [recorded results](../../python/results/heat_benchmark.json)
and [interpretation](../../python/README.md#measured-first-application).

Preserving a Cartesian factorization improves this implementation's
explicit-product path. In the recorded run the factorized prism takes
6.82 ms versus 2.32 ms for dense eigensolving; the factorized torus takes
6.93 ms versus 14.29 ms. No conventional product-aware baseline was timed.
Explicit neighborhood extraction is expensive: its largest slowdown
against the dense baseline is the torus (about 135 times), followed by
the irregular example (about 87 times). These measurements establish
no general advantage over specialized graph or matrix algorithms.
