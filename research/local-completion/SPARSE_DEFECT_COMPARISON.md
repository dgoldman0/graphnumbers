# Sparse defects and heat traces: focused literature comparison

30 September 2026. Primary-source comparison for the sparse-defect
application of `graphnumbers-local`. The accompanying
[proof note](SPARSE_DEFECTS_AND_RELATIVE_HEAT.md) establishes the local
defect elements and their relative heat functional. Relative heat traces,
uniformization, and low-rank matrix-function updates are established tools;
this comparison makes no novelty claim for those methods.

## 1. The relevant computational problem

Let two finite simple graphs on the same vertex set have combinatorial
Laplacians \(L\) and \(L'=L+E\). If they differ by \(q\) edge insertions or
deletions, then

\[
E=BJB^T=\sum_{e=\{u,v\}}\sigma_e
(e_u-e_v)(e_u-e_v)^T,\qquad \sigma_e\in\{-1,1\}.
\]

Thus \(\operatorname{rank}E\le q\) and its trace norm satisfies
\(\|E\|_{S_1}\le2q\). The quantity of interest is the *relative heat trace*

\[
\delta_t=\operatorname{tr}(e^{-tL'}-e^{-tL}),
\]

or its normalized version \(\delta_t/n\). These are different numerical
targets: a fixed absolute tolerance on a normalized trace eventually hides
every fixed finite defect as \(n\) grows. The main accuracy target should be
absolute error in the unnormalized defect response, with normalized values
also reported when useful.

Sparse perturbation methods already address this problem directly. A
comparison only with dense eigendecomposition would therefore substantially
understate conventional capabilities.

## 2. Strong conventional algorithms

### Low-rank matrix-function updates

Beckermann, Kressner and Schweitzer [1] approximate
\(f(A+D)-f(A)\) using Krylov spaces seeded by the low-rank factors of \(D\).
Their analysis covers polynomial exactness, convergence for the exponential,
and network applications. In the symmetric rank-one case, an orthonormal
basis \(U_m\) of \(\mathcal K_m(A,b)\) gives the compressed difference

\[
X_m(f)=f(U_m^T(A+bb^T)U_m)-f(U_m^TAU_m).
\]

One can compute \(\operatorname{tr}X_m(f)\) without forming the full matrix
update. The paper also treats removal and higher-rank updates. Its proposed
method already preserves cancellation before computing a large result.

Cortinovis, Kressner and Massei [2, Theorems 2–3] give the especially relevant
trace improvement. For symmetric \(A,J\), \(D=BJB^T\), and \(U_m\) spanning
\(\mathcal K_m(A,B)\), the compressed trace difference is exact for every
polynomial of degree at most \(2m\). For an interval \(I\) containing both
spectra,

\[
|\operatorname{tr}(f(A+D)-f(A))-\operatorname{tr}X_m(f)|
\le4n\inf_{p\in\Pi_{2m}}\|f-p\|_I.
\]

The same paper develops divide-and-conquer and local-submatrix algorithms
for banded and hierarchical matrices. Its symmetric trace theorem directly
justifies a shared block Krylov baseline for mixed insertions and deletions.
The degree-doubling result is specific to the symmetric polynomial trace
setting; it does not generally extend to rational Krylov updates or the
whole matrix update.

### Polynomial filters, stochastic traces and locality

Shuman, Vandergheynst, Kressner and Frossard [3] use shifted Chebyshev
recurrences to apply graph spectral multipliers through local communication.
This supplies a conventional sparse polynomial-filter baseline. A polynomial
of degree \(m\) has finite propagation through \(m\) graph steps, so local
evaluation of polynomial observables alone is not a distinctive consequence
of the graph-number completion.

Ubaru, Chen and Saad [4] combine stochastic trace estimation, Gaussian
quadrature and Lanczos. Their stated analysis considers analytic functions
on a spectral interval and symmetric positive-definite matrices. A singular
graph Laplacian can be shifted by \(\alpha I\), with
\(g(x)=e^{-t(x-\alpha)}\), to fit that assumption without changing the desired
matrix function. Tsitsulin and collaborators [5] apply stochastic Lanczos
quadrature specifically to graph spectral distances involving heat traces.
Their paper distinguishes normalized and combinatorial Laplacians; a
benchmark must retain the project's combinatorial Laplacian convention.

For defects, independent randomized estimates of the two large traces are
a weak baseline. The same probe vectors should be used for both matrices,
so cancellation occurs within each quadratic-form difference. Statistical
uncertainty must be reported separately from polynomial or Krylov truncation
error. A signed matrix-function difference need not be positive semidefinite,
so a relative-error theorem for positive traces cannot simply be reused.

Benzi and Razouk [6] prove decay bounds for functions of sparse matrices and
derive sparse approximations with linear complexity under suitable uniform
conditions. Frommer, Schimmel and Schweitzer [7] analyze graph-coloring
probing for sparse approximation and trace estimation of decaying matrix
functions. These works show why local neighborhoods and repeated sparse
matrix-vector operations are established alternatives to dense methods.

### Uniformization

For a common degree bound \(D\),
\(P=I-L/D\) and \(P'=I-L'/D\) are symmetric stochastic matrices. The identity

\[
e^{-tL}=e^{-tD}\sum_{j\ge0}\frac{(tD)^j}{j!}P^j
\]

is standard uniformization; Rao and Teh [8, Section 3.1] provide a primary
modern presentation. The package's exact rational Poisson bounds and local
histogram arithmetic are an implementation choice for certifying this
identity. Uniformization itself is established prior art.

## 3. Conventional perturbation bounds

The following calculations are direct consequences of the displayed
Laplacian update and standard finite-dimensional matrix identities. They
are included as benchmark constraints, without attributing their derivation
to this project as an original result.

Since both Laplacians are positive semidefinite, Duhamel's formula yields

\[
e^{-tL'}-e^{-tL}
=-\int_0^t e^{-(t-s)L'}E e^{-sL}\,ds,
\qquad |\delta_t|\le t\|E\|_{S_1}\le2qt.
\]

For edge removals alone the relative trace is nonnegative; additions give
the opposite sign. Rank-one eigenvalue interlacing bounds the change per
edge by one. Telescoping over valid intermediate graphs therefore gives
\(|\delta_t|\le q\) for mixed edits. Sobieczky [9, Theorem 1.8 and the
following discussion] applies interlacing directly to edge deletion in
regularized random walks and relates it to combinatorial Laplacians.

Uniformization also admits a tail certificate proportional to defect count.
Both \(P,P'\) are spectral-norm contractions, and telescoping powers gives

\[
|\operatorname{tr}((P')^j-P^j)|
\le\|(P')^j-P^j\|_{S_1}
\le j\|P'-P\|_{S_1}\le\frac{2qj}{D}.
\]

For \(N\sim\operatorname{Poisson}(tD)\), the omitted contribution after
orders \(0,\ldots,M\) is consequently at most

\[
\frac{2q}{D}\mathbb E[N\mathbf1_{N>M}]
=2qt\,\mathbb P(N\ge M).
\]

This bound is independent of graph size. After dividing by \(n\), it bounds
the normalized defect response. \(D=0\) is the trivial unchanged edgeless
case. Weighted edges require replacing \(q\) by the sum of absolute edge
weight changes and using a common weighted-degree bound.

The same tail estimate strengthens the analytic comparison for the
shared block Krylov method. Orthogonal compression preserves the spectral
contraction bounds and cannot increase the trace norm of the update.
The full and compressed relative trace moments agree through degree
\(2m\) by [2, Theorem 2]. Bounding both remaining tails gives

\[
|\delta_t-\operatorname{tr}X_m(e^{-t\cdot})|
\le 4qt\,\mathbb P(N\ge 2m).
\]

The [proof note](SPARSE_DEFECTS_AND_RELATIVE_HEAT.md#7-an-equally-volume-independent-krylov-bound)
derives this dimension-independent bound. It is an exact-arithmetic
estimate: floating point projection, numerical rank deflation, and
exponential evaluation need separate error control. The benchmark uses
this stronger bound when selecting a comparable moment budget.

## 4. Infinite defects and spectral-shift theory

Shirai [10] explicitly studies differences of heat kernels and Green
functions on countable bounded-degree graphs when Dirichlet conditions are
imposed on a finite set. Section 2 sums the diagonal heat-kernel differences
and controls distant contributions by jump counts and Poisson tails. Its
operator comes from a reversible transition probability plus bounded
potential. Dirichlet deletion differs from deleting an edge and adjusting
the combinatorial degrees, so its specific formulas should not be quoted
as the formula for our cut. It is nevertheless direct primary evidence
that finite-defect relative heat traces on infinite graphs are an established
subject.

Isozaki and Korotyaev [11, Section 6.2, equation (6.11)] study discrete
Schrödinger operators with trace-class potentials on the lattice and use
the spectral-shift trace formula

\[
\operatorname{tr}(f(H)-f(H_0))=\int\xi(\lambda)f'(\lambda)\,d\lambda.
\]

Their perturbations are potentials and their Laplacian normalization
differs from ours. The paper gives the relevant surrounding framework;
it does not identify the graph-number algebra or directly prove an
edge-removal result for it.

For bounded-degree infinite graphs on a common vertex set, finitely many
edge changes give bounded self-adjoint Laplacians with finite-rank
difference. Duhamel's formula directly makes their heat-semigroup difference
trace class. Thus a relative trace can exist even when either heat operator
has infinite trace. This operator argument requires neither a finite
probability measure on rooted graphs nor the completed graph algebra.

The [proof note](SPARSE_DEFECTS_AND_RELATIVE_HEAT.md) identifies a more
specific role for the algebra: \(P_n-C_n\) converges to an element recording
the *entire local geometry of a cut*, with local variation growing as
\(4r\). That element lies outside the finite signed-measure model. Its
limiting relative heat trace is computed from the same stabilized local
data. The geometric object and the scalar spectral observable are
different levels of representation.

The proved cut formula
\(\delta_t=(1-e^{-4t})/2\) has a spectral interpretation compatible with
\(\xi=-1/2\) on \((0,4)\) under the displayed trace-formula convention.
This is an inference from the project's formula, not a claim found in the
cited papers. There is also a useful order-of-limits issue: for fixed finite
\(n\), both graphs are connected and the heat-trace difference tends to
zero as \(t\to\infty\); taking \(n\to\infty\) first gives a subsequent
limit of \(1/2\).

## 5. Benchmark criteria and scope of the conclusions

A strong comparison evaluates the same defect response at the same times
and absolute tolerances using these methods:

1. Exact rational local moments with the defect-sensitive Poisson tail.
2. Shared-subspace block Krylov seeded by the edge incidence vectors, using
   [2] for its polynomial exactness and approximation interpretation.
3. Direct local polynomial recurrence without graph-isomorphism aggregation,
   to measure the cost or benefit of histogram reuse itself.
4. Small-matrix eigendecomposition as an independent numerical reference.

Large irregular-graph comparisons can also use paired stochastic
Chebyshev/Lanczos estimates and graph-coloring probing. Conventional
Cartesian factorization must be allowed wherever it is available: the
Kronecker-sum Laplacian identity already factors heat traces.

Relevant measurements include graph construction and neighborhood
extraction separately from query time, cold and warm caches separately,
maximum local size and degree, number of edited edges, and memory. For
parameter sweeps, all methods should reuse computed moments, projected
matrices, or background results. Growing-graph comparisons should provide
the same local-oracle or explicit-adjacency access: charging one method to
construct the whole graph while another receives a local oracle measures
the input model.

Exact rational enclosures and floating-point Krylov outputs provide different
guarantees. Reports should distinguish observed accuracy from a deterministic
arithmetic certificate and give the runtime for each method. A floating-point method can
in principle be supplemented with validated arithmetic, so certification
alone does not establish an intrinsic impossibility for competing methods.

The principal candidate benefit is a reusable *geometric*
representation: a defect can be added, multiplied by a background factor,
passed to several local observables, and taken to a limit while retaining
explicit approximation contracts. Sparse cancellation, matrix-function
locality, tensor factorization, and relative heat traces are established
tools. A substantive computational claim requires evidence that the common
representation saves repeated work across those operations. The cut element
already motivates the completion's beyond-measure part independently of
whether its heat evaluation wins a speed comparison.

## Recorded comparison

The [reproducible benchmark](../../python/examples/defect_benchmark.py)
and [results](../../python/results/defect_benchmark.json) include 14 finite
cases, with 12 dense/sparse numerical references, and four infinite cases.
The shared block Krylov implementation is faster than the exact local
histogram route in all finite cases. All numerical references fall inside
the rational intervals; the largest Krylov/dense discrepancy is 1.98e-13.
Degree-matched Krylov outputs receive the dimension-independent analytic
bound above, with floating errors explicitly excluded. No speed advantage
for graph-number heat evaluation has been established by this experiment.

The cut-line element and its product with the line are computed directly
with certificates. Factorized scalar evaluation is substantially faster
than constructing the product histogram, but this uses the standard
Cartesian heat identity. The demonstrated role of the completed algebra
is retaining the full local defect geometry and a reusable approximation
contract. Whether that representation pays off across many observables
or interacting defects remains a separate application question.

## References inspected

[1] Bernhard Beckermann, Daniel Kressner, Marcel Schweitzer. *Low-rank updates
of matrix functions* (2017 preprint). https://arxiv.org/abs/1707.03045 ;
full text https://arxiv.org/pdf/1707.03045 . Sections 2–3 and 6.

[2] Alice Cortinovis, Daniel Kressner, Stefano Massei. *Divide and conquer
methods for functions of matrices with banded or hierarchical low-rank
structure* (2021 preprint). https://arxiv.org/abs/2107.04337 ;
full text https://arxiv.org/pdf/2107.04337 . Theorems 2–3, Sections 3 and 5.

[3] David I. Shuman, Pierre Vandergheynst, Daniel Kressner, Pascal Frossard.
*Distributed Signal Processing via Chebyshev Polynomial Approximation*.
https://arxiv.org/abs/1111.5239 ; full text https://arxiv.org/pdf/1111.5239 .

[4] Shashanka Ubaru, Jie Chen, Yousef Saad. *Fast Estimation of tr(f(A)) via
Stochastic Lanczos Quadrature*. SIAM Journal on Matrix Analysis and
Applications 38(4), 1075–1099 (2017).
https://doi.org/10.1137/16M1104974 .
https://www.cs.cornell.edu/courses/cs6241/2020sp/readings/Ubaru-2017-fast.pdf .

[5] Anton Tsitsulin and collaborators. *Just SLaQ When You Approximate:
Accurate Spectral Distances for Web-Scale Graphs* (2020).
https://arxiv.org/abs/2003.01282 ; https://arxiv.org/pdf/2003.01282 .

[6] Michele Benzi, Nader Razouk. *Decay bounds and O(n) algorithms for
approximating functions of sparse matrices*. Electronic Transactions on
Numerical Analysis 28, 16–39 (2007–2008).
https://etna.ricam.oeaw.ac.at/volumes/2001-2010/vol28/abstract.php?pages=16-39 .

[7] Andreas Frommer, Claudia Schimmel, Marcel Schweitzer. *Analysis of probing
techniques for sparse approximation and trace estimation of decaying matrix
functions* (2020 preprint). https://arxiv.org/abs/2009.01589 ;
https://arxiv.org/pdf/2009.01589 .

[8] Vinayak Rao, Yee Whye Teh. *Fast MCMC Sampling for Markov Jump Processes
and Extensions*. JMLR 14 (2013), Section 3.1.
https://jmlr.org/papers/volume14/rao13a/rao13a.pdf .

[9] Florian Sobieczky. *An interlacing technique for spectra of random walks
and its application to finite percolation clusters* (version 4, 2008).
https://arxiv.org/abs/math/0504518 ; https://arxiv.org/pdf/math/0504518 .
Theorem 1.8 and the following discussion of combinatorial Laplacians.

[10] Tomoyuki Shirai. *A Trace Formula for Discrete Schrödinger Operators*.
Publications of the Research Institute for Mathematical Sciences 34(1),
27–41 (1998). https://doi.org/10.2977/PRIMS/1195144826 ;
https://ems.press/content/serial-article-files/40676 . Section 2.

[11] Hiroshi Isozaki, Evgeny Korotyaev. *Inverse problems, trace formulae for
discrete Schrödinger operators* (2011). https://arxiv.org/abs/1103.2193 ;
https://arxiv.org/pdf/1103.2193 . Section 6.2, equation (6.11).
