# graphnumbers-local 0.2.0

A research library for exact graph arithmetic and certified local
approximations, imported as `graphlocal`. Python 3.10 or later; the core
has no third-party runtime dependencies. The mathematical object is the
[local completion](../research/local-completion/README.md).

This package is independent of the historical `reboot/` software.

## Run or install

From this directory:

```sh
PYTHONPATH=src python3 examples/quickstart.py
PYTHONPATH=src python3 tests/run_tests.py --output results/verification.json
python3 -m pip install .
```

The benchmark additionally needs NumPy and SciPy:

```sh
python3 -m pip install '.[benchmark]'
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 examples/heat_benchmark.py --output results/heat_benchmark.json
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 examples/defect_benchmark.py --output results/defect_benchmark.json
```

## Exact arithmetic and local limits

```python
from fractions import Fraction
from graphlocal import Finite, Line, cycle, exp, heat_return, path, reconstruct

H = Finite.from_graph(path(2), normalize=True)
assert (H * H).finite() == Finite.from_graph(cycle(4), normalize=True)

L = Line()                         # the normalized-cycle limit
grid = L * L                      # infinite square-lattice element
local = grid.approximate(radius=2, k=1, epsilon="1e-8")
assert local.histogram.norm(1) == 13
assert local.error == 0

series = exp(H / 10).approximate(radius=1, k=1, epsilon="1e-8")
assert series.error <= Fraction("1e-8")

result = reconstruct(L.local(2), [path(4), path(5)])
assert result.coefficients == (Fraction(-1), Fraction(1))
assert result.cost == 9

heat = heat_return(grid, time="1/2", epsilon="1e-8")
print(float(heat.interval.midpoint))  # approximately 0.2169320140
print(heat.interval.lower, heat.interval.upper)  # exact rational endpoints
```

`Finite` combines isomorphic connected components with rational
coefficients. Empty graphs represent zero; isolated vertices represent
integers. Accepted coefficients are `int`, `Fraction`, or exact
rational/decimal strings. Use `"0.1"` or `Fraction(1, 10)` for one tenth;
floating-point inputs are rejected. Arbitrary real coefficients and
arbitrary completed elements require additional effective representations.

Operators build lazy algebra expressions. `.finite(max_vertices=...)`
materializes finite expressions as exact combinations and can decide
their equality up to graph isomorphism. `.local(radius)` computes exact
local data when available, without materializing a full product.
`exp(...)` instead supplies certified approximations. Equality of a
single local histogram does not imply equality of completed elements.

`from_networkx(G)` imports a simple undirected NetworkX graph if that
optional package is already present. Labels and attributes are discarded;
loops, directed graphs and multigraphs are rejected.

## Accuracy contract

`X.approximate(r, k, epsilon)` returns a `LocalApproximation` with a finite
rational `histogram` and a nonnegative rational `error` satisfying

$$\|T_rX-\text{histogram}\|_{1,k}\le\text{error}\le\varepsilon.$$

The radius is explicit; unbounded degrees can require infinitely many
types even at a fixed radius. Supported constructors provide effective
finite descriptions or controlled series tails. The method does not
claim to evaluate an arbitrary infinite array from finitely many inputs.

| Operation | Certificate or behavior |
| --- | --- |
| `a.add(b)`, `a.scale(c)` | Sum or scale the error bounds |
| `a.multiply(b)` | Bound both input errors and their cross term |
| `a.truncate(r, k)` | Preserve the bound at weaker radius/weight |
| `VERTICES`, `EDGES`, `ISOLATED`, `walk_observable(n)` | Evaluate to rational intervals |
| `polynomial`, `polynomial_derivative`, `exp`, `exp_derivative` | Algebraic calculus with local error control |
| `reconstruct(histogram, catalog)` | Exact representative and catalog-optimal coefficient-mass certificate |
| `catalog(max_vertices, max_degree)` | Complete small graph catalog, subject to a work limit |

For a `LocalObservable(r, k, C, function)`, its caller certifies the
global growth bound `abs(function(B)) <= C * |B|**k`. Custom `Element`
implementations likewise supply mathematical approximation and norm
bounds. Arbitrary input histograms are not automatically validated as
balanced coherent elements of the full completion.

The implementation uses exact isomorphism, not hash equality as a
substitute. Backtracking and graph growth limit practical sizes.
Materialized and local products default to a 10,000-vertex limit per
constructed graph; the isomorphism search has a 100,000-node work budget.
Exponential and reconstruction routines have explicit term/search budgets.
`BudgetExceeded` means the implementation stopped without a result.
`OutOfSpan` supplies a dual witness against the specified catalog only.

No general inverse operation or test of equality of arbitrary limits is
provided. Scalar division is exact rational division.

## Heat return and its domain

`heat_return(X, time, epsilon)` returns a rational interval of radius at
most epsilon. For `X = Finite.from_graph(G, normalize=True)` it encloses

$$\frac{\operatorname{tr}(e^{-t\Delta_G})}{|V(G)|},\qquad
\Delta_G=\operatorname{diag}(\deg)-A_G.$$

Each edge has jump rate one. An unnormalized finite input gives the
unnormalized trace. For the line and its products the same call evaluates
the corresponding infinite-graph return probability. Signed combinations
are supported when a finite global total-variation bound is supplied.

The input must supply a maximum-degree bound and a global variation bound.
Built-in finite elements, `Line`, and their sums/products do so. General
algebra exponentials do not supply those bounds and are rejected by this
application. `exp(X)` is the exponential in graph arithmetic; the heat
operator here acts on vertices of a graph.

Uniformization gives a Poisson series of lazy-walk return probabilities.
Terms through order M depend on radius `ceil(M/2)`, including the precise
boundary-degree issue. A rational geometric bound encloses the remaining
Poisson tail, and the interval includes any input approximation error.
`certificate.to_data(include_histogram=True)` exports the rational
parameters, retained moments and local data for inspection.

The [analysis note](../research/local-completion/EFFECTIVE_ANALYSIS_AND_HEAT.md)
proves these guarantees and a limitation: for positive time, heat return
has no continuous linear extension to the full signed completion.
It is uniformly continuous on classes with fixed degree and total-variation
bounds. The global bound is therefore part of the mathematical contract.

## Sparse defects and relative heat

`relative_heat(X, time, epsilon)` uses a uniform edit bound in place of
global total variation. It computes the **unnormalized** heat-trace
correction by default, preserving the signal of a fixed defect as the
graph grows. Its output is a rational interval of radius at most epsilon.

```python
from graphlocal import CutLineDefect, Line, SparseEdgeDifference, cycle, relative_heat

cut_cycle = SparseEdgeDifference(cycle(128), [(0, 127, -1)])
finite_response = relative_heat(cut_cycle, "1/2")

E = CutLineDefect()                # limit of P_n - C_n, without normalization
assert E.local(5).norm(0) == 20
response = relative_heat(E, "1/2")
print(float(response.interval.midpoint))   # approximately 0.4323323584

surface = E * Line()               # planar cut, per transverse volume
surface_response = relative_heat(surface, "1/2")
```

`SparseEdgeDifference` takes distinct `(u, v, sign)` edge edits, with
sign +1 for insertion and -1 for deletion. Its local calculation visits
only roots near edited endpoints, then combines rooted types exactly.
`normalize=True` divides both the difference and its edit budget by the
common vertex count. Construction still validates the explicit graph;
the locality statement concerns the subsequent correction calculation.

The cut-line element E satisfies `||T_r E||_1 = 4r`, so it has no finite
global signed-measure representation. A degree bound of two and a
uniform one-edge edit budget nevertheless give

$$\mathcal H_t(E)=\frac{1-e^{-4t}}2,\qquad
\mathcal H_t(E L^d)=\frac{1-e^{-4t}}2\,h_{\rm line}(t)^d.$$

For a degree bound D and weighted edit budget q, retaining moments
through M leaves absolute error at most
`2*q*t*Pr[Poisson(t*D) >= M]`. This bound is independent of graph volume.
The implementation also encloses the exponential normalization using
rational arithmetic and accounts for input approximation error.

Sums and scalar multiples propagate edit bounds. Multiplication by a
bounded-degree element of finite variation C propagates q to q*C.
Arbitrary products of defects need not have this certificate: E*E is
a valid algebra element but is currently rejected by `relative_heat`.
Generic `Finite` expressions do not automatically infer an edit pairing;
use `SparseEdgeDifference` when that structure is known.

Custom elements supplying `edit_bound` must certify both the lazy-return
moment bound `abs(d_j) <= 2*q*j/D` and the whole-response bound
`abs(H_t) <= q*min(1, 2*t)`, as well as degree and zero vertex mass.
These follow for the implemented constructors from finite edits and
the proved product extension. The metadata is additional analytic
information, not a property guaranteed for every element of the completion.

The [proof](../research/local-completion/SPARSE_DEFECTS_AND_RELATIVE_HEAT.md)
establishes the controlled domain, exact cut formula, and Cartesian
propagation. It is compatible with the earlier obstruction: heat still
has no continuous linear extension to the unrestricted signed completion.
The [literature comparison](../research/local-completion/SPARSE_DEFECT_COMPARISON.md)
credits relative heat, uniformization, and low-rank matrix updates.

The [defect experiment](examples/defect_benchmark.py) compares rational
certificates to shared-subspace block Krylov updates, sparse polynomial
traces, and dense eigensolves. Numerical baselines do not certify floating
point roundoff; their values are checked against our exact intervals.
Every comparison uses unnormalized corrections and fixed absolute accuracy.
See the [recorded results](results/defect_benchmark.json).

## Measured first application

[Recorded results](results/heat_benchmark.json), with t=1/2 and requested
absolute error 1e-8. Times below are milliseconds from one run with one
BLAS thread, rounded for readability. Graph construction is excluded;
local caches start cold. The dense baseline includes matrix construction
and `scipy.linalg.eigh(..., eigvals_only=True)`.

| Input | Explicit graph local method | Dense eigensolve pipeline | Factorized input | Limiting element |
| --- | ---: | ---: | ---: | ---: |
| C64 | 2.46 | 0.31 | — | 0.33 |
| C256 | 9.12 | 2.45 | — | 0.36 |
| C768 | 35.40 | 32.63 | — | 0.42 |
| C128 Cartesian K2 | 25.85 | 2.32 | 6.82 | 0.77 |
| C24 Cartesian C24 | 1923.78 | 14.29 | 6.93 | 2.11 |
| Irregular degree-at-most-three graph, 128 vertices | 77.72 | 0.89 | — | — |

All six numerical baseline values lie inside the computed rational
enclosures. The square-lattice result is approximately 0.2169320140,
with interval radius below 1.94e-9, using one radius-seven ball of 113
vertices. The factorized torus produces exactly the same certificate
as its materialized graph. The limiting lattice has the same retained
local data; its exact heat value need not equal that of the finite torus.

Preserving the torus factorization reduces this implementation's local
calculation by about 278 times in the recorded run. Direct extraction
from an explicit graph is generally slower than the dense baseline at
these sizes. The specialized cycle-spectrum formula is faster still
(recorded separately). These are six synthetic examples and single-run
timings, with different numerical guarantees; they establish no general
performance advantage or optimality against sparse solvers.

The first demonstrated use is retaining algebraic structure and known
limits through a certified computation. Rooted isomorphism and repeated
neighborhood extraction are the clearest performance targets.

## Measured defect application

The [defect results](results/defect_benchmark.json) contain 14 finite cases
and four infinite cases at t=1/2 and t=2, requesting absolute error 1e-8.
Each reported runtime is the median of three samples with one BLAS thread.
Local caches start cold. Shared explicit graph construction is recorded
separately; numerical method times include matrix construction.

| Unnormalized correction | Time t | Exact local certificate, ms | Block Krylov, ms | Dense reference, ms |
| --- | ---: | ---: | ---: | ---: |
| One cut in C512 | 0.5 | 2.65 | 1.08 | 26.96 |
| One cut in C4096 | 0.5 | 12.54 | 7.22 | — |
| One cut in C4096 | 2 | 27.43 | 8.16 | — |
| Two mixed edits, irregular graph on 128 vertices | 2 | 91.33 | 1.12 | 1.87 |

Block Krylov is faster in all 14 finite cases; the median local/Krylov
runtime ratio is about 12.1. All computed Krylov, dense, and sparse
references lie inside the rational intervals, without an added containment
allowance. The largest observed Krylov/dense discrepancy is 1.98e-13.
This is observed agreement, not a proof of the floating methods' accuracy.
The Krylov comparison also receives a dimension-independent analytic
truncation bound; roundoff and numerical rank deflation remain uncertified.

For the infinite product E*L, direct local certificates take 17.10 ms
at t=1/2 and 243.92 ms at t=2. Applying the proved heat factorization to
separately certified E and L intervals takes 1.24 ms and 4.86 ms,
respectively. Both routes use no finite volume approximation and satisfy
the requested tolerance. Conventional methods also have access to this
Cartesian factorization; these internal speedups establish no exclusive
algorithmic advantage.

For one cut the affected-root count is 14 at t=1/2 and 24 at t=2 for
every tested cycle size. Actual finite-input runtime still grows with n;
the current representation and whole-graph cache keys introduce overhead.
The mathematical certificate has volume-independent degree and radius
requirements, but the current implementation is not constant-time in n.

The result is a useful extension of what this algebra can represent and
evaluate: full signed local defect geometry, beyond finite measures, can
be combined with background factors and passed to a controlled spectral
observable. A speed advantage over established low-rank methods has not
been demonstrated. Reuse across many observables and interacting defects
is a more discriminating next experiment than another single heat trace.

## Verification and provenance

[Tests](tests/test_graphlocal.py) use standard-library `unittest`, exact
rational full-matrix comparisons, independent permutation enumeration,
and 70-digit closed forms. They include deliberately inaccurate input
oracles with valid bounds, so error propagation is exercised. The
[result record](results/verification.json) gives the current test count.
These tests accompany the mathematical proofs and are not formal proof
verification.

The [defect tests](tests/test_defects.py) check exact sparse cancellation,
growing cut-line variation, closed-form moments and relative heat,
normalization, signed products, and separated versus interacting cuts.
The optional [baseline tests](tests/test_defect_baselines.py) check Krylov
polynomial trace exactness and compare numerical methods independently.
All 39 tests passed with NumPy/SciPy; running with site packages disabled
passes the 33 core tests and skips the six optional baseline tests.

Graph isomorphism and catalog reconstruction are adapted from the existing
research verifiers, which remain unchanged. The runtime package imports
none of those scripts. Uniformization is established Markov-chain
methodology; the analysis note gives a primary reference. The project
makes no originality claim for that method or the full completion.
