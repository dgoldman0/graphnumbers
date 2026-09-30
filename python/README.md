# graphnumbers-local 0.1.0

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

## Verification and provenance

[Tests](tests/test_graphlocal.py) use standard-library `unittest`, exact
rational full-matrix comparisons, independent permutation enumeration,
and 70-digit closed forms. They include deliberately inaccurate input
oracles with valid bounds, so error propagation is exercised. The
[result record](results/verification.json) gives the current test count.
These tests accompany the mathematical proofs and are not formal proof
verification.

Graph isomorphism and catalog reconstruction are adapted from the existing
research verifiers, which remain unchanged. The runtime package imports
none of those scripts. Uniformization is established Markov-chain
methodology; the analysis note gives a primary reference. The project
makes no originality claim for that method or the full completion.
