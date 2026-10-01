# Quantitative finite-graph approximation

30 September 2026. Item 5 in the research plan, independently of the
deferred two-generator and derivation questions.
The only structural input is the
[representation theorem](REPRESENTATION_THEOREM.md).

This note gives explicit size and coefficient bounds, an exact finite
reconstruction problem, and sharp examples of the cost of signed
cancellation. The bounds establish a finite procedure; they are far
too large to promise a practical algorithm for general input.

## 1. A rate from one additional moment bound

Let $a$ be a completed real element. Fix a maximum observation radius
$R\ge0$, a maximum weight exponent $K\ge1$, an integer $q\ge1$, and
a tolerance $\varepsilon>0$. Suppose a bound $P$ is known with

$$p_{R+1,K+q}(a)\le P.$$

Use the degree filter $Q_D$ from the representation theorem, retaining
all vertices and removing edges incident to original vertices of degree
above $D$. Its transformed coherent array is $a^{(D)}$.
For every $r\le R$ and $1\le k\le K$, the existing tail estimate gives

$$
\|a_r^{(D)}-a_r\|_{1,k}
\le 2\sum_{|B|>D}|B|^k|a_{r+1}(B)|
\le 2D^{-q}p_{r+1,k+q}(a)
\le 2D^{-q}P.
\tag{1}
$$

The last inequality follows from truncation: increasing radius and
weight exponent can only increase the seminorm bound. Consequently

$$D=\max\left(1,\left\lceil(2P/\varepsilon)^{1/q}\right\rceil\right)\tag{2}$$

suffices. If $P=0$, take $D=1$.

Put

$$b=\sum_{j=0}^RD^j,\qquad M=(D+1)b.$$

The finite realization lemma supplies an $x$ with
$T_R(x)=a_R^{(D)}$, using connected graphs of maximum degree at most $D$
and size at most $M$. By compatibility, $T_r(x)=a_r^{(D)}$ for $r\le R$.
Thus the same $x$ approximates every requested seminorm to error at most
$\varepsilon$. This is an explicit rate conditional on the supplied
higher-moment bound, rather than a uniform rate for all completed elements.

## 2. A finite bound on coefficient mass

Enumerate all connected graphs $G_1,\ldots,G_N$ with maximum degree
at most $D$ and at most $M$ vertices. Let $H$ be their integer radius-$R$
histogram matrix, with one row for each occurring rooted ball type.
Its $j$th column is $T_R(G_j)$ and its column sum is $|G_j|\le M$.
Let $d=\operatorname{rank}H$.

The target $u=a_R^{(D)}$ lies in the column space of $H$ by the finite
realization lemma. Choose $d$ independent columns and $d$ rows forming
an invertible integer matrix $A$. Solve $Ac=u_I$ and set all other
graph coefficients to zero. Independence implies $Hc=u$ on all rows.

Since $|\det A|\ge1$, the entries of $A^{-1}$ are bounded by the
absolute values of its cofactors. Every column of a cofactor submatrix
has Euclidean norm at most its column sum, hence at most $M$.
Hadamard's inequality yields

$$
\|c\|_1\le dM^{d-1}\|u\|_1.
\tag{3}
$$

Define the **coefficient mass** of a finite graph combination by

$$\mathcal C(x)=\sum_G|c_G|\,|G|,\qquad x=\sum_Gc_GG,$$

combining isomorphic connected components before computing it. Then

$$
\boxed{\mathcal C(x)\le dM^d\|u\|_1
\le dM^d\|a_{R+1}\|_1.}
\tag{4}
$$

The final inequality uses contraction under the degree-filter pushforward.
At every radius $s$ and exponent $k\ge1$, the constructed finite element
also obeys $p_{s,k}(x)\le M^k\mathcal C(x)$.

For a bound involving only $D,R$, one can replace $d$ in (4) by the
very loose rooted-type count

$$Q=\sum_{h=1}^{b}h\,2^{\binom h2}.$$

Indeed $d\le Q$ and $dM^d\le QM^Q$. This counts labeled rooted graphs
as an upper bound; neither connectedness nor the degree restriction is
needed to make that upper bound valid.

The construction uses at most $d$ connected graph types. It provides an
existence bound, without asserting that these coefficients minimize mass.

## 3. Minimizing signed cancellation on a finite catalog

For any specified finite graph catalog with histogram matrix $H$ and
sizes $n_j=|G_j|$, define

$$
\tau(u)=\min\left\{\sum_j n_j|c_j|:Hc=u\right\}.
\tag{5}
$$

When $u$ lies in the column span this is a finite linear program, with
dual

$$
\tau(u)=\max\{f\cdot u: |(H^Tf)_j|\le n_j\ \forall j\}.
\tag{6}
$$

A primal solution and a dual feasible vector having the same objective
value are an exact optimality certificate. The dual constraint says that
the absolute average of the local observable $f$ is at most one on each
catalog graph. It is a certificate for that catalog, not for all graphs
unless a separate universal inequality is proved.

For completeness, basis enumeration is sufficient in exact arithmetic.
Split each coefficient into positive and negative parts. The resulting
linear program has an optimum at a basic feasible point with at most
$d=\operatorname{rank}H$ independent nonzero columns. On the independent
row space, the dual feasible set is a bounded polytope; choose an optimal
vertex. Complementary slackness puts all nonzero primal columns among
its active constraints. Extend those columns to an independent active
basis. On each selected column the dual value is its signed size, with
the sign prescribed whenever the primal coefficient is nonzero.
Enumerating bases and the remaining signs therefore includes a matching
primal-dual certificate. An implementation budget may stop that search.

Write the normalized coefficients as $\alpha_j=n_jc_j$ and let
$m=\sum_Bu(B)$. Summing $Hc=u$ gives $\sum_j\alpha_j=m$. The positive
and negative coefficient masses therefore satisfy

$$\mathcal C_+-\mathcal C_-=m,\qquad
\mathcal C_-^{\min}=(\tau(u)-m)/2.\tag{7}$$

For mass-one targets, this directly measures the least necessary negative
mass. Positivity of the target histogram does not guarantee positive
coefficients for a restricted catalog.

The accompanying `reconstruct_local.py` enumerates a size/degree-bounded
catalog, solves rational histogram systems, and searches independent
column bases for exact primal-dual certificates. It uses rational
arithmetic throughout. It reports budget exhaustion separately from
infeasibility; an insufficient catalog gives no general impossibility
claim. Input targets are finite rational histograms. Obtaining a filtered
histogram from arbitrary infinite real data still requires supplied
values or a controlled approximation to them.

From this directory, the included radius-two example is reproducible with

```sh
python3 reconstruct_local.py --input reconstruction_example.json --output reconstruction_example_result.json
python3 verify_spectral_approximation.py --output spectral_approximation_results.json
```

The input's adjacency bit rows describe the rooted target ball with vertex
zero distinguished; coefficients are integers or rational strings.
The output records every catalog graph, its coefficient, and the dual
observable, so the certificate can be checked independently. The
[recorded example](reconstruction_example_result.json) has optimal cost
nine and negative mass four. The [verification results](spectral_approximation_results.json)
contain four complete catalog certificates alongside finite regression
fixtures. The basis-enumeration solver is practical only for small catalogs;
even the connected graphs through five vertices can exhaust its default
search budget. These finite checks accompany the proofs, rather than establishing the
universal statements by enumeration.

## 4. A sharp size-versus-cancellation example

Let $L$ be the infinite-line element and let $P_n$ be the path on $n$
vertices. For every $R\ge1$,

$$T_R(P_{2R+1}-P_{2R})=T_R(L).\tag{8}$$

Each path has the same two copies of every boundary-root ball, one at
each end. The longer path has one additional central root whose ball
is the length-$2R$ path rooted at its midpoint. All other terms cancel.

There are three exact regimes when maximum connected-component size is
the constraint:

| Maximum size | Exact radius-$R$ reconstruction of $L$ |
| --- | --- |
| At most $2R$ | Impossible: the target ball has $2R+1$ vertices. |
| At most $2R+1$ | Unique: $P_{2R+1}-P_{2R}$, with coefficient mass $4R+1$ and negative mass $2R$. |
| At least $2R+2$ | Minimum mass one and negative mass zero, attained by $C_{2R+2}/(2R+2)$. |

For uniqueness in the middle row, every connected graph on at most
$2R+1$ vertices has a vertex of eccentricity at most $R$: take a center
of a spanning tree, whose radius is at most $\lfloor n/2\rfloor$.
Choose one whole-graph rooted ball for each such graph. Against the
histograms of these graphs, those rows form a triangular matrix ordered
by vertex count, with positive diagonal and zero entries between
nonisomorphic graphs of equal size. The histogram columns are independent.
Equation (8) is consequently the unique representation in that entire
size-bounded catalog, even without a degree restriction.

For the last row, the normalized cycle has the required path ball at
every root. Every representation has mass at least its vertex character
$V(L)=1$, so its coefficient mass one is optimal.

This distinguishes two costs sharply: minimizing graph size can force
large signed cancellation, while one additional vertex eliminates it.

## 5. Diverging cost beyond the finite-measure model

For every finite graph combination $x$ and radius $r$,

$$\|T_rx\|_1\le\mathcal C(x).\tag{9}$$

Let $J(c)=\sum_jc_jz_j$ be the cycle-difference family in the representation
theorem, with $N_j=3^j$. At radius $N_m$,

$$\|T_{N_m}J(c)\|_1=2\sum_{j\le m}|c_j|.$$

Therefore any finite approximant satisfying
$\|T_{N_m}x-T_{N_m}J(c)\|_1\le\varepsilon$ must obey

$$\mathcal C(x)\ge2\sum_{j\le m}|c_j|-\varepsilon.\tag{10}$$

If $c\notin\ell^1$, every sequence of finite graph combinations converging
to $J(c)$ has coefficient mass tending to infinity. Its vertex character
tends to zero, so both its positive and negative coefficient masses tend
to infinity. This is an unavoidable cost, independent of the reconstruction
algorithm. More generally, any completed element with unbounded local
total variation forces diverging coefficient mass in every finite
approximating sequence.

The quantitative results use finite-dimensional linear algebra, moment
tail estimates, and the existing graph representation theorem. They do
not depend on character separation, the Wiener theorem, two-generator
identifications, or derivation theory. Optimal general bounds and an
efficient large-input implementation remain open.
