# Local geometry retained beyond scalar spectral data

30 September 2026. The rook–Shrikhande pair gives an exact example of the
information retained by the local graph-number representation and lost by
scalar spectral compression. The pair is classical: Shrikhande's original
work is [1], and the primary research paper [2, discussion after Corollary
7.4] explicitly compares the two graphs' common spectrum and different
clique numbers. The calculations below specialize that known example to
the present algebra and give an executable verification.

The companion [interaction note](DEFECT_INTERACTIONS.md) derives exact
line-cut identities before selecting a scalar observable. Reuse timings
are recorded by the [benchmark](../../python/examples/reuse_benchmark.py).

## 1. Two explicit graphs

Let \(R=K_4\square K_4\), the rook graph on
\(\mathbb Z_4\times\mathbb Z_4\): vertices are adjacent when exactly one
coordinate changes. Let \(S\) be the Cayley graph of the same group with
connection set

\[
\mathcal D=\{\pm(1,0),\pm(0,1),\pm(1,1)\}.
\]

Both graphs have 16 vertices, 48 edges and degree 6. Any two distinct
vertices have exactly two common neighbors, giving the exact matrix identity

\[
A_R^2=4I+2J,\qquad A_S^2=4I+2J.
\tag{1}
\]

For \(R\), this follows directly from rows and columns. For \(S\),
\(|\mathcal D\cap(d+\mathcal D)|=2\) for every nonzero
\(d\in\mathbb Z_4^2\); the accompanying program checks all matrix entries
using integer common-neighbor counts.

On constant vectors the adjacency matrix acts by 6. On the orthogonal
complement, (1) gives eigenvalues \(2\) or \(-2\). Their multiplicities
\(a,b\) satisfy \(a+b=15\) and \(6+2a-2b=0\). Thus both graphs have

\[
\operatorname{spec}(A)=\{6^1,2^6,(-2)^9\},\qquad
\operatorname{spec}(\Delta)=\{0^1,4^6,8^9\}.
\tag{2}
\]

The identity proves the full spectra; verifying finitely many power moments
is an independent arithmetic check.

## 2. A nonzero graph number invisible to every scalar spectral trace

Write \(U(G)=G/|V(G)|\) and set \(X=U(R)-U(S)\).
For every function \(f\) defined on \(\{0,4,8\}\),

\[
\frac1{16}\operatorname{tr}f(\Delta_R)
=\frac1{16}\operatorname{tr}f(\Delta_S)
=\frac{f(0)+6f(4)+9f(8)}{16}.
\tag{3}
\]

In particular, all Laplacian moments agree, as do the heat traces for every
\(t\ge0\),

\[
\mathcal H_t(U(R))=\mathcal H_t(U(S))
=\frac{1+6e^{-4t}+9e^{-8t}}{16},
\]

and the shifted resolvent traces for every \(a>0\),

\[
\frac1{16}\operatorname{tr}(aI+\Delta)^{-1}
=\frac1{16}\left(\frac1a+\frac6{a+4}+\frac9{a+8}\right).
\]

Adjacency spectral traces likewise agree. Both graphs are vertex-transitive,
so even the diagonal spectral measure at any individual vertex is the same
within and between these graphs. This statement concerns scalar traces and
diagonal spectral measures; it does not identify their full matrix kernels.

Their rooted radius-one balls differ. Removing the root from the rook
ball leaves \(K_3\sqcup K_3\). Removing the root from the Shrikhande ball
leaves \(C_6\): its cyclic order at the origin is
\((1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1)\).
Denote the two rooted balls by \(B_R,B_S\). Each has seven vertices and

\[
T_1X=\delta_{B_R}-\delta_{B_S},\qquad
p_{1,k}(X)=2\,7^k>0.
\tag{4}
\]

Thus a nonzero element of the graph-number algebra is annihilated by every
scalar Laplacian spectral trace in (3). Evaluating more times or more
spectral functions cannot recover this particular lost distinction.

## 3. A geometric witness that survives Cartesian backgrounds

Let \(Q_4(G)\) count the unlabeled four-vertex cliques of \(G\). Every
four-clique containing a chosen root corresponds to a triangle among its
neighbors, so

\[
Q_4(G)=\sum_{v\in V(G)}q_4(B_1(G,v)),\qquad
q_4(B)=\tfrac14\#\{\text{triangles in }B[N(o)]\}.
\]

The bound \(0\le q_4(B)\le |B|^3/24\) makes \(Q_4\) a continuous
linear functional on the completion. The rook graph has its four rows and
four columns as its eight four-cliques. The Shrikhande graph has none,
since its neighbor graph \(C_6\) has no triangle. Consequently

\[
V(X)=0,\qquad Q_4(X)=\frac8{16}=\frac12.
\tag{5}
\]

Every clique of size at least three in \(G\square H\) lies in a single
coordinate fiber: relative to a fixed member of the clique, neighbors
changing different coordinates would fail to be adjacent to each other.
Counting the two kinds of fiber gives

\[
Q_4(G\square H)=|V(H)|Q_4(G)+|V(G)|Q_4(H).
\]

Extending by bilinearity and continuity yields the point-derivation identity

\[
Q_4(YZ)=Q_4(Y)V(Z)+V(Y)Q_4(Z),\qquad Y,Z\in A.
\tag{6}
\]

In particular,

\[
Q_4(XY)=\tfrac12 V(Y).
\tag{7}
\]

Every mass-one background therefore retains a radius-one geometric witness
with value \(1/2\). This includes normalized finite graphs and the line or
its Cartesian powers. For a background with zero vertex functional, (7)
does not decide nonvanishing; the previously proved integral-domain result
still gives \(XY\ne0\) whenever \(Y\ne0\).

For every finite background \(H\), the Cartesian Laplacian is the Kronecker
sum \(\Delta_G\otimes I+I\otimes\Delta_H\). The eigenvalue multisets for
\(R\square H\) and \(S\square H\) therefore coincide, preserving every
scalar spectral trace despite (7). For bounded-degree finite-variation
background elements, convolution of rooted spectral measures gives the
same conclusion for their spectral integrals. In particular the heat
functional satisfies \(\mathcal H_t(XY)=0\), and its Laplace transform
gives zero shifted-resolvent difference for \(a>0\). These assertions use
that controlled domain; they do not posit spectral observables on every
element of the completion.

## 4. Meaning for the software comparison

The example provides a representational reason to retain rooted geometry
when an application needs both spectral and motif observables. A spectral
summary cannot reconstruct the four-clique statistic even with perfect
arithmetic and every heat time available. A stored local histogram can
evaluate that statistic alongside any other observable covered by its radius
and error contract. Cartesian multiplication transports the distinction in
a quantitatively explicit way.

Ordinary adjacency lists and direct motif-counting algorithms also distinguish
the pair. This is consequently a test of information retained, not evidence
of a speed advantage over conventional graph algorithms or a new
cospectrality result. Multi-observable benchmarks should compare histogram
reuse with direct local counting and retain preprocessing costs. Full
Laplacian data also retain adjacency; the information loss occurs when they
are compressed to scalar spectral data.

## 5. Related precedent for interacting defects

The alternating combination of responses from several defect subsets is a
classical inclusion-exclusion construction. Schaden [3, equation (5)]
defines an irreducible multiobject spectral function by an alternating sum
of heat traces. Equations (14)–(16) show cancellation of Brownian-loop
contributions unless the loop encounters every object. The setting is a
bounded continuum domain with positive local potentials or compatible
local boundary conditions. Its sign and ultraviolet conclusions require
those hypotheses; they do not automatically apply to degree-adjusted
graph edge removals.

Shajesh and Schaden [4, equations (24)–(27)] similarly isolate the two-body
part of Green functions through multiple scattering. These papers provide
relevant primary precedent for subtracting individual defect responses to
isolate interaction. The graph-algebra investigation can retain the entire
signed local interaction as an element and test several observables on it.
Its concrete signs, locality thresholds and certified tails still require
their own discrete proofs.

## 6. Exact computational verification

Run from the repository root:

```sh
PYTHONPATH=python/src python python/examples/cospectral_geometry.py
```

The standard-library example constructs both graphs, checks every entry of
(1), verifies all rooted neighborhoods, counts four-cliques independently,
and checks integer adjacency and Laplacian trace moments through degree 12.
It also constructs both products with \(K_2\), obtaining four-clique counts
16 and 0 on 32 vertices and normalized difference \(1/2\). The normalized
radius-one histogram difference has variation 2 and weighted norms 14 and
98 at exponents one and two. The script reports 674 exact checks; its four
unit tests additionally check the moment routine on an irregular path and
validate clique counting and arguments. It uses no numerical eigensolver.

## References

[1] S. S. Shrikhande. *The Uniqueness of the L2 Association Scheme*.
Annals of Mathematical Statistics 30(3), 781–798 (1959).
https://doi.org/10.1214/aoms/1177706207 . Original bibliographic record;
the modern primary discussion in [2] supplies the accessible comparison.

[2] Laura Mančinska, Irene Pivotto, David E. Roberson, Gordon Royle.
*Cores of Cubelike Graphs*. European Journal of Combinatorics 87, 103092
(2020). https://doi.org/10.1016/j.ejc.2020.103092 ;
https://arxiv.org/pdf/1808.02051 . Discussion after Corollary 7.4.

[3] Martin Schaden. *Irreducible Many-Body Casimir Energies of Intersecting
Objects* (2011 version). https://arxiv.org/pdf/1011.2475 .
Equations (4)–(5), (14)–(16).

[4] K. V. Shajesh, Martin Schaden. *Many-Body Contributions to Green's
Functions and Casimir Energies*. Physical Review D 83, 125032 (2011).
https://doi.org/10.1103/PhysRevD.83.125032 ;
https://arxiv.org/pdf/1103.3048 . Equations (24)–(27).
