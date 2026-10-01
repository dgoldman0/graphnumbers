# Radius one: free link coordinates and exact finite realization

1 October 2026. First foundation checkpoint for the
[development plan](../../reviews/local-completion-2026-09-30/DEVELOPMENT_PATH.md).
The proofs below are self-contained apart from the established local product
and completion definitions. Finite verification accompanies them; independent
referee review of this new note is pending.

Write A for the complex local Cartesian graph completion, M1 for its
radius-one rooted-ball monoid, and E1 for the intersection of all weighted
l1 spaces on M1. Let C be the countable set of nonempty connected finite
simple graph types. The empty graph is allowed as a link.

## 1. The monoid is free

Every radius-one ball is a cone c(H): add a new root adjacent to every
vertex of the finite graph H. Its link is H, and these constructions are
inverse on isomorphism types. Neighbors of a product root come from one
coordinate at a time; two in different coordinates are never adjacent.
Therefore

\[
\operatorname{lk}(B\star_1D)=\operatorname{lk}(B)\sqcup\operatorname{lk}(D),
\qquad M_1\cong\mathbb N^{(\mathcal C)}.
\]

The last identification uses unique decomposition into connected components.
The one-vertex rooted ball corresponds to the empty link and to exponent zero.
If alpha records link-component multiplicities, its ball has size

\[
w(\alpha)=1+\sum_{J\in\mathcal C}|V(J)|\alpha_J.
\]

Consequently E1 is the convolution algebra of coefficient families a_alpha
with sum_alpha |a_alpha| w(alpha)^k finite for every integer k>=1.
This is a weighted absolutely convergent power-series algebra on disc
variables, not the entire-function algebra arising from cut defects.

## 2. Every finite radius-one array has an explicit realization

**Theorem.** Every finitely supported array whose root degrees are at most D
is T1(x) for a finite rational linear combination x of connected graphs of
at most D+1 vertices, if its coefficients are rational. Real or complex
coefficients give the corresponding linear combination over that field.

**Proof.** It suffices to realize a point mass delta_c(H). Put n=|V(H)|
and use the unrooted connected graph G=c(H). The new center has degree n.
The other degree-n vertices are precisely the universal vertices of H;
all universal vertices of G are interchangeable by automorphisms. Their
radius-one balls therefore all have type c(H). Every other vertex of G
has degree less than n. If u(H) counts universal vertices of H, then

\[
T_1(G)=(1+u(H))\delta_{c(H)}
       +\sum_{v:\deg_G(v)<n}\delta_{B_1(G,v)}.                 \tag{1}
\]

Begin with delta_K1=T1(K1). Induct on n: each summand on the right after
the first has a smaller link and has already been realized by a combination
of connected graphs of at most n vertices. Subtract those realizations
from G and divide by 1+u(H). This realizes delta_c(H), using graphs of
at most n+1 vertices. Linearity proves the claim. No positive realization
or positive coefficients are asserted. QED.

For example, the two-leaf star has histogram
delta_c(2K1)+2 delta_K2, while T1(K2)=2 delta_K2. Hence the finite signed
element P3-K2 realizes the pure two-leaf-star type at radius one.

The construction gives a triangular basis: cones of all link graphs on
at most D vertices have a square radius-one histogram matrix, ordered by
link size, with diagonal 1+u(H). Its determinant is the product of these
positive integers. It gives an explicit alternative to the general
catalog optimality search; it does not minimize coefficient mass.

In particular, with B1 defined as the closure of T1(A) inside E1,

\[
B_1=E_1.                                                     \tag{2}
\]

Indeed finite arrays are dense in every weighted l1 intersection (truncate
an enumeration; each summable weighted tail tends to zero), and every such
array lies in T1(A0). This argument establishes the closure in (2); it does
not claim that T1(A) itself equals E1 or that (1) supplies a continuous
section into the full all-radius completion.

This direct proof avoids assuming that an arbitrary radius-one array has
already been extended to an all-radius balanced family before applying
Step B of the representation theorem.

## 3. All continuous radius-one characters

**Theorem.** The continuous unital characters of E1, and the continuous
characters of A bounded by a radius-one seminorm, are exactly

\[
\chi_z(a)=\sum_\alpha a_\alpha z^\alpha,
\qquad z\in\overline{\mathbb D}^{\mathcal C}.                 \tag{3}
\]

**Proof.** A continuous linear functional on E1 is bounded by C times
one of its increasing defining norms. Let e_J be the point mass whose
link is J. Its nth power has ball size 1+n|V(J)|. Multiplicativity gives

\[
|\chi(e_J)|^n\le C(1+n|V(J)|)^k,
\]

so taking nth roots gives |chi(e_J)|<=1. Freeness determines its value
on every point mass; density determines (3) on E1. Conversely any disc
assignment defines (3) absolutely, with |chi_z(a)|<=sum |a_alpha|.
Absolute double summation proves multiplicativity.

If chi on A is bounded by C p_(1,k), it vanishes on ker T1 and defines
a bounded functional on T1(A). Equation (2) extends it uniquely to E1;
continuity of multiplication preserves its character property. The
converse follows by composition with T1. QED.

The topology of pointwise convergence on these characters is exactly
the product topology of the discs. Coordinate evaluation gives one
direction; uniform l1 tail control of each fixed a gives the other.
This classifies this part of the spectrum only. General character
completeness at radius at least two is still open.

## 4. Specializations and limits of the conclusion

| Existing character or map | Disc coordinates |
| --- | --- |
| Root degree s^deg | z_J=s^|V(J)|, for |s|<=1 |
| Isolated-vertex character | z_J=0 for every nonempty J; the constant monomial is one |
| Counts of selected connected link components | Choose their variables freely; set all other z_J=1 |
| Normalized edge H=K2/2 | Its link is K1, so its powers use the single coordinate z_K1 |

For the last row, the radius-one coefficient weights are (1+n)^k.
The full radius-r hypercube balls have polynomial size in n, so the
induced all-radius topology remains the rapidly decreasing coefficient
topology: the earlier smooth disc-algebra identification follows.
The geometric polydisc retracts still require finite graphs with the
specified connected link types at every vertex. Free abstract link
coordinates do not produce such finite graphs automatically.

## 5. Reproducible finite evidence

From the repository root, using only the Python standard library:

```sh
python -S research/local-completion/verify_foundation_checkpoint.py
```

[verify_foundation_checkpoint.py](verify_foundation_checkpoint.py) uses
its own adjacency sets, exhaustive permutation canonicalization and exact
rational arithmetic. It constructs the triangular realizations for every
link type through four vertices and recomputes their histograms directly.
It also builds the independent connected-host catalog through five vertices
and computes the exact rank of its radius-one histogram matrix. Product
links are extracted from materialized Cartesian graphs, rather than by
reusing the link-union rule under test.

The [result record](foundation_checkpoint_results.json) includes the actual
realization coefficients, rank and catalog scope. These finite instances
check the constructive mechanism; (1)–(3) are proved for arbitrary size
above. No radius-two factorization result follows from them.
