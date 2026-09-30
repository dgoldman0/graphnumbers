# Graph structure and ordinary real analysis

This supplement records the clarification following the v0.1 proof note on
30 September 2026. The notation and completion are those of
[the note](local_graph_completion.pdf).

The subsequent [literature review](LITERATURE_REVIEW.md) places this calculus
within established locally convex algebra theory. In particular, the
edge-count identity below is a point-derivation identity; the
[dual-number and degree-polynomial calculation](COMPARISON_LEMMAS.md#4-the-edge-observable-is-a-point-derivation)
explains this directly. These formulas do not constitute a novelty claim.

## Finite graphs and the scalar line

Every finite simple undirected graph enters as the sum of its connected
components. Distinct isomorphism classes remain distinct in the completion.
The empty graph is zero, K1 is the multiplicative identity, and n isolated
vertices represent the scalar n. A graph consisting of one edge is distinct
from the scalar 2.

The embedding of real numbers is a -> a K1. Every defining seminorm satisfies

$$p_{r,k}(aK_1)=|a|.$$

Consequently the scalar line is a closed copy of the usual real topological
field. Limits, series, differentiation and integration restricted to this line
agree with their ordinary real counterparts. The rational graph span is dense,
so the same completed algebra can be obtained by starting with rational graph
combinations. This assertion concerns rational coefficients; general graph
fractions have not been embedded.

## Ordinary differentiation in graph counting identities

Let V and E denote the continuous linear extensions of vertex and edge counts.
Cartesian graph arithmetic gives

$$V(XY)=V(X)V(Y),$$

$$E(XY)=E(X)V(Y)+V(X)E(Y).$$

Also V(K1)=1 and E(K1)=0. For any polynomial f with real coefficients,

$$V(f(X))=f(V(X)), \qquad E(f(X))=f'(V(X))E(X).$$

**Proof.** Induction on m gives V(X^m)=V(X)^m and, for m >= 1,
E(X^m)=m V(X)^(m-1) E(X). Apply linearity to the polynomial. The constant term
contributes to V and contributes zero to E.

These identities also hold when f(z)=sum_m a_m z^m has infinite radius of
convergence. For each defining seminorm p, submultiplicativity gives

$$\sum_{m\geq 0}p(a_mX^m)\leq\sum_{m\geq 0}|a_m|p(X)^m<\infty.$$

Completeness supplies f(X). Continuity of V and E permits their application
term by term, and the differentiated scalar power series also converges.
Thus the edge-count identity is forced by the chosen graph arithmetic and
ordinary power-series differentiation.

For example,

$$E(X^2)=2V(X)E(X), \qquad E(\exp X)=\exp(V(X))E(X).$$

For the normalized cycle limit L, V(L)=E(L)=1, so

$$V(\exp(tL))=\exp(t), \qquad E(\exp(tL))=t\exp(t).$$

These observables constrain the graph-valued function. Agreement of vertex and
edge counts alone does not establish equality of graph elements.

## Graph directions and paths

For F(X)=X^2, a direction H can be any element of the graph algebra. Expanding
with a real parameter t gives

$$\frac{(X+tH)^2-X^2}{t}=2XH+tH^2\longrightarrow 2XH.$$

Thus DF_X[H]=2XH. When X and H are finite graphs, the derivative represents
two copies of their Cartesian product. The proof note similarly establishes
D exp(X)[H]=exp(X)H, with a remainder estimate in every defining seminorm.
These variations are real linear variations of embedded graphs. Their relation
to combinatorial edge edits and to global graph geometry requires further study.

If Q is any continuous linear graph observable and X(t) is a differentiable
path, continuity and linearity imply

$$Q(X'(t))=\frac{d}{dt}Q(X(t)).$$

For a continuous path on a compact interval, the Riemann integral exists in
the completion and

$$Q\left(\int_a^b X(t)\,dt\right)=\int_a^b Q(X(t))\,dt.$$

Both identities follow by applying Q to the defining difference quotients or
Riemann sums and passing to the limit. They apply to vertices, edges, and the
linear counts of fixed connected motifs established in the note. Every graph
derivative or integral must therefore reproduce ordinary calculus on those
measurements.

The scalar line retains ordinary real analysis. The whole algebra has the
specific compatible calculus proved here and in the note. General inverse and
implicit function theorems, nonlinear evolution problems, and the geometric
interpretation of more general variations need additional hypotheses and work.

Background on calculus in locally convex spaces, including continuous linear
maps and differentiation, is available in Kriegl and Michor,
[*The Convenient Setting of Global Analysis*, section 1.3](https://www.mat.univie.ac.at/~kriegl/Skripten/apbook.pdf).
