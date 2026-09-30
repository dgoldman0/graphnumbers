# Higher defect interactions: cyclic couplings and exact tree geometry

30 September 2026. All edge rates are one, Laplacians are combinatorial,
and graph traces are unnormalized. This note extends the pair calculation
in [PLANAR_DEFECTS.md](PLANAR_DEFECTS.md) and the exact bridge reduction in
[BRANCHING_DEFECTS.md](BRANCHING_DEFECTS.md). The matrix identities are
classical rank-one expansion and determinant identities. The purpose is
to identify precisely how the graph completion's retained defect geometry
controls its heat interactions; no priority claim is made.

## 1. Exact cyclic expansion for any number of defects

Let G be a finite simple graph, L its Laplacian, and e_1,...,e_k distinct
selected edges, with oriented incidence columns b_i and B_i=b_i b_i^T.
For k>=1 define

\[
 I_n=\sum_{S\subseteq[k]}(-1)^{k-|S|}
       \operatorname{Tr}\left(L-\sum_{i\in S}B_i\right)^n,
 \qquad H_F(t)=\sum_{n\ge0}\frac{(-t)^n}{n!}I_n.
\]

In particular I_0=0. Introduce the cross moments

\[
 c_{ij}(a)=b_i^T L^a b_j,\qquad a\ge0.
\]

**Cyclic expansion theorem.** For n>=1,

\[
 I_n=\sum_{m=k}^n(-1)^m\frac nm
  \sum_{\substack{i_1,\ldots,i_m\in[k]\\
                    \{i_1,\ldots,i_m\}=[k]}}
  \sum_{\substack{a_1+\cdots+a_m=n-m\\a_j\ge0}}
        \prod_{j=1}^m c_{i_j i_{j+1}}(a_j),
 \qquad i_{m+1}=i_1.                                      \tag{1}
\]

Repeated defect labels must be included. The condition is full support
of the label word, rather than each label occurring exactly once.

To prove (1), expand every noncommuting power into words in L and the
B_i. Inclusion-exclusion removes every word omitting at least one
selected B_i. A surviving word with m defect letters has sign (-1)^m.
The trace of the cyclic word

\[
 B_{i_1}L^{a_1}B_{i_2}L^{a_2}\cdots B_{i_m}L^{a_m}
\]

is the product in (1). To count correctly, mark one of its m defect
positions in each linear length-n word. Rotation moves that marked
position to the beginning. Summing over the n possible linear origins
and then dividing by the m markings gives n/m. This double-counting
argument remains valid for periodic cyclic words.

For mixed insertions and deletions, write the perturbations as
sigma_i B_i with sigma_i in {+1,-1}. Replace (-1)^m in (1) by
product_j sigma_(i_j). Weighted perturbations work in the same way.
Every expression is invariant under reversing any incidence orientation:
each b_i appears an even total number of times in each cyclic product.

An equivalent generating function exposes why only k-dimensional
transfer data are needed. Let U have columns b_i, set
C(z)=U^T(I-zL)^(-1)U, and X=diag(x_1,...,x_k). The determinant lemma gives

\[
 \det\left(I-z\left(L-\sum_i x_iB_i\right)\right)
       =\det(I-zL)\det(I+zX C(z)).                           \tag{2}
\]

Expanding the logarithm of the second determinant and applying Boolean
inclusion-exclusion to x selects all monomials whose support is [k].
Together with log det(I-zA)=-sum_(n>=1) z^n Tr(A^n)/n, this proves (1)
again. Retaining only squarefree monomials would lose repeated edits
that contribute to the actual interaction.

For bounded-degree infinite graphs and finitely many selected edges,
all polynomial differences above are finite rank and the same formulas
hold for their finite traces. The cross moments stabilize on finite
exhaustions. Relative heat is trace class by Duhamel, and its Taylor
series converges in trace norm on bounded t-intervals. For example,
expanding a polynomial difference against L bounds its trace norm by
a constant times n times a fixed operator-norm bound to power n-1;
this is sufficient for convergence after division by n!.

## 2. The first possible order and its signed coefficient

For i!=j let

\[
 s_{ij}=\min\{a\ge0:c_{ij}(a)\ne0\},\quad
 w_{ij}=1+s_{ij},\quad \gamma_{ij}=c_{ij}(s_{ij}),             \tag{3}
\]

with w_ij=infinity if all cross moments vanish. A cyclic covering word
is a closed word visiting every defect label. Its weight is the sum of
the w_ij of its successive transitions. Set w_ii=1 and gamma_ii=2,
though a minimum covering word for k>=2 never needs a self-transition.
Let nu be the minimum weight among covering words.

**First-order test.** Every I_n with n<nu vanishes. At n=nu,

\[
 I_\nu=\nu
  \sum_{\substack{(i_1,\ldots,i_m)\text{ covers }[k]\\
                   \sum_j w_{i_j i_{j+1}}=\nu}}
       \frac{(-1)^m}{m}\prod_j\gamma_{i_j i_{j+1}}.          \tag{4}
\]

The sum includes all lengths m and all ordered words. A nonzero value
in (4) determines the first heat term, (-1)^nu I_nu t^nu/nu!.
If it is zero, the onset is later and (1) computes the next coefficients.
Thus the minimum walk weight is a rigorous lower bound; its attainment
as the actual first order requires evaluating the signed sum.

Indeed, a nonzero term in (1) needs a_j>=s_(i_j i_(j+1)). This proves
the lower bound. At the minimum, equality must hold at every gap,
which gives (4). If the graph of finite couplings w_ij is disconnected,
no covering word exists and every mixed moment and H_F vanish. In a
finite N-vertex graph, Cayley-Hamilton makes testing c_ij(a)=0 for
0<=a<N sufficient to certify vanishing for every a.

The relation to spatial geometry is explicit. If d_ij is the minimum
graph distance between endpoints of e_i and e_j, then s_ij>=d_ij.
At that distance,

\[
 c_{ij}(d_{ij})=(-1)^{d_{ij}}
  \sum_{u\in e_i,\,v\in e_j} b_i(u)b_j(v)N_{d_{ij}}(u,v),   \tag{5}
\]

where N_d counts length-d shortest paths (and is zero for more distant
pairs). A shortest matrix-power connection uses only off-diagonal
Laplacian entries. Equation (5) can cancel across endpoint pairs;
separation alone therefore need not determine the coupling order.
The coefficients beyond it also record degrees and alternative walks.

## 3. An explicit formula for three defects

Write a=w_12, b=w_23, c=w_31. Every minimum covering walk on three
vertices is either a triangle using each edge once or a doubled
spanning path. Consequently

\[
 \nu=\min\{a+b+c,\,2(a+b),\,2(b+c),\,2(c+a)\}.             \tag{6}
\]

Terms containing an infinite weight are omitted. A closed covering
walk corresponds to a connected Eulerian multigraph on these three
vertices. If all three edge multiplicities are odd, removing excess
pairs leaves the triangle; if they are even, removing excess pairs
leaves a doubled spanning path. Positive weights make these the only
minimizers.

The exact coefficient is

\[
\begin{split}
 \frac{I_\nu}{\nu}={}&
 -2\gamma_{12}\gamma_{23}\gamma_{31}
       \mathbf1_{a+b+c=\nu}\\
 &+\gamma_{12}^2\gamma_{23}^2\mathbf1_{2(a+b)=\nu}
  +\gamma_{23}^2\gamma_{31}^2\mathbf1_{2(b+c)=\nu}
  +\gamma_{31}^2\gamma_{12}^2\mathbf1_{2(c+a)=\nu}.          \tag{7}
\end{split}
\]

The triangle has six ordered label words, yielding 6nu/3=2nu.
Each doubled spanning path has four, yielding 4nu/4=nu.
Absent couplings simply remove the corresponding terms.

### Repetition can be necessary

In G=K_4, select the path edges (0,1),(1,2),(2,3), oriented in that
order. Since L=4I-J and all b_i have coordinate sum zero,
c_ij(a)=4^a b_i^T b_j. The two end edges have zero coupling at every
order; adjacent selected edges have gamma=-1 and weight one. There
is no finite-weight Hamiltonian triangle. The doubled path has nu=4,
I_4=4, and

\[
 H_F(t)=t^4/6+O(t^5).                                      \tag{8}
\]

### Minimum-order terms can cancel

Use vertex set {0,1,2,3,4,5} and edges

    (3,5), (2,4), (1,5), (0,5), (1,2), (1,3), (3,4), (4,5).

Select e_1=(3,5), e_2=(2,4), e_3=(1,5), in these orientations.
Exact integer powers give

\[
 (s_{12},s_{23},s_{31})=(2,1,0),\qquad
 (\gamma_{12},\gamma_{23},\gamma_{31})=(-1,-2,1).
\]

Thus (a,b,c)=(3,2,1) and nu=6. The triangle term in I_6 is -24;
the doubled path through label 3 contributes +24. They cancel.
Direct exact powers, equivalently (1), give

\[
 I_0=\cdots=I_6=0,\quad I_7=-56,\quad I_8=-1288,
 \qquad H_F(t)=t^7/90+O(t^8).                              \tag{9}
\]

This example rules out characterizing the exact onset solely by a
shortest covering walk. Equation (4) also needs its signed amplitudes.

## 4. On an ambient tree, geometry determines the leading term exactly

Let G be a finite tree or a bounded-degree infinite tree. Select k>=2
distinct edges F. Let T be the minimal finite subtree containing those
edges, let s=|E(T)|, and let ell be the number of leaf edges of T.
Every leaf edge of T belongs to F. Define

\[
 \nu=2s-\ell,\qquad
 p(T)=\prod_{v:\deg_T(v)\ge2}(\deg_T(v)-1)! .              \tag{10}
\]

**Tree geometry theorem.** The first nonzero heat term is

\[
 H_F(t)=(-1)^{k-\ell}
       \frac{p(T)}{(2s-\ell-1)!}\,t^{2s-\ell}
       +O(t^{2s-\ell+1}).                                 \tag{11}
\]

All preceding mixed Laplacian moments vanish, and

\[
 I_\nu=(-1)^{\nu+k-\ell}\nu p(T).                        \tag{12}
\]

Extra branches of G outside T do not alter this leading coefficient.
The assertion concerns the small-time sign; higher coefficients and
signs at later times require further information.

### Proof

The bridge theorem reduces the interaction to the ell terminal edges
of T, with the sign (-1)^(k-ell). Indeed, the leaf edges of the quotient
after cutting F are exactly the extreme selected edges, which are the
leaf edges of its spanning subtree T. Hence it suffices to use only
these terminal edges as the selected defects.

Let v_i be the leaf vertex of terminal edge i. Orient every incidence
from its interior endpoint to v_i. For distinct terminal edges, the
unique shortest connection between their supports uses the interior
endpoints. Its length is d_T(v_i,v_j)-2. Therefore

\[
 w_{ij}=d_T(v_i,v_j)-1,\qquad
 \gamma_{ij}=(-1)^{d_T(v_i,v_j)-2}.                        \tag{13}
\]

A minimum covering word has no consecutive identical labels: removing
such a repeat preserves coverage and reduces its weight by one. A
covering cyclic word of m terminal labels without these repetitions
concatenates unique paths between consecutive leaves. Every edge of T is traversed at least
twice, and every occurrence of a leaf traverses its terminal edge once
in each direction. The terminal edges therefore contribute exactly
2m traversals, and the s-ell internal edges contribute at least
2(s-ell). It follows that

\[
 \sum_jw_{i_j i_{j+1}}
  =\sum_j d_T(v_{i_j},v_{i_{j+1}})-m
  \ge m+2(s-\ell)\ge2s-\ell.                            \tag{14}
\]

An optimal word has m=ell, each terminal label once, and traverses
every tree edge exactly twice. Such words exist: follow a contour
traversal of a plane embedding of T. Their cyclic products all have
absolute value one and sign (-1)^(nu-ell), by (13).

The number of oriented cyclic leaf orders with this optimal traversal
is p(T). To see this, prescribe a cyclic order of incident edges at
each internal vertex; there are (deg_T(v)-1)! choices there. Every
such rotation system on a tree produces one contour cycle, as an
induction removing a leaf verifies. Its leaf order traverses every
edge exactly twice. Conversely, an optimal cyclic leaf order traverses
each directed edge once and determines these local cyclic orders.
Thus this correspondence is a bijection; reversing every order counts
the reversed contour separately when it is distinct.

Each cyclic leaf order has ell ordered starting points in (4), so its
contribution is nu times (-1)^ell times (-1)^(nu-ell). Summing all p(T)
orders gives (-1)^nu nu p(T), and the bridge reduction restores the
factor (-1)^(k-ell). Substitution into the heat Taylor series proves
(11). All quantities are local to the finite subtree and its relevant
finite neighborhoods, so the argument also applies to the infinite
bounded-degree tree via local stabilization. QED.

### Consequences and checks

For two terminal cuts, p(T)=1 and nu=2s-2. This recovers the line pair
formula with one interior interval of length s-1. For a k-arm spider
with only its terminal edges selected and arm lengths summing to s,
p(T)=(k-1)! and the heat begins

\[
 \frac{(k-1)!}{(2s-k-1)!}\,t^{2s-k}.                       \tag{15}
\]

For arms (2,2,2), this is t^9/20160; for arms (2,2,2,2), it is
t^12/6652800, explaining the earlier exact fixtures. A binary quartet
with edges (0,1),(0,2),(0,3),(1,4),(1,5), selecting its four terminal
edges, has s=5, ell=4, p(T)=4, and first heat term t^6/30. The
four-arm spider with arm lengths (2,1,1,1) has the same s and ell but
p(T)=6 and first term t^6/20. Thus even when the number of leaves
and span size agree, branching degrees can change the leading amplitude.
The binary quartet and the six-vertex cancellation example were also
verified independently using exact subset matrix powers and direct
cyclic-word enumeration.

## 5. Scope of the geometry-analysis connection

The full collection c_ij(a) connects local graph geometry to all heat
interaction coefficients through (1). Its first nonzero orders provide
a finite weighted coupling geometry, and its signed amplitudes detect
cancellation invisible to separation alone. For an ambient tree, unique
paths remove that ambiguity enough to give the exact geometric formula
(11), including onset, leading amplitude, and small-time sign.

These are statements about one observable and its controlled domain.
The bridge reduction remains an identity of graph elements across all
applicable observables. The cyclic formulas alone do not assert that
heat determines the full local graph element, nor that small-time signs
persist for all t. The established moment-profile bounds provide the
convergence framework; sharper spatial estimates can use the exact
vanishing orders identified here.

## 6. Certified examples, implementation and verification

The [independent verifier](../../python/examples/higher_interaction_verification.py)
uses full integer matrix powers on edited subsets and direct cyclic-word
enumeration. It checks the repeated-label and three-defect cancellation
examples, a four-defect example with two consecutive canceled orders,
and the binary quartet. It also exhaustively checks all 249 nonempty
selected cut sets of the 14 unlabelled trees through six vertices,
including 51 single cuts. These agree with the tree formula; single
cuts have the separate leading term 2t.

The four-defect cancellation graph has edges (0,1),(0,2),(0,4),(1,2),
(1,3), with every edge except (1,2) selected. The minimum covering
weight is five. At order five, terms with four and five defect letters
give +10 and -10. At order six, the contributions +264,-408,+144
also cancel. The first nonzero moment is I_7=28, so heat begins
-t^7/180. Both exact enumeration routes agree.

For a separate time-dependent sign example, take K_(1,4) centered at
zero, add edge (1,2), and select (0,1),(0,3),(0,4). The leading heat
term is +t^3, but rational certificates show

    H_F(1) in (0.02567457, 0.02567459),
    H_F(3) in (-0.03857343, -0.03857341).

These decimal intervals are outward relaxations of the exact rational
enclosures in the [result record](../../python/results/higher_interaction_verification.json).
Their strict opposite signs prove a crossing between the two times.

Library version 0.5.0 implements `interaction_moments(X, order)` using
the incidence cross moments and a dynamic program indexed by used-label
masks, last labels and polynomial degree. A mask records support; it
does not discard repeated occurrences. The method performs no rooted
isomorphism calculation and constructs no edited-subset matrices, but
its worst-case work remains exponential in the number of active edits.
`max_work` makes that limitation explicit. Results contain both
Laplacian and uniformized moments; a zero computed prefix alone is
not reported as an identically zero interaction.

`tree_interaction_leading(X)` computes (11) by pruning unselected
exterior branches. It returns the span, terminal-edge count, branching
factor, exact order and rational coefficient, including normalization.
Runtime tests compare with independent full-matrix subset sums, mixed
insertions/deletions, bridge reductions, and harmless exterior branches.

The [decay note](INTERACTION_DECAY.md) converts support distances into
moment and heat bounds. Those estimates provide geometric control
even where signed cancellation complicates the exact leading term.

The rank-one transfer and determinant methods belong to established
matrix-function theory. The primary low-rank and many-body references
in [PLANAR_DEFECTS.md](PLANAR_DEFECTS.md#6-prior-work-and-the-comparison-to-make)
remain relevant. The results here concern their exact use with this
completion's graph interactions; computational superiority and novelty
are not inferred from these formulas.
