# Entire-function arithmetic of finite deletions

1 October 2026. Repaired finite-deletion proof from the
[development plan](../../reviews/local-completion-2026-09-30/DEVELOPMENT_PATH.md).
Independent referee review of this new note is pending. In particular the
edited graph is allowed to be disconnected and to have finite components.

Let Gamma be an infinite connected locally finite simple graph that is
vertex-transitive, bipartite and d-regular, with finite d>=2. Delete a finite
nonempty edge set epsilon. Write Gamma' for the result, S for the endpoints
of the deleted edges, and m=|S|<=2|epsilon|. All rooted balls are induced.
Write A_C for the complex local Cartesian graph completion.

## 1. The defect and its affected roots

Let G_n be the finite graph induced by vertices within distance n of S
in Gamma, and let G'_n delete epsilon from G_n. For each fixed radius r,
T_r(G'_n-G_n) stabilizes when n>=2r: roots farther than r-1 from S are
unaffected, and the r-balls of the remaining roots lie within distance
2r-1 of S. Stabilization is exact, including every polynomial weight.
These finite signed elements therefore converge to a defect D in A_C.

For r>=1 define

\[
V_r=\{o:\operatorname{dist}_\Gamma(o,S)\le r-1\},\qquad c_r=|V_r|,
\qquad b_r=B_r(\Gamma,o_0).
\]

Vertex transitivity makes b_r independent of o0. Then

\[
T_rD=A_r-c_r\delta_{b_r},\qquad
A_r=\sum_{o\in V_r}\delta_{B_r(\Gamma',o)}.                   \tag{1}
\]

**Exact affected-root assertion.** In a bipartite graph the endpoints of
an edge have opposite distance parity from a root and their distances
differ by one. Thus an edge contained in its r-ball has an endpoint at
distance at most r-1. If o is outside V_r, no deleted edge can occur in
the ball or in a path of length at most r, and its ball is unchanged.

If o is in V_r, take an endpoint u in S nearest to o. A shortest o-to-u
path avoids all deleted edges: the first deleted edge on such a path
would have an endpoint in S closer than u. Hence u is still at distance
at most r-1 in Gamma', and its full degree, now less than d, is visible.
If o itself is in S its root degree drops; otherwise its root degree is d
and this interior degree drop distinguishes its ball from b_r. In either
case B_r(Gamma',o) is not b_r. Therefore

\[
\|T_rD\|_1=2c_r,\qquad c_r\longrightarrow\infty.             \tag{2}
\]

The divergence follows since the finite neighborhoods of the nonempty
set S exhaust the infinite connected graph. Radius zero has T0(D)=0.

## 2. The finite regular-component exception

Let F_r be the interior-regular face of M_r: all vertices at distance
less than r have the root's degree. The
[face lemma](MIXED_MEDIUM_ARITHMETIC.md#2-a-multiplicative-face-of-locally-regular-balls)
gives B star_r C in F_r if and only if B,C are in F_r.

**Classification lemma.** For r>=m+1, the atoms of A_r in F_r are exactly
the whole rooted graphs of the finite regular components of Gamma'.
Their total coefficient mass N is their total number of vertices, and

\[
N\le m\le2|\epsilon|.                                      \tag{3}
\]

**Proof.** An affected root outside S has degree d and sees the deficient
nearest endpoint from Section 1 in its interior, so its ball is not in F_r.
For a root o in S, its degree in Gamma' is less than d. If its component
contains any vertex outside S, a shortest path to the first such vertex
has at most m edges: preceding vertices are distinct members of S. That
vertex has degree d and lies in the interior for r>=m+1, again excluding
F_r. Otherwise the component is contained in S, hence has at most m
vertices and is wholly visible. Its ball lies in F_r precisely when the
component is regular. Conversely any finite regular component must have
degree less than d: a finite component with every degree still d could
not have been separated from the connected infinite Gamma by deletions.
All its vertices have therefore lost an incident edge and belong to S.
All of its rooted versions occur in A_r. QED.

This counts rooted mass, not components. Isolated vertices are regular
components of degree zero and are included.

## 3. Character discs from Rouché's theorem

For |w|<=1 define a semicharacter on M_r by w^(root degree) on F_r and
zero outside F_r, with 0^0=1. The face property and additive root degree
make it multiplicative and unital. Its modulus is at most one, so it
defines a continuous character chi_(r,w) of A_C.

Let the finite regular components be C_j of degrees d_j<d. For r>=m+1,
the classification lemma gives

\[
\chi_{r,w}(D)=q(w)-c_rw^d,\qquad
q(w)=\sum_j |C_j|w^{d_j},\qquad |q(w)|\le N\quad(|w|\le1).   \tag{4}
\]

In particular q is independent of r in this range. Put R_r=c_r-N.
For |lambda|<R_r and |w|=1,
|q(w)-lambda|<c_r=|c_rw^d|. Rouché's theorem says that
q(w)-lambda-c_rw^d has d zeros in the open unit disc, counting
multiplicity. Thus every such lambda is a character value of D.
The character image is compact, so it also contains the closed disc
of radius R_r. For every polynomial f,

\[
\max_{|z|\le R_r}|f(z)|\le\|T_rf(D)\|_1.                    \tag{5}
\]

Here R_r tends to infinity. Equation (5), rather than exclusion of every
positive atom from F_r, is the required lower estimate.

## 4. The entire-function algebra and ambient units

**Theorem.** The map f to f(D) identifies the entire functions O(C), with
their compact-open topology, with the closed unital subalgebra generated
by D. Furthermore

\[
\sigma_{A_{\mathbb C}}(D)=\mathbb C,\qquad
f(D)\text{ is a unit}\ \Longleftrightarrow\ f\text{ has no zeros in }\mathbb C,
\qquad \sigma_{A_{\mathbb C}}(f(D))=f(\mathbb C).              \tag{6}
\]

**Proof.** Local submultiplicativity gives, with p=p_(r,k)(D),

\[
p_{r,k}\Bigl(\sum_n a_nD^n\Bigr)\le\sum_n |a_n|p^n
\le2\sup_{|z|\le2\max(1,p)}|f(z)|.                          \tag{7}
\]

The last inequality follows from Cauchy's coefficient estimate on that
larger disc. Thus every entire series defines an element and multiplication
agrees with multiplication of entire functions. The character bound (5)
passes to these convergent series and controls every compact-open
seminorm, since R_r tends to infinity. The map is injective with continuous
inverse onto its image. Completeness, or applying these bounds to any
convergent sequence of polynomial expressions, shows that its image is
closed and is exactly the closed generated algebra.

Every complex number is attained by some character value of D. A zero
of f therefore prevents f(D) from being a unit. If f is zero-free, its
entire reciprocal supplies the inverse in the same closed subalgebra.
Applying this equivalence to f-lambda proves (6). Real Taylor
coefficients give the corresponding real form. QED.

Only constants in this subalgebra have finite signed rooted-graph measures:
a uniform bound on ||T_r f(D)||1 would, by (5), make f bounded on the
whole plane, hence constant by Liouville. For exponential units the estimates
give, at r>=m+1,

\[
e^{|t|(c_r-N)}\le\|T_r e^{tD}\|_1\le e^{2|t|c_r}.           \tag{8}
\]

These are bounds; an exact norm formula for general deletions is not claimed.

## 5. Edge cases, finite checks and scope

For the line with both edges at vertex zero deleted, c_r=2r+1, N=1
and q(w)=1. The isolated vertex survives in the regular face. If D'
denotes the vertex-deletion defect, D=1+D'; these have the same closed
unital generated algebra. Isolating a two-vertex line segment instead
gives N=2 and q(w)=2w, illustrating why a component count is wrong.
Isolating a lattice square gives N=4 and q(w)=4w². A split-off three-vertex
path is not regular and contributes zero once the full component is visible.

The [foundation verifier](verify_foundation_checkpoint.py) independently
extracts balls from implicit line and square-lattice adjacency, applies
the actual edge deletions, and classifies interior regularity. It compares
the resulting positive regular mass with (3)–(4), including the isolated
vertex, edge, square and nonregular path. The finite buffers and tested
radii are recorded explicitly in
[foundation_checkpoint_results.json](foundation_checkpoint_results.json).
These fixtures target the finite-component exception and affected-root
count; the infinite-radius and entire-function assertions rest on the proofs.

Bipartiteness excludes edits confined to a boundary sphere. Vertex
transitivity supplies one negative background atom. A non-bipartite or
nontransitive extension needs new hypotheses. More generally, the same
Rouché argument works for a regular homogeneous negative background if
the positive mass retained by F_r is a_r and c_r-a_r tends to infinity;
this is a sufficient criterion, not a verification of those hypotheses
for further media. Insertions and degree-preserving edits, joint
independence of different backgrounds, and E/P independence remain open.
