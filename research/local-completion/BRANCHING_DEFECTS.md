# Branching bridge cuts and irreducible graph interactions

30 September 2026. These results concern actual signed graph
identities, hence apply to every observable for which the corresponding
finite or controlled limiting elements are defined. Inclusion-exclusion
and connected-component decomposition are classical; the contribution
here is their explicit use and quantitative interpretation in this local
completion. No novelty claim about cluster expansions is made.

## 1. Definitions and the finite reduction theorem

Let G be a connected finite simple graph, and let F be a nonempty set of
k selected edges. Define its irreducible cut interaction by

    C_F(G) = sum_{S subset F} (-1)^(k-|S|) [G\S].                (1)

Disjoint union is addition in the graph algebra. For k>=1 the alternating
coefficients sum to zero, so equivalently one may replace [G\S] by the
relative element [G\S]-[G]. For k=0 the two conventions differ; an API
returning zero for an empty defect set uses the relative convention.

Suppose every edge of F is a bridge of G. Removing all of F gives k+1
connected blocks. The quotient Q has one vertex for each block and one
edge for each selected cut. It is a tree. Cycles inside the individual
blocks cause no difficulty. Conversely, if one retains every selected
edge in the quotient, including possible loops and parallel edges, that
quotient is a tree precisely when the selected edges are bridges.

Let L subset F be the edges incident to a leaf vertex of Q. Put ell=|L|.
For k>=2, distinct leaves have distinct incident edges, so ell is also
the number of leaf vertices. For k=1 both quotient vertices are leaves
but L consists of a single edge; this case is handled separately.

**Theorem.** The exact finite graph-algebra identity is

    C_F(G) = (-1)^(k-ell) C_L(G).                 (2)

Thus selected internal edges of the quotient need not be included in
the irreducible interaction calculation. Their contribution is the
parity factor in (2). Those underlying edges remain present in G and
can still influence its geometry and every spectral calculation; only
their selection as additional defects has been eliminated.

### Proof by tracking one connected component

For a nonempty connected set U of quotient vertices, let G_U be the
graph obtained by taking all original blocks in U and every original
edge between them. Since Q is a tree, G_U is a component of G\S exactly
when all boundary edges delta_Q(U) are cut, no edge internal to U is
cut, and cuts on the exterior quotient Q[V(Q)\U] are arbitrary.
Consequently its coefficient in (1), before combining isomorphic terms,
is

    (-1)^(k-|delta_Q(U)|)
       sum_{T subset E(Q[V(Q)\U])} (-1)^|T|.

It vanishes unless the exterior has no edges. If it survives, its
coefficient is (-1)^|E(Q[U])|.

For a tree, a connected U whose complement is independent can omit
only leaves. Indeed, an omitted nonleaf has at least two neighbors;
independence forces all those neighbors into U, but their unique path
runs through the omitted vertex, contradicting connectedness of U.
Conversely, deleting any subset of leaves from a tree with at least
three vertices leaves a nonempty connected set. Therefore, for k>=2,

    C_F(G) = sum_{J subset Leaves(Q)}
                 (-1)^(k-|J|) G_(V(Q)\J).        (3)

Contracting the internal selected edges turns Q into the star with
ell leaf edges, namely the quotient associated to L. Its surviving
terms are exactly the same G_(V(Q)\J), with coefficients
(-1)^(ell-|J|). This proves (2). For k=1 one has L=F and the conclusion
is immediate. Coinciding unlabelled graph types are simply added; their
possible additional cancellation does not affect the identity. QED.

Formula (3) also constructs the answer directly with at most 2^ell
connected graph terms, instead of evaluating 2^k edited full graphs.
The quotient vertices represent entire uncut connected blocks, rather
than necessarily single vertices.

## 2. Quantitative edit certificates

Each term [G\S]-[G] has edit budget |S|. The direct expansion gives

    q(C_F) <= sum_{S subset F}|S| = k 2^(k-1).

The reduced representation (2) improves this to

    q(C_F) <= ell 2^(ell-1).                      (4)

Both statements concern certified upper bounds, not a claim that the
minimum possible edit budget has been found. Maximum degree is bounded
by that of G because all edits are deletions. In particular the existing
relative-heat estimates apply using the reduced budget.

For a path-like quotient ell=2, equation (2) recovers the parity
reduction to the two extreme cuts, with budget four. A genuine branch
can require three or more terminal cuts. The complexity depends on
this boundary count; a tree with many leaves need not admit a large
reduction. For a star every selected edge is already a leaf edge.

## 3. Branching changes higher-interaction signs

Take G=K_(1,3) and select all three edges. Equation (3) becomes

    C_F = -K_(1,3) + 3P_3 - 3K_2 + K_1.          (5)

Using the elementary Laplacian spectra gives

    H_t(C_F) = exp(-t)[1-exp(-t)]^3 > 0,  t>0.    (6)

This contrasts with the negative three-cut interaction of an ordered
line, proved in DEFECT_INTERACTIONS.md. A universal rule assigning a
higher-interaction sign solely from the number of cuts therefore fails
even among trees.

More generally, for the k-edge star with k>=2,

    C_F = sum_{j=0}^k (-1)^j binom(k,j) K_(1,j),
    H_t(C_F) = exp(-t)[1-exp(-t)]^k.              (7)

Here K_(1,0)=K_1. To verify the heat identity, for j>=0 write the star
heat trace as

    1+(j-1)exp(-t)+exp(-(j+1)t).

This expression also equals one at j=0. The alternating binomial sum
annihilates the constant and affine-in-j terms for k>=2, leaving (7).
The k=1 cut has its usual separate expression 2K_1-K_2 and heat
1-exp(-2t).

As a nontrivial reduction example, subdivide one arm of K_(1,3) once
and select all four edges of the resulting tree. Its quotient has
three leaf edges and one internal edge. The complete four-edge
interaction equals minus the interaction of its three leaf edges on
the same underlying subdivided star. This is an identity of graph
combinations, regardless of the observable subsequently chosen.

## 4. Infinite bounded-degree trees and bridge-connected graphs

Let G now be a connected locally finite graph with a uniform finite
maximum-degree bound D, and let F be finitely many bridges. In
particular G may be any bounded-degree infinite tree. Individual
infinite components after a cut are not assigned unnormalized graph
numbers. Instead define each relative cut response by finite connected
induced exhaustions:

    D_S = lim_n ([G_n\S]-[G_n]),  S subset F.

Take G_n increasing and containing the selected edges and arbitrarily
large neighborhoods of their endpoints. At radius r, only roots within
r of edited endpoints in either graph can have changed balls. These
are finitely many roots. Once the exhaustion contains their r-balls,
all further terms cancel exactly. For example including the original
2r-neighborhood of every selected endpoint suffices: deletions cannot
shorten distances, and each relevant ball lies in that neighborhood.
Thus every local marginal stabilizes exactly, including every
polynomial weight, and the limits exist in A independently of the
chosen exhaustion.

Each D_S has degree certificate D, mass zero, and edit budget at most
|S|. The infinite irreducible interaction is

    C_F = sum_{S subset F}(-1)^(k-|S|)D_S.

Removing F still gives a finite quotient tree of its k+1 components,
even when the blocks themselves are infinite. Let L be its leaf edges.
For sufficiently large connected exhaustions the selected-cut quotient
has exactly the same incidence tree. Applying the finite identity and
taking the locally stabilized limit gives

    C_F = (-1)^(k-ell) C_L,
    q(C_F) <= ell 2^(ell-1).                      (8)

This supports certified relative heat and other controlled observables
without requiring a finite signed-measure representation. Whether a
particular interaction has finite marginal variation is a separate
question; no universal beyond-finite-measure claim is made merely from
its having an infinite ambient tree.

For a three-ray tree, cut the three edges incident to its central
vertex. At radius one all noncentral-root contributions cancel in the
three-fold interaction, while the central root gives

    -delta_(K_(1,3),center) + 3delta_(P_3,center)
       -3delta_(K_2,root) + delta_(K_1,root).

The local variation at that radius is eight. This is a direct example
where a genuine three-branch local pattern survives. It cannot be
represented as an interaction of two extreme cuts on a line.

## 5. What remains true for selected edges on cycles

For general selected edges, the quotient of the fully cut graph is a
multigraph: a selected edge may become a loop, and several selected
edges may join the same two blocks. Omitting loops or merging parallel
edges would lose information essential to the interaction.

A general component expansion still explains the cancellation. Let U
be a nonempty set of full-cut blocks. Let F_int(U) be the selected edges
whose endpoints are both in those blocks, F_ext(U) the selected edges
whose endpoints both lie outside, and A a subset of F_int(U). Form the
graph H_(U,A) by taking the uncut blocks in U and restoring exactly A.
Tracking its occurrences in the alternating sum gives

    C_F(G) = sum_{U: F_ext(U) is empty}
               sum_{A subset F_int(U): H_(U,A) connected}
                    (-1)^|A| H_(U,A).             (9)

Indeed, the selected boundary edges must be cut and the restored
internal edges must be exactly A; exterior cuts are arbitrary and
cancel unless F_ext(U) is empty. Different (U,A) pairs may yield
isomorphic graphs, in which case their coefficients combine.

On a tree quotient, connectedness forces A to contain every internal
edge and connected U with independent complement omits only leaves;
this is precisely the earlier reduction. On a cyclic quotient there
can be many connected choices of A, and a connected U with independent
complement need not omit only leaves. The leaf-edge theorem therefore
does not extend to general cyclic connections.

For the explicit smallest example, take G=C_3 and select its three
edges. There are no quotient leaves, but

    C_F = -C_3 + 3P_3 - 3K_2 != 0,
    H_t(C_F) = -[1-exp(-t)]^3 < 0,  t>0.          (10)

This both disproves a leaf-only reduction without the bridge condition
and supplies a cyclic interaction with a different sign from the
three-edge star. Genuinely two-dimensional lattice defects generally
fall in this cyclic case: graph-theoretic bridge reduction should be
checked, rather than inferred from the visual arrangement of cuts.

## 6. Implementation, verification and interpretation

Library version 0.4.0 implements (2) in `bridge_cut_reduction` and
`EdgeInteraction`. It validates all edits, checks that the original graph
is connected and the fully cut graph has k+1 components, and retains the
leaf-incident selected edges. The subset-work budget is applied after
reduction. Mixed insertions/deletions and cyclic cuts are supported by
the general inclusion-exclusion constructor without bridge reduction.
The empty edit list returns zero by the relative convention.

The independent [exact verifier](../../python/examples/branch_planar_verification.py)
enumerates all 25 unlabelled trees with one through seven vertices and
their 942 nonempty cut sets. All 942 component-basis identities and
3,763 rooted-histogram identities passed; 370 selections have a proper
internal-edge reduction. It also checks branching stars, subdivided
arms, the triangle obstruction, and buffered square-lattice defects.
The [result record](../../python/results/branch_planar_verification.json)
contains exact integer moments and rational leading heat coefficients.

The runtime tests additionally cover cycles inside uncut bridge-connected
blocks, work-budget reduction, normalization, mixed edits and certified
star/triangle heat enclosures against independent 70-digit formulas.
These finite checks support the implementation; the argument above proves
the identity for arbitrary selected bridge sets.

The verification covers the following distinctions:

- Enumerate all cut subsets for small trees and compare the exact finite
  graph combination with (2) and (3), allowing isomorphic terms to merge.
- Include a path quotient, a three-leaf star, a subdivided star with
  internal selected edges, and a graph with cycles inside its uncut
  blocks but only bridges selected.
- Check the degree, mass-zero, and reduced edit-budget metadata.
- For an infinite tree model, compare growing finite induced exhaustions
  at a fixed radius and verify exact stabilization before using heat
  truncation certificates.
- Verify the three-star and triangle formulas independently by their
  full Laplacian eigenvalues. They prevent an incorrect universal sign
  assertion based on cut count.
- Test cyclic quotient cases against (9), including selected edges that
  become loops or parallel edges after the full-cut contraction.

The reduction uses graph decomposition before selecting an observable.
Its same resulting element can feed local motifs, Laplacian moments,
heat times, and resolvents with the appropriate existing certificates.
Conventional graph or matrix algorithms can also use these identities;
any computational advantage specific to the library must be measured
against an implementation that is allowed the same structural reduction.
