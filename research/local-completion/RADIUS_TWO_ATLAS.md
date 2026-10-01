# Radius-two atlas, finite balanced slices, and decorated-neighbor coordinates

1 October 2026. Step 4 of the development plan. Computational checkpoint and
new written proofs; independent referee review is deferred at the author's
request. This note concerns the **local Cartesian graph completion**.

The atlas covers every radius-two rooted simple graph with at most nine
vertices and maximum degree four. A second catalogue covers **all**
radius-two types of maximum degree three: the Moore bound is ten vertices.
The computations find unique prime factorizations for every type in these
catalogues. They also support two general results proved below: a basis for
every finite supported balanced slice, and additive decorated-neighbor
coordinates with a useful sufficient irreducibility test. General freeness
of the rooted-ball monoid remains open.

## 1. Reproducible catalogues

Sources and data are in [radius_two_atlas/](radius_two_atlas/).

| Catalogue | Rooted ball types | Connected host graphs | Nonunit decomposable types | Irreducible types |
|---|---:|---:|---:|---:|
| Degree at most 3, through 10 vertices | 443 | 2,571 | 9 | 433 |
| Degree at most 4, through 9 vertices | 29,032 | 14,598 | 69 | 28,962 |

The identity type is excluded from both factorization columns. All graphs
have a specified root at vertex zero. The second catalogue is a size-bounded
part of the degree-four slice; degree-four radius-two balls can have up to
17 vertices. Neither catalogue is closed under unrestricted multiplication.

Exact numbers of rooted types by vertex count:

| Vertices | Degree at most 3 | Degree at most 4 |
|---:|---:|---:|
| 1 | 1 | 1 |
| 2 | 1 | 1 |
| 3 | 3 | 3 |
| 4 | 10 | 10 |
| 5 | 20 | 47 |
| 6 | 42 | 183 |
| 7 | 74 | 895 |
| 8 | 107 | 4,443 |
| 9 | 101 | 23,449 |
| 10 | 84 | Outside the recorded degree-four scope |

### Enumeration and canonicalization

`atlas.cpp` starts with the one-vertex graph. At each stage it appends one
vertex with every nonempty neighbor subset compatible with the degree cap,
then removes isomorphic duplicates. This generates every connected graph:
remove a leaf of a spanning tree to obtain a connected induced graph on one
fewer vertex, and apply induction. Vertex deletion preserves the degree cap.
Taking induced BFS balls at every root of every host generates the ball
catalogue. Conversely, each ball in the claimed slice occurs as its own
finite host, so the catalogue is exhaustive.

Canonicalization uses ordered equitable partitions and exhaustive
individualization. At a discrete partition the full adjacency mask is
compared. The only branch pruning exchanges twins whose transposition fixes
all other vertices. Fixed roots occupy initial singleton cells. These are
exact isomorphism keys, rather than probabilistic hashes. The key need not
be the minimum mask across *all* permutations; it is the minimum over this
isomorphism-invariant search tree.

The independent verifier enumerates every labelled connected graph through
six vertices and canonicalizes by explicit degree-preserving permutations.
It compares whole sets of host and rooted isomorphism classes, including
duplicate detection. Full-scope enumeration beyond six uses the algorithm
and argument above; it has not received the separately deferred review.

### File format

Each catalogue has:

- `balls.tsv`: ball ID, vertex count, adjacency mask; root zero.
- `hosts.tsv`: host ID, vertex count, adjacency mask.
- `histograms.tsv`: host ID followed by `ball_id:multiplicity` entries.
- `products.tsv`: every retained unordered nonunit pair and its product ID.
- `results.json`: factorization witnesses, coordinate ranks and collisions,
  basis host IDs, rerooting pairs, rank certificates, and data SHA-256 hashes.

Masks use the pairs `(0,1),(0,2),...,(n-2,n-1)` in that order, least bit first.
IDs are local to a catalogue. This is unnormalized vertex-counting data.

## 2. Complete factorization within the catalogued slices

For radius-two balls B and C, write d(B) for the root degree. Their product
has the coordinate axes and one mixed vertex for each pair of root neighbors:

\[
 |B\star_2 C|=|B|+|C|-1+d(B)d(C),\qquad
 d(B\star_2 C)=d(B)+d(C).
\]

The generator enumerates every unordered pair satisfying the necessary
size/root-degree bounds, constructs the product, and retains it exactly
when its maximum degree is within the cap. The verifier independently
materializes full Cartesian graphs and extracts their balls by BFS; it
also checks that no admissible pair was omitted.

Each factor embeds as an induced coordinate axis. Thus every factor of a
catalogued ball is within its size and degree bounds. A nonunit factor is
strictly smaller than the product. Recursively expanding all binary
decompositions therefore gives every prime factorization in the full
monoid of each catalogued ball, not just decompositions over a chosen
short list of potential generators.

There are 9 retained nonunit product pairs in the degree-three catalogue
and 71 in the degree-four catalogue. Some products have multiple binary
splittings from regrouping the same prime factors. After this regrouping,
every catalogued ball has exactly one prime multiset. This is finite
evidence for freeness; it gives no general factorization theorem for M_2
or for higher radii.

**Working conjecture.** M_2 is the free commutative monoid on its
star_2-irreducibles, so its product relations are generated by permutation
and regrouping of prime factors. The recorded product table gives finite
support for this conjecture. Larger-degree and larger-size decomposable
types are needed to test it beyond the present slice.

## 3. A general basis theorem for finite supported balanced slices

This result applies at any fixed radius, independently of the catalogue
size. Work over Q for finite realizations, or over R or C in the completion.
Let S=S(r,N,Delta) be all rooted r-ball types of at most N vertices and
maximum degree at most Delta. Let V_S be their finite-dimensional coordinate
space inside E_r. Write B_r for the closed radius-r image of the local
completion. Define H to be the unrooted connected graphs F with at most N
vertices, maximum degree at most Delta, and at least one r-center (a vertex
whose eccentricity is at most r).

**Finite-slice basis theorem.**

\[
 B_r\cap V_S = \operatorname{span}\{T_r(F): F\in H\},
 \qquad \dim(B_r\cap V_S)=|H|.
\]

The displayed finite host histograms form a basis. All rerooting constraints
on this supported slice are generated by the following finite collection.
For each F in H, choose one orbit of r-centers as reference; compare its
rooted injective embedding count with that for every other orbit of
r-centers in F.

**Proof: universal constraints.** For a rooted pattern (F,u) of radius at
most r, let J_(F,u)(B) count injective edge-preserving maps F to B taking u
to the root. For any finite host G,

\[
 \sum_{v\in V(G)}J_{(F,u)}(B_r(G,v))=\operatorname{inj}(F,G),
\]

independently of u. Every embedding is contained in the relevant r-ball
because every pattern vertex is within r of u. Also
J_(F,u)(B) is bounded by |B|^(|F|-1), so the functional is continuous in
the weighted local topology. Differences between two r-center choices
therefore vanish on B_r, by density and continuity.

**Proof: independence of the constraints.** Index the square matrix J by
source and target rooted types in S. Order both by vertex count and then
edge count. An injective edge-preserving map cannot decrease either count;
at equal counts it is an isomorphism. J is triangular with the positive
rooted automorphism counts on its diagonal. Its rows are a basis of V_S^*.
Grouping source rows by their underlying unrooted graph gives |H| groups.
Within each group the differences from a reference row are independent;
over all groups their rank is |S|-|H|. Thus the proposed common kernel has
dimension |H|, and B_r intersect V_S is contained in that kernel.

**Proof: enough independent host columns.** Every T_r(F), F in H, is
supported on S. In a nontrivial combination of these histograms, choose a
host with largest vertex count and nonzero coefficient. Its full-graph
rooted types occur at its r-centers, with positive multiplicities. No
smaller host or different host of the same size contributes to those
types. The combination cannot vanish. Thus the |H| histograms are linearly
independent and lie in B_r intersect V_S. The two dimension bounds agree,
proving both assertions. QED.

This describes an **intersection with a supported coordinate slice**.
Discarding coordinates outside S from an arbitrary balanced array can
destroy balance. The theorem also supplies a finite all-radius extension
through a linear combination of finite graphs, while making no claim
about unrestricted one-radius surjectivity or a continuous section for
the full infinite-dimensional image.

### Exact dimensions and executed certificates

| Radius | Degree cap | Vertex cap | Ball coordinates | Balanced dimension | Independent rerooting constraints |
|---:|---:|---:|---:|---:|---:|
| 2 | 3 | 6 | 77 | 47 | 30 |
| 2 | 3 | 7 | 151 | 96 | 55 |
| 2 | 3 | 8 | 258 | 180 | 78 |
| 2 | 3 | 9 | 359 | 271 | 88 |
| 2 | 3 | 10, complete at this degree | 443 | 353 | 90 |
| 2 | 4 | 5 | 62 | 31 | 31 |
| 2 | 4 | 6 | 245 | 107 | 138 |
| 2 | 4 | 7 | 1,140 | 445 | 695 |
| 2 | 4 | 9 | 29,032 | 11,975 | 17,057 |
| 3 | 2 | 7, complete at this degree | 15 | 12 | 3 |

All dimensions follow from the theorem and the exact catalogue groupings.
Separate matrices were actually evaluated for degree three through ten
vertices and degree four through five and six vertices. In these cases:

1. The integer rooted-injection constraint matrix D annihilates the integer
   host histogram matrix H exactly.
2. Elimination modulo the prime 1,000,003 gives rank lower bounds for D and H.
3. These lower bounds sum to the number of ball coordinates. Hence they
   certify the exact characteristic-zero ranks and kernel equality.

The result files retain independent host IDs and rooted-type pairs defining
the constraints. No full numerical 29,032-column rerooting matrix was
evaluated; its dimension is a consequence of the triangular theorem.
The radius-three control is separately enumerated from paths and cycles,
which exhaust connected degree-two graphs. Its two ranks and their exact
annihilation are evaluated over rational arithmetic. Higher-degree
radius-three catalogues remain outside the computation.

For comparison, the simpler transports recording only the induced union
of the endpoints' one-neighborhoods have rank 18 on the degree-three
catalogue. They leave an upper bound 425 instead of 353. This explains
why that restricted family cannot certify the full balanced image.

### Finite reconstruction

`reconstruct_supported` in `analyze.py` processes basis hosts in decreasing
vertex count. It reads the coefficient at one full-graph rooted type,
divides by that type's multiplicity in the host, and subtracts the whole
host histogram. The remaining full-graph types in the same group record
any compatibility failure. Rational arithmetic gives exact host
coefficients and a residual. The residual vanishes precisely on the
supported balanced image, by the theorem; a nonzero residual is retained
as an obstruction.

For example, with P_n the n-vertex path,

\[
 T_2(C_6/6)=T_2(P_5-P_4).
\]

The finite signed difference realizes the radius-two neighborhood of the
uniform six-cycle (also that of the infinite line). A point mass at an
endpoint-rooted P_3 fails the balance criterion. These examples are
executed independently by actual rooted-ball extraction.

## 4. Decorated-neighbor geometry gives additive coordinates

The familiar coordinate family in the computation consists of connected
root-link counts, the earlier root-edge component counts, root degree,
the scaled second sphere-log coefficient, the second open-walk cumulant,
and closed-walk cumulants of orders three through five. All are determined
at radius two. Closed moments of order six are deliberately excluded:
their general evaluation needs radius three.

The five-vertex trees with root-neighbor degree profiles (2,2) and (1,3)
have identical values of this entire familiar family. Both have two
second-sphere vertices. The distribution of those vertices between the
root branches supplies additional local information.

### Definition and product rule

For a rooted radius-two ball B with root o, form a decorated graph L(B)
on the neighbors of o. Its vertex label at u is

\[
 a(u)=\deg_B(u)-\deg_B(o).
\]

Each pair u,v carries the label

\[
 b(u,v)=\bigl(\mathbf 1_{u\sim v},\ |N_B(u)\cap N_B(v)|-2\bigr).
\]

The zero pair label is treated as absence of an edge. Connected components
retain both their vertex labels and their nonzero pair labels; isolated
labeled vertices count as components.

**Decorated-neighbor product lemma.**

\[
 L(B\star_2 C)\cong L(B)\sqcup L(C).
\]

**Proof.** Root neighbors separate into the two coordinate directions.
For a neighbor from B, its degree and the root degree both increase by
d(C); its vertex label is unchanged. Two neighbors in the same direction
keep their adjacency and all their common neighbors. Neighbors in different
directions are nonadjacent and have exactly two common neighbors: the root
and their mixed-coordinate vertex. Their pair label is therefore zero.
All vertices and edges used here are visible within radius two. QED.

Consequently the count nu_Q of each connected decorated type Q is a
nonnegative integer-valued additive coordinate on M_2. The same definition
works by radius-two truncation in every M_r with r at least two. It gives
bounded semicharacters

\[
 s_z(B)=\prod_Q z_Q^{\nu_Q(B)},\qquad |z_Q|\le1,
\]

and faces defined by forbidding any specified set of decorated component
types. Counts are at most the root degree, so their linear local
observables have the required polynomial growth.

**Irreducibility certificate.** If L(B) is nonempty and connected, B is
irreducible. Each nonunit factor has a root neighbor, and a nontrivial
product would split L(B) into at least two nonempty components.
The converse fails: K_(2,3), rooted on its two-vertex side, is a five-vertex
prime with three isolated decorated neighbors. Any product with root-degree
split 1+2 has at least six vertices by the product-size formula.

### Measured improvement and remaining collisions

| Catalogue | Familiar coordinate rank | Rank with decoration | Familiar distinct vectors | Distinct vectors with decoration |
|---|---:|---:|---:|---:|
| Complete degree-three radius-two catalogue | 11 | 63 | 151 | 188 |
| Degree-four catalogue through nine vertices | 23 | 1,456 | 3,622 | 9,505 |

Ranks are computed over exact rational arithmetic. The augmented family
includes the familiar coordinates as well as decoration counts. Each
family has fewer distinct vectors than ball types, so neither separates
its full catalogue. They are rank/separation measurements, not dimensions
of the full space of additive functions on M_2.

The connected-decoration criterion certifies 398 of the degree-three
irreducibles and 23,856 of the degree-four irreducibles. The remaining
35 and 5,106 are certified by exhaustive factor enumeration in their
respective slices. These are object counts, not accumulated assertion counts.

An explicit surviving collision in the degree-four catalogue is IDs 94
and 113 (both six vertices, rooted at zero):

- ID 94: edges 04, 05, 15, 23, 25, 34.
- ID 113: edges 04, 05, 14, 23, 25, 35.

Their familiar and decorated coordinates agree. One contains a five-cycle
with a pendant vertex; the other has a triangle attached to a path. Their
different cycle geometry establishes nonisomorphism independently of
canonicalization. Further separating coordinates and expressions in the
universal rooted-pattern cumulants remain research tasks.

## 5. Faces recorded in the atlas

The atlas records the interior-regular face from the deletion work, and the
bipartite and triangle-free faces. Each assertion is global at fixed radius:
factors embed as induced axes, and the Cartesian product of bipartite
(respectively triangle-free) graphs has the same property. An induced
product ball inherits it. The interior-regular case uses the earlier face
lemma. For each recorded product the exact face identity
1_F(B star C)=1_F(B)1_F(C) is also checked.

| Face | Degree-three members | Degree-four members |
|---|---:|---:|
| Interior regular | 206 | 1,604 |
| Bipartite | 47 | 346 |
| Triangle free | 183 | 1,322 |

The zero sets of the nonnegative decorated-component counts give further
proved faces. This is a catalogue of useful faces and a general construction,
with a complete classification of faces still open.

## 6. Reproduction and status

From the repository root, with Python 3.10+ and a C++17 compiler:

```sh
python -S research/local-completion/radius_two_atlas/run.py
```

The driver builds in a temporary directory, regenerates both catalogues,
runs exact analysis and the separate finite oracles, and writes a manifest
of source/data hashes. It requires no graph library, network service, or
floating-point linear algebra. The reported finite rank certificates are
integer calculations modulo a specified prime; the coordinate ranks use
rational arithmetic.

The verifier separately checks catalogue coverage through six vertices,
every retained/eligible product pair, all degree-three host histograms,
all degree-four hosts through seven vertices plus a recorded sample beyond,
small injection counts, finite reconstruction, and explicit geometric
counterexamples. The general proofs remain written proof checkpoints.

**Step 4 is complete as the planned staged exploratory atlas.** Its results
are the finite factorization data, a constructive basis theorem for
supported balanced slices, and additional geometry-sensitive additive
coordinates. General freeness, a complete separating coordinate theory,
classification of all faces, and unrestricted character completeness remain
open. Degree-four types on 10--17 vertices and substantial higher-radius
catalogues are outside this checkpoint.
