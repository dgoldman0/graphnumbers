# The span of the local positive cone

1 October 2026. Proof checkpoint for M2 of the
[development plan](../../reviews/local-completion-2026-09-30/DEVELOPMENT_PATH.md).
Independent referee review of this new note is pending.

Work over the real local Cartesian graph completion A. Its cone P consists
of elements with nonnegative coefficients in every local marginal.
The representation theorem identifies a positive element with a finite
unimodular measure on connected locally finite rooted simple graphs whose
polynomial ball-size moments are all finite. Its total mass is V(x), and
||T_r x||1=V(x) at every radius.

For an element having a finite signed representing measure mu, distinguish
the two quantities

\[
\sum_B |a_r(B)|\,|V(B)|^k,
\qquad
\int |B_r(G,o)|^k\,d|\mu|(G,o).                              \tag{1}
\]

The first is a defining seminorm of A. Pushforward can cancel signs, so
its finiteness does not supply finiteness of the second.

## 1. Characterization

**Theorem.** An element x belongs to P-P if and only if it has a finite
signed representing measure mu such that

\[
\int |B_r(G,o)|^k\,d|\mu|(G,o)<\infty
\quad\hbox{for every finite r and integer k>=1}.              \tag{2}
\]

In that case the Jordan parts mu+ and mu- are unimodular and represent
positive elements x+ and x-, with x=x+-x-.

**Necessity.** If x=y-z for y,z in P, uniqueness of the signed representing
measure gives mu=mu_y-mu_z. The domination |mu|<=mu_y+mu_z proves (2).

**Sufficiency, local balance to an edge measure.** Define a signed measure
on isomorphism classes of graphs with a distinguished directed edge by

\[
\nu(A)=\int\sum_{v\sim o}\mathbf1_A(G,o,v)\,d\mu(G,o).       \tag{3}
\]

It is finite: its total variation is at most integral deg(o) d|mu|,
which is finite by (2) at radius one. Let iota(G,o,v)=(G,v,o).
For a directed-edge cylinder A, the two sums obtained from A and iota(A)
are finite-radius local functions with polynomial growth, bounded by
the root degree. The balance identities of x therefore say
nu(A)=nu(iota(A)). The passage from the local arrays to these integrals
is legitimate because (2) gives absolute integrability. Directed-edge
cylinders generate the Borel sigma-algebra and form a determining class
for finite signed measures. Uniqueness extends the equality to

\[
\iota_*\nu=\nu.                                             \tag{4}
\]

**Jordan parts.** Choose disjoint Borel Hahn supports A+ and A- for mu+
and mu-. The edge lifts of mu+ and mu- in (3) are positive and supported
on the disjoint inverse images of A+ and A- under the map forgetting
the second root. They are therefore exactly the Jordan parts of nu.
An invertible measurable map preserves the Jordan decomposition, so
(4) and its uniqueness give flip invariance of both lifted measures.
Thus mu+ and mu- are involution invariant. By Aldous–Lyons Proposition
2.2, after normalizing any nonzero part, each is unimodular. A zero part
needs no normalization, and isolated-root mass is harmless.

Finally mu+ and mu- are dominated by |mu|, so (2) gives their required
ball-size moments. The representation theorem supplies x+,x- in P;
uniqueness of local marginals gives x=x+-x-. QED.

The use of the finite directed-edge measure is essential to this proof.
Neither mere finite total variation nor an informal Jordan decomposition
of the separate local arrays substitutes for (2).

## 2. Order consequences

**The cone does not generate A.** Every difference y-z of positive
elements satisfies ||T_r(y-z)||1<=V(y)+V(z), uniformly in r. The cut-line
defect E has ||T_r E||1=4r, so E is outside P-P. The same holds for every
element with unbounded local variation.

**P-P is a vector lattice.** On this subspace the order is exactly the
order of representing measures. A finite signed measure is nonnegative
if all its local marginals are nonnegative, by uniqueness of the positive
extension from cylinders. The Jordan parts above therefore give the
positive and negative parts in the ordered vector space. In particular
|x|=x++x-, and sup(x,y)=(x+y+|x-y|)/2 belongs to P-P.

**The cone is normal in the seminorm sense.** If 0<=x<=y, coefficientwise
positivity at each radius gives

\[
p_{r,k}(x)\le p_{r,k}(y).                                    \tag{5}
\]

This is the usual monotone-seminorm criterion for a normal cone. More
explicitly, if a<=z<=b with p_(r,k)(a),p_(r,k)(b)<epsilon, coefficientwise
comparison gives p_(r,k)(z)<2epsilon; finite intersections give the
neighborhood formulation of normality. Normality does not imply generation.

**The lattice modulus is not continuous in the inherited topology.** Put
x_n=U(C_(2n))-U(C_n), n>=3. For every fixed radius its two normalized
cycle marginals eventually agree, so x_n tends to zero in A. Their global
rooted graph measures have disjoint support. Thus |x_n|=U(C_(2n))+U(C_n)
has vertex mass two for every n and does not tend to zero. The positive
span with its inherited topology is consequently not a topological vector
lattice. This also exhibits why taking absolute values after truncation
cannot recover the global Jordan parts.

## 3. Sources and verification scope

The all-radius representation and uniqueness used here are in
[REPRESENTATION_THEOREM.md](REPRESENTATION_THEOREM.md), Sections 3–4.
The positive-measure implication from edge-flip invariance to full mass
transport is Aldous and Lyons, *Processes on Unimodular Random Networks*,
Proposition 2.2 ([primary text](https://arxiv.org/html/math/0603062v6)).
Their [errata](https://www.stat.berkeley.edu/~aldous/Papers/urn-pub.pdf)
distinguish this edge-flip condition from the weaker assertion of
reversibility of the rooted random walk; the weaker assertion is not
used here.

The foundation verifier records the cycle cancellation example by direct
rooted-ball extraction, and compares local absolute mass zero with global
Jordan mass two. That fixture checks the distinction in (1), not the
general measure theorem. The local-cylinder extension and the preservation
of Jordan parts under the edge lift are the proof obligations discharged
in Section 1. The subsequent [positive-cone checkpoint](STRICT_POSITIVE_CONES.md)
supplies the labelled-network encoding and proves strict inclusion of the
closed finite positive cone in P, using the cited non-soficity theorem.
That result is separate from this note's Jordan-decomposition argument.
