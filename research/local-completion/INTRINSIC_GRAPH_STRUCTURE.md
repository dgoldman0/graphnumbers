# Intrinsic graph structure: partial rigidity results

30 September 2026. This note investigates the real topological algebra
$A=\mathcal A_{\mathrm{loc}}$ already constructed. Its operations,
topology, and finite-graph embedding are unchanged.

**The full reconstruction question remains open.** We prove three
restrictions on continuous automorphisms:

1. The neighborhood-kernel filtration is determined by the topology up
   to cofinal refinement. This does not recover individual radius labels
   or the rooted-ball coordinates.
2. A continuous injective unital algebra homomorphism that rescales each
   connected finite graph is the identity.
3. A continuous unital algebra endomorphism that fixes all but finitely
   many Cartesian-prime connected graphs is the identity.

The last two statements rule out every nontrivial diagonal rescaling,
including changes on infinitely many prime generators, and every
nontrivial substitution confined to finitely many prime generators.
They do not classify automorphisms that move infinitely many generators
to other, possibly completed, elements.

## 1. The reconstruction question and its distinctions

Let $\Gamma\subset A$ be the embedded ordinary finite graphs, including
the empty graph. Let

$$P_{\mathrm{loc}}=\{X\in A:a_r^X(B)\ge0\text{ for all }r,B\}.$$

This is the closed cone of coefficientwise positive balanced arrays.
Its mass-one slice is the unimodular probability laws with all the
required polynomial moments. It should be distinguished from

$$P_{\mathrm{fin}}=\overline{\operatorname{cone}(\Gamma)}.$$

We know $P_{\mathrm{fin}}\subseteq P_{\mathrm{loc}}$; equality has not
been established. The signed approximation theorem does not prove
approximation by positive finite graph combinations.

Any characterization of $\Gamma$ or either cone solely from the real
topological algebra must be preserved by every continuous unital
real-algebra automorphism. This is a necessary test, not itself a
definition or reconstruction theorem. Even proving invariance under
all automorphisms would leave the proposed intrinsic characterization
to be supplied.

Throughout, an automorphism has a continuous inverse. In this Fréchet
setting that also follows for a continuous bijective linear map from
the open mapping theorem. None of the arguments below assumes that an
arbitrary automorphism preserves the dense finite algebra $A_0$.

## 2. What the topology already recovers

Write

$$N_r=\ker T_r=\{X:a_r^X=0\},\qquad r\ge1.$$

These are closed ideals, $N_{r+1}\subseteq N_r$, and
$\bigcap_rN_r=0$. For every $k\ge1$,

$$\ker p_{r,k}=N_r.$$

**Filtration theorem.** The family $(N_r)$ is cofinal, in the direction
of smaller kernels, among the kernels of all continuous seminorms on
$A$. Consequently its cofinal class is intrinsic to the topology.

**Proof.** A continuous seminorm $q$ is bounded by a constant times a
finite maximum of defining seminorms. Monotonicity of $p_{r,k}$ gives
some $R,K,C$ with

$$q(X)\le C p_{R,K}(X).$$

Hence $N_R\subseteq\ker q$. Conversely every $N_r$ is itself the kernel
of a continuous seminorm. The same statement holds if only continuous
submultiplicative seminorms are used, since the defining seminorms are
submultiplicative.

In particular, for every continuous linear automorphism $\Phi$ and
every $r$, some $s$ satisfies

$$N_s\subseteq\Phi^{-1}(N_r).\tag{1}$$

Applying the argument to $\Phi^{-1}$ gives the reverse cofinal
comparison. One can equivalently define the intrinsic family of
linear subspaces containing the kernel of some continuous seminorm;
it is exactly the family containing at least one $N_r$.

This is a general consequence of this presentation of the topology,
not a graph reconstruction theorem. It says that observing a fixed
radius after applying a continuous map requires only some finite
radius beforehand. It does not prove $\Phi(N_r)=N_r$, nor recover ball
types, coefficient signs, polynomial weight exponents, or graph size.

The filtration is genuinely nontrivial at every stage. For $r\ge1$,

$$U(C_{2r+2})-U(C_{2r+3})\in N_r\setminus N_{r+1},\qquad U(G)=G/|V(G)|.$$

At radius $r$ both normalized cycles have the centered path ball on
$2r+1$ vertices; at radius $r+1$ their whole, different cycles appear.
This also reproves that every continuous seminorm has a nonzero kernel.

## 3. Sparse graph changes and weighted local convergence

The rigidity proofs use convergence in every original weighted
seminorm, not just weak convergence of bounded observables.

Set $b(D,r)=\sum_{j=0}^rD^j$. If both graphs under comparison have maximum
degree at most $D$, every $r$-ball has at most $b(D,r)$ vertices.
Changing an edge affects only roots within distance $r$ of its endpoints.
For a single added edge on a common $N$-vertex set, at most $2b(D,r)$
roots change. Thus, writing $B=b(D,r)$,

$$p_{r,k}(U(G')-U(G))\le\frac{4B^{k+1}}{N}.\tag{2}$$

For a graph $G$ on $N$ vertices and $G^+$ obtained by attaching one
new leaf, at most $B$ old root balls change and one new root is added.
Comparing the unnormalized histograms and then their normalizations gives

$$p_{r,k}(U(G^+)-U(G))\le\frac{2(B+1)B^k}{N+1}.\tag{3}$$

Indeed, the histogram difference has weighted norm at most
$(2B+1)B^k$, and the change of normalization contributes at most
$B^k/(N+1)$. Bounds (2) and (3) tend to zero for fixed $D,r,k$ as
$N\to\infty$.

Let $L=\lim_{n\to\infty}U(C_n)$ denote the infinite-line element.
We have $V(L)=1$, $E(L)=1$, and $L\ne0$. Continuity of multiplication gives

$$U(G\square C_n)=U(G)U(C_n)\longrightarrow U(G)L.\tag{4}$$

## 4. Diagonal rigidity

**Theorem.** Suppose $\Phi:A\to A$ is a continuous injective unital
real-algebra homomorphism and, for every connected nonempty finite
graph $G$, there is a real $c_G$ such that

$$\Phi(G)=c_GG.$$

Then $\Phi$ is the identity. Surjectivity is unnecessary.

**Proof.** Since $U(C_n)\to L$, continuity and the vertex character give

$$c_{C_n}=V(\Phi(U(C_n)))\longrightarrow c:=V(\Phi(L)).$$

It follows that $\Phi(L)=cL$. Injectivity and $L\ne0$ imply $c\ne0$.

Fix a connected $G$ with $m=|V(G)|>1$ and put $X=U(G)L$. Form $J_n$
by taking the disjoint union of $G\square C_n$ and $C_{mn}$ and adding
one edge between them. The two parts have the same number $mn$ of
vertices. Their maximum degrees, including the new edge, have a bound
depending on $G$ alone. Equations (2) and (4) therefore show

$$U(J_n)\longrightarrow Y=\tfrac12(X+L).\tag{5}$$

Each $J_n$ is connected, so $\Phi(U(J_n))=c_{J_n}U(J_n)$. Applying $V$
shows $c_{J_n}\to d:=V(\Phi(Y))$, and hence $\Phi(Y)=dY$. On the other
hand, multiplicativity gives

$$\Phi(Y)=\tfrac c2(c_GX+L).$$

The elements $X,L$ are linearly independent: both have vertex value one,
whereas the point-derivation identity for $E$ gives

$$E(X)=1+|E(G)|/m>1=E(L).$$

Comparison of their coefficients yields $cc_G=d=c$, so $c_G=1$.
The unit $K_1$ is fixed as well. Thus $\Phi$ is the identity on $A_0$,
and by continuity and density it is the identity on $A$.

**Consequence for prime rescalings.** The standard identification
$A_0\cong\mathbb R[X_P:P\text{ Cartesian-prime connected}]$ allows
arbitrary algebraic substitutions $P\mapsto\lambda_PP$ with
$\lambda_P\ne0$. They are diagonal on the connected-graph basis.
The theorem shows that such an automorphism extends continuously to
$A$ only when every $\lambda_P=1$. This includes infinite collections
of sign changes; it is not just a finite-support obstruction.

## 5. Changing only finitely many prime generators is impossible

**Bridge lemma.** A connected graph containing a bridge is
Cartesian-prime. In a Cartesian product of two connected nontrivial
graphs, each edge belongs to a four-cycle: choose an incident edge in
the other coordinate. An edge on a cycle is not a bridge.

**Theorem.** If a continuous unital real-algebra endomorphism
$\Phi:A\to A$ fixes all but finitely many Cartesian-prime connected
graphs, then $\Phi=\operatorname{id}$.

**Proof.** The cycles $C_n$ for $n\ge5$ are Cartesian-prime. Indeed,
a nontrivial product with all vertex degrees two would have both
factors connected and one-regular, hence would be $K_2\square K_2=C_4$.
Thus $\Phi(C_n)=C_n$ for all sufficiently large $n$, giving $\Phi(L)=L$.

Fix any connected nonempty $G$. Let $Q_n$ be $G\square C_n$ with one
new leaf attached. The leaf edge is a bridge, so $Q_n$ is prime. Its
vertex count tends to infinity, so it eventually avoids the finite
exceptional set of primes. Thus $\Phi(U(Q_n))=U(Q_n)$ eventually.
The degree bound is fixed as $n$ varies, and (3) and (4) give

$$U(Q_n)\longrightarrow U(G)L.$$

Taking limits and using $\Phi(L)=L$ yields

$$\Phi(U(G))L=U(G)L.$$

The already proved domain theorem and $L\ne0$ allow cancellation.
Hence $\Phi(G)=G$ for every connected $G$, and density finishes the proof.

The theorem does not require injectivity or surjectivity. It rules out
nontrivial permutations of finitely many prime generators, finite
polynomial shears, and finite translations, regardless of whether a
particular substitution passes a preliminary spectral test.

### An explicit discontinuous sign change

On $A_0$, send $K_2\mapsto-K_2$ and fix every other prime generator;
call this algebraic automorphism $\theta$. Put $H=K_2/2$, let $Q_n$ be
$K_2\square C_n$ with a leaf attached, and take $n\ge5$. Then

$$Z_n=U(Q_n)-H U(C_n)\longrightarrow0,$$

but

$$\theta(Z_n)=U(Q_n)+H U(C_n)\longrightarrow2HL\ne0.$$

In fact $V(\theta(Z_n))=2$ for every $n$. At radius one the exact
weighted error before substitution is

$$p_{1,k}(Z_n)=\frac{2^k+5^k+2\cdot4^k}{2n+1}.\tag{6}$$

The added leaf has degree one, its neighbor changes from degree three
to four, and all other vertices retain degree three; every link is
edgeless. This obstruction also shows why a symmetry of the normalized
edge's disk subalgebra need not extend to the full graph completion.

### A stronger algebraic obstruction to translation

Sending $K_2\mapsto K_2+1$ and fixing the other prime generators gives
an algebraic automorphism of $A_0$. Yet $1-K_2/3$ is a unit of $A$ by
the normalized-edge unit criterion, while its proposed image
$2/3-K_2/3$ has vertex value zero and is not a unit. Consequently no
unital algebra homomorphism $A\to A$, even a discontinuous one, can
restrict to this substitution. This example and the sign change use
different obstructions.

## 6. Two tempting positivity arguments fail

The graph-positive cone is not the cone of sums of squares, nor its
topological closure. The real degree character

$$\chi_{-1}(X)=\sum_Ba_1^X(B)(-1)^{\deg(o_B)}$$

is continuous and multiplicative. It is nonnegative on every sum of
real squares and on their closure, whereas $\chi_{-1}(K_2)=-2$.
Thus a positive finite graph can lie outside that closure. Conversely,
$(1-K_2)^2=1-2K_2+C_4$ has negative $K_2$-ball coefficients and is
outside $P_{\mathrm{loc}}$, even though it is a square. These cones
are incomparable. Both counterexamples also apply to $P_{\mathrm{fin}}$.

Likewise, multiplying each rooted coefficient by $t^{\deg(o)}$ is
multiplicative in the ambient local convolution algebras, but generally
fails mass-transport balance. On $P_3$, transport from a leaf to the
degree-two center has weighted outgoing total $2t$ and incoming total
$2t^2$. Equality forces $t\in\{0,1\}$; $t=0$ kills $K_2$ and gives no
automorphism. This rules out this particular way to extend degree-disk
symmetries. No assertion about arbitrary extensions follows from it.

## 7. What is still needed

We have not proved that every continuous automorphism preserves
$\Gamma$, $P_{\mathrm{loc}}$, or $P_{\mathrm{fin}}$. We have not
constructed an automorphism moving any of them either. Invariance of
individual $N_r$ and intrinsic recovery of $I$, $V$, $H$, or all rooted
coefficient maps are also unsettled by this note.

Any remaining nonidentity automorphism must move infinitely many
Cartesian-prime generators, and its action cannot be diagonal on the
connected-graph basis. It need not preserve $A_0$. Its action must obey
the cofinal locality constraint (1), preserve units and spectra, and
respect the sparse-perturbation limits used above.

A useful next target is an intrinsic description of a distinguished
character or cone that holds for such general maps. A successful
reconstruction proof must characterize the relevant structures without
assuming the graph coordinates it is trying to recover. The present
results establish restrictions, not full intrinsic graph reconstruction.

The subsequent [reflection investigation](REFLECTION_EXTENSION.md)
constructs a continuous surjective endomorphism sending $H$ to $-H$,
with a nontrivial kernel. It proves further necessary conditions for an
injective extension and excludes a coordinated infinite algebraic
involution by an explicit continuity obstruction. The automorphism
question remains unresolved.

## 8. Dependencies, source scope, and verification

The proofs use the established domain theorem, the line limit,
continuity of $V$ and $E$, the degree characters, and the normalized-edge
unit criterion. The cofinal-kernel argument is standard locally convex
reasoning; the graph arguments and their estimates are supplied above.

For the classical finite graph algebra and Cartesian factorization,
see Imrich–Klep–Smertnig,
[*Monoid algebras and graph products*](https://arxiv.org/abs/2407.02615)
and its [author manuscript](https://math.smertnig.at/paper/graphproduct.pdf).
The discussion of Cartesian factorization credits Sabidussi and Vizing.
A focused search for automorphisms of this local graph completion did
not supply a classification theorem used here. This is not a new
exhaustive literature review or a priority claim.

The accompanying [verifier](verify_intrinsic_structure.py) checks the
weighted sparse-change estimates against directly constructed graphs,
the exact sign-change witness, bridge and square fixtures, mixture
observables, strict filtration examples, and the two positivity
obstructions. All [256 recorded checks](intrinsic_structure_results.json)
passed. These finite checks support the examples; the universal
rigidity assertions depend on the proofs above.
