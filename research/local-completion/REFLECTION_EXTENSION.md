# The normalized-edge reflection: a surjective endomorphism and the lifting problem

30 September 2026. Continuation of the
[intrinsic-structure investigation](INTRINSIC_GRAPH_STRUCTURE.md).
Write $A=\mathcal A_{\mathrm{loc}}$ and $H=K_2/2$.

**The automorphism question remains unresolved.** We construct an explicit
continuous surjective unital real-algebra endomorphism $F:A\to A$ with
$F(H)=-H$, together with continuous linear positive maps $R,O:A\to A$
satisfying $FR=\operatorname{id}$ and $FO=-\operatorname{id}$.
The endomorphism is not injective: $F(P_3+K_1)=0$.

The reflection is an automorphism on a proper closed subalgebra generated
by graphs whose connected components have uniform degree parity.
We also prove a necessary signed-image condition for any injective
extension, and an exact block criterion for a general automorphism
extension. That criterion does not assume preservation of the kernel
of the degree retraction.

## 1. A graph operation realizing the reflection

For a finite simple graph $G$, put

$$\epsilon_G(v)=(-1)^{\deg_G(v)}.$$

Let $QG$ have the same vertices as $G$, keeping precisely the edges $uv$
for which $\epsilon_G(u)=\epsilon_G(v)$. The parity is computed in the
original graph. It is constant on each connected component $C$ of $QG$;
write this common sign as $\epsilon_G(C)$. Define

$$F(G)=\sum_{C\in\operatorname{Comp}(QG)}\epsilon_G(C)\,C.\tag{1}$$

This is a finite integral graph combination. Disjoint union is respected,
so (1) extends real-linearly to $A_0$. It fixes $K_1$, and

$$F(K_2)=-K_2,\qquad F(P_3)=-K_1.\tag{2}$$

For the second identity, the two odd-degree leaves and the even-degree
center become three isolated vertices with signs $-1,-1,+1$.

### Multiplicativity

Root degrees add in a Cartesian product:

$$\epsilon_{G\square J}(u,v)=\epsilon_G(u)\epsilon_J(v).$$

For an edge changing the $G$ coordinate, equality of endpoint parities
is exactly equality in $G$; the $J$ contribution cancels. The same holds
in the other coordinate. Consequently

$$Q(G\square J)=QG\square QJ.$$

Its connected components are products of components of $QG$ and $QJ$,
with the products of their signs. Hence

$$F(G\square J)=F(G)F(J).\tag{3}$$

### Continuity in the actual completion

The rooted $r$-ball of $QG$ at $v$, together with $\epsilon_G(v)$, is
determined by $B_{r+1}(G,v)$. That ball reveals the original degrees of
all vertices that can participate in the output $r$-ball. Edge deletion
can only decrease its size. The output histogram is therefore a signed
pushforward of the input $(r+1)$-histogram, with multipliers of modulus one.
For every $X\in A_0$,

$$p_{r,k}(F(X))\le p_{r+1,k}(X).\tag{4}$$

This estimate includes cancellations in arbitrary signed inputs. It
extends $F$ uniquely to a continuous unital algebra endomorphism of $A$.
Its outputs satisfy mass-transport balance because they are limits of
finite graph combinations. Equivalently, on $QG$ the sign is constant
along every retained edge, so it does not unbalance a component's
uniform-root counting measure.

## 2. Continuous positive preimages of both signs

### A right inverse for $F$

Let $o(G)$ be the number of odd-degree vertices. Form $E(G)$ by attaching
one new leaf to each such vertex. Every original vertex now has even
degree, and each new leaf has odd degree. The parity filter deletes all
the newly attached edges and retains all original edges. Thus

$$F(E(G))=G-o(G)K_1.$$

Define the ordinary finite graph

$$R(G)=E(G)\sqcup o(G)K_1.\tag{5}$$

Then $F(R(G))=G$. The construction respects disjoint union and therefore
defines a real-linear map on $A_0$. It is not asserted to preserve products.

For $r\ge1$, all output contributions associated with an original root
$v$ are determined by $B_r(G,v)$: the ball rooted at $v$, the ball rooted
at its new leaf if present, and the extra isolated vertex if present.
The first two balls each have size at most $2|B_r(G,v)|$. Hence

$$p_{r,k}(R(X))\le(2^{k+1}+1)p_{r,k}(X),\qquad r\ge1.\tag{6}$$

These seminorms also dominate radius zero, so $R$ extends continuously
to $A$. The identity $FR=\operatorname{id}$ extends by density.
In particular, **$F$ is surjective on the completed algebra**.

### Positive preimages of negatives

Form $O(G)$ by attaching a path of two edges to each even-degree original
vertex, using two new vertices for each path. All original vertices
then have odd degree. Each new middle vertex has degree two and each
new terminal vertex has degree one. The filter retains the original
graph with negative sign; every new path is cut into isolated vertices
whose $+1$ and $-1$ contributions cancel. Therefore

$$F(O(G))=-G.\tag{7}$$

This construction also respects disjoint union. Its rooted output
contributions are determined by the original $r$-ball for $r\ge1$.
There are at most three roots associated with each original root, and
their output balls have size at most $3|B_r(G,v)|$. Thus

$$p_{r,k}(O(X))\le3^{k+1}p_{r,k}(X),\qquad r\ge1.\tag{8}$$

Consequently $O$ extends continuously and $FO=-\operatorname{id}$ on $A$.
Both $R$ and $O$ are positive for the coefficientwise cone
$P_{\mathrm{loc}}$: their local formulas are pushforwards with
nonnegative integer multiplicities. They also preserve $P_{\mathrm{fin}}$.
Neither map is claimed to be an algebra homomorphism.

Every element $\sum_i c_iG_i$ of $A_0$ is the image under $F$ of

$$\sum_{c_i\ge0}c_iR(G_i)+\sum_{c_i<0}|c_i|O(G_i),$$

a positive finite graph combination. No conclusion that arbitrary
completed elements are differences of positive elements is needed or
asserted. Such a conclusion would conflict with the established
elements beyond the finite signed-measure model.

### The kernel is essential

The nonzero ordinary graph $P_3\sqcup K_1$ lies in $\ker F$. Thus this
surjective map is not an automorphism. Its existence proves that the
equation $\Phi(H)=-H$ is compatible with continuity, multiplication,
and surjectivity simultaneously. Injectivity is the remaining property
for this construction, and cannot be inferred from the others.

The linear section gives a topological vector-space decomposition

$$A\cong\ker F\oplus A,
\qquad X\longmapsto(X-RF(X),F(X)).$$

The inverse is $(k,y)\mapsto k+R(y)$. This is a statement about topological
vector spaces; no direct-product decomposition of algebras is claimed.

## 3. An actual reflection on a proper graph subalgebra

Let $A_{\mathrm{par}}$ be the closure of the real span of finite graphs
whose connected components each have uniform degree parity. Different
components may have different parities. This class is closed under
disjoint union and Cartesian product.

For a connected graph in this class, no edge is deleted and
$F(G)=(-1)^{\deg_G(v)}G$, with a sign independent of $v$. Thus $F^2$ is
the identity on the generating span. At every positive radius its
histogram action is simply

$$a_r(B)\longmapsto(-1)^{\deg(o_B)}a_r(B),$$

so every $p_{r,k}$ with $r\ge1$ is preserved. It follows that

$$F|_{A_{\mathrm{par}}}:A_{\mathrm{par}}\to A_{\mathrm{par}}$$

is a continuous involutive algebra automorphism. It contains $H$, all
regular finite graphs, their normalized limits, and some irregular
graphs such as the three-leaf star, whose degrees are all odd.

The subalgebra is proper. Its radius-two coefficients vanish on any
ball witnessing an edge between vertices of opposite degree parity;
the relevant endpoint degrees are visible at that radius. The endpoint
rooted whole $P_3$ ball is one such coordinate, and has coefficient two
on $P_3$. Hence $P_3\notin A_{\mathrm{par}}$.

This does not contradict diagonal rigidity for the full algebra: the
domain of the reflection in this paragraph is a proper closed subalgebra.

## 4. Every injective extension must produce mixed signs

There is a universal necessary condition stronger than excluding
diagonal formulas. Let $L=\lim U(C_n)$ and form $J_n$ by joining
$K_2\square C_n$ to $C_{2n}$ with one bridge. Both parts have $2n$
vertices. The sparse-change estimates give

$$U(J_n)\longrightarrow Y=\tfrac12(1+H)L.\tag{9}$$

**Theorem.** If $\Phi:A\to A$ is a continuous injective unital algebra
homomorphism with $\Phi(H)=-H$, then for all sufficiently large $n$,
$\Phi(J_n)$ has both positive and negative coefficients at some fixed
finite radius. In particular, it cannot send every connected finite
graph to a nonzero scalar multiple of an ordinary graph, or even to
an element of $P_{\mathrm{loc}}\cup(-P_{\mathrm{loc}})$.

**Proof.** Continuity and multiplication give

$$\Phi(U(J_n))\longrightarrow W=\tfrac12(1-H)\Phi(L).$$

Injectivity ensures $\Phi(L)\ne0$, and the domain theorem ensures
$W\ne0$. Yet $V(W)=0$, since $V(H)=1$. Choose a radius $r$ at which
$a_r^W\ne0$. Its absolutely summable coefficients sum to zero, so
at least one is positive and another negative. Convergence of those
two coordinates forces the same signs in $\Phi(U(J_n))$ for every
sufficiently large $n$. Positive normalization does not change signs.

For the explicit noninjective map $F$, the behavior is visible directly.
Let $v$ be the bridge vertex in the prism $K_2\square C_n$. Its original
degree becomes four, and the bridge vertex in the cycle has degree
three. Filtering isolates both, with cancelling signs, and leaves

$$F(J_n)=P_{2n-1}-(K_2\square C_n-v),\qquad n\ge5.\tag{10}$$

This has vertex value zero and mixed signs. Its normalized radius-one
coefficients on stars with one, two, and three leaves are respectively

$$\frac1{2n},\qquad \frac12-\frac3{2n},\qquad -\frac12+\frac1n.$$

They converge to the radius-one marginal of $(1-H)L/2$.

## 5. Exact criterion for a general automorphism extension

Identify $B=\overline{\mathbb R[H]}$ with the real smooth disk algebra.
Let $P=SD:A\to B$ be the degree retraction, let $J=\ker P$, and write

$$A=B\oplus J$$

as topological vector spaces. The subspace $J$ is a closed ideal.
Let $\sigma:B\to B$ be $f(H)\mapsto f(-H)$.

Any continuous homomorphism taking $H$ to $-H$ restricts to $\sigma$ on
$B$, by density of polynomials. It must therefore have the form

$$\Phi(b+j)=\sigma(b)+\alpha(j)+T(j),\tag{11}$$

where $\alpha:J\to B$ and $T:J\to J$ are continuous real-linear maps.

**Lifting criterion.** Formula (11) is a continuous unital algebra
automorphism exactly when $T$ is bijective and the following identities
hold for every $b\in B$ and $j,k\in J$:

$$\begin{aligned}
\alpha(bj)&=\sigma(b)\alpha(j),&
T(bj)&=\sigma(b)T(j),\\
\alpha(jk)&=\alpha(j)\alpha(k),&
T(jk)&=\alpha(j)T(k)+\alpha(k)T(j)+T(j)T(k).
\end{aligned}\tag{12}$$

**Proof.** Compare the $B$ and $J$ components of the products of $b$
with $j$, and of $j$ with $k$. These comparisons give all multiplicativity
conditions, since multiplication on $B$ is already handled by $\sigma$.
The block form is bijective exactly when $T$ is bijective, with inverse

$$b+j\longmapsto
\sigma^{-1}\bigl(b-\alpha(T^{-1}j)\bigr)+T^{-1}j.$$

Both $J$ and $B$ are Fréchet spaces, so a continuous bijection $T$ has
continuous inverse. This proves necessity and sufficiency.

This criterion allows $\alpha\ne0$, hence does not assume $\Phi(J)=J$.
In all cases a necessary linear condition is

$$T(Hj)=-H T(j).\tag{13}$$

Thus multiplication by $H$ on $J$ must be continuously similar to its
negative. Equivalently, the $B$-module $A/B$ must admit the corresponding
semilinear symmetry. Conditions (12) impose further product constraints;
solving the linear similarity problem alone would not suffice.

### The natural linear involution fails multiplication

The map

$$U=(\operatorname{id}-P)+\sigma P$$

is a continuous real-linear involution of $A$ and sends $H$ to $-H$.
It corresponds to $\alpha=0$, $T=\operatorname{id}_J$, which violates
(13) whenever $j\ne0$.

For an explicit witness take

$$j=P_3-K_2-\tfrac14C_4=P_3-2H-H^2.$$

Its degree generating function is zero, so $j\in J$, while its
radius-two histogram is nonzero. Then

$$U(Hj)-U(H)U(j)=2Hj\ne0.$$

The reflected retraction $\sigma P$ is instead a continuous algebra
endomorphism with kernel $J$. These two formulas supply different
partial lifts; neither gives an automorphism of $A$.

### The degree kernel cannot simply be assumed invariant

There are already distinct continuous retractions onto $B$. Let
$\tau(B)$ count triangles through the root of a one-ball and define

$$D_{\triangle}(X)(z)=
\sum_{B\in\mathcal B_1}a_1^X(B)(-1)^{\tau(B)}z^{\deg(o_B)}.$$

Both root degree and root triangle count add under Cartesian products.
The coefficient bounds are the same as for $D$, and hypercubes have no
triangles. Hence $D_{\triangle}$ is a continuous unital homomorphism
with $D_{\triangle}S=\operatorname{id}$. It differs from $D$:

$$D(K_3)=3z^2,\qquad D_{\triangle}(K_3)=-3z^2.$$

For example, $K_3-3H^2$ lies in $\ker D$ but not in
$\ker D_{\triangle}$. Distinct retractions do not prove the existence
of an automorphism exchanging their kernels. They do show that uniqueness
of the retraction cannot be used without an additional characterization.

### A natural algebraic involution fails continuity

The preceding linear involution suggests a stronger construction on
$A_0$. Using its polynomial presentation, change coordinates from each
Cartesian-prime generator $P\ne K_2$ to

$$Z_P=P-S(D(P)).$$

Together with $H$, these are again free polynomial generators: each
change only subtracts a polynomial in $H$. Define the algebraic
involution $\theta$ by $\theta(H)=-H$ and $\theta(Z_P)=Z_P$.
For a prime generator its explicit formula is

$$\theta(P)=P+S\bigl(D(P)(-z)-D(P)(z)\bigr).\tag{14}$$

This is bijective, multiplicative, and satisfies
$D\theta=\sigma D$ on all of $A_0$. Its action changes infinitely many
prime generators. Nevertheless, it is discontinuous in our topology.

Let $Q_n$ be $K_2\square C_n$ with one added leaf, for $n\ge6$, and set

$$X_n=U(Q_n)-H U(C_n)\longrightarrow0.$$

The bridge makes $Q_n$ prime. It has one degree-one vertex, one
degree-four vertex, and $2n-1$ degree-three vertices. The cycles are
prime and have only degree-two vertices. Consequently

$$\theta(X_n)=U(Q_n)+H U(C_n)
-\frac{2(2n-1)}{2n+1}H^3-\frac{2}{2n+1}H
\longrightarrow 2H(L-H^2)\ne0.\tag{15}$$

At radius two, the balls of $HL$ and $H^3$ have eight and seven vertices
respectively. Thus the limiting seminorm is
$2(8^k+7^k)>0$. More directly, the seven-vertex radius-two hypercube-ball
coefficient in (15) is exactly $-2(2n-1)/(2n+1)$: old roots in $Q_n$
have balls of size at least eight, and its leaf has degree one.

This rules out a coordinated infinite polynomial substitution, beyond
the finite-prime and diagonal cases treated previously. It does not
rule out every solution of the lifting criterion.

## 6. Current conclusion and verification

The reflection extends to a continuous surjective algebra endomorphism
of the whole completion, and to an automorphism of $A_{\mathrm{par}}$.
The whole-algebra automorphism problem remains open: no pair
$(\alpha,T)$ satisfying (12) with $T$ bijective has been constructed,
and no general impossibility proof has been obtained.

Surjective endomorphisms do not supply a counterexample to invariance
under automorphisms. The finite graph embedding and positive cones
therefore have the same unresolved intrinsic status as before.

The [exact verifier](verify_reflection_extension.py) checks the parity
filter directly, its Cartesian multiplicativity, both positive preimage
constructions, weighted estimates on signed inputs, the explicit mixed-sign
family, the failed linear involution, the discontinuous algebraic
involution, and the triangle-weighted retraction.
Its finite output is [recorded separately](reflection_extension_results.json).
Universal continuity, surjectivity, and the lifting criterion follow from
the proofs here, not from a finite search for automorphisms.

The arguments use the existing domain theorem, the degree retraction,
and the sparse-change estimates. The block calculation is elementary
split-algebra bookkeeping. Focused searches supplied no external
classification theorem used to decide this extension problem. No priority
claim is made for these graph operations or deductions.
