# Nonspectral point derivations and obstructions to local coefficient flows

30 September 2026. Write $A=\mathcal A_{\mathrm{loc}}$ for the real
local Cartesian graph completion and $V$ for its vertex-mass character. This note keeps
the distinction between scalar observables and maps taking values in
the graph algebra itself. The new nonspectral transforms use graph
coordinates already supplied by the construction. Their existence does
not intrinsically recover those coordinates from the bare topological
algebra.

We prove three statements:

1. Neighbor-link component counts supply infinitely many linearly
   independent continuous point derivations at $V$.
2. Every continuous derivation $A\to A$ that is diagonal on connected
   finite graphs is zero.
3. A finite-radius multiplicative root weight preserves mass-transport
   balance under coefficient reweighting only when it is the identity
   weight or the projection onto isolated vertices. A finite-radius
   additive root weight preserves balance this way only when it is zero.

These are scalar-calculus and rigidity results. They do not classify
arbitrary derivations or automorphisms of $A$.

## 1. Infinitely many nonspectral tangent directions

For a rooted graph $(G,o)$, let $\operatorname{lk}(G,o)$ be the induced
graph on the neighbors of $o$. For each nonempty connected finite graph
$F$, let $c_F(G,o)$ count the connected components of this link that are
isomorphic to $F$. The Cartesian product satisfies

$$
\operatorname{lk}(G\square H,(o,p))
=\operatorname{lk}(G,o)\sqcup\operatorname{lk}(H,p),
$$

so every $c_F$ is Cartesian-additive. Moreover,

$$0\le c_F(G,o)\le\deg_G(o)/|V(F)|\le |B_1(G,o)|.$$

Consequently

$$
\delta_F(X)=\sum_{B\in\mathcal B_1}a_1^X(B)c_F(B)
$$

is a continuous real-linear functional, with
$|\delta_F(X)|\le p_{1,1}(X)$. Local convolution gives

$$
\delta_F(XY)=\delta_F(X)V(Y)+V(X)\delta_F(Y).\tag{1}
$$

Thus $\delta_F$ is a continuous point derivation at $V$. The statement
concerns scalar-valued maps, not an $A$-valued derivation.

**Independence theorem.** The family $(\delta_F)$, indexed by connected
finite nonempty graphs, is linearly independent.

**Proof.** Suppose a finite linear combination
$\sum_F\alpha_F\delta_F$ is zero. Induct on $m=|V(F)|$. Test the
identity on the cone $\operatorname{cone}(F)$ obtained by adjoining a
new vertex adjacent to every vertex of $F$. Every root link has at most
$m$ vertices. A component of a root link has exactly $m$ vertices only
when that root is universal in the cone. Its link is then isomorphic
to $F$: the new apex has link $F$, and removing any other universal
vertex simply replaces that universal vertex of $F$ by the apex.
There are $1+u(F)$ such vertices, where $u(F)$ counts the universal
vertices of $F$. Every other link component has fewer than $m$ vertices.
After the coefficients at smaller sizes have vanished, evaluation gives

$$0=(1+u(F))\alpha_F,$$

so $\alpha_F=0$. The argument starts at $F=K_1$ and exhausts the finite
combination. $\square$

Put $\mathfrak m_V=\ker V$, and let
$\overline{\mathfrak m_V^2}$ denote the closure of the linear span of
products of pairs of elements of $\mathfrak m_V$. Equation (1) means
that every $\delta_F$ vanishes on this closed subspace. Their restrictions
to $\mathfrak m_V$ remain independent, since
$\delta_F(X)=\delta_F(X-V(X)1)$. Therefore the topological cotangent
space

$$\mathfrak m_V/\overline{\mathfrak m_V^2}$$

is infinite-dimensional. This is a statement at the specified character
$V$; it does not characterize $V$ intrinsically.

## 2. Continuous diagonal derivations vanish

**Theorem.** Let $D:A\to A$ be a continuous real-linear derivation:
$D(XY)=D(X)Y+XD(Y)$. Suppose that for every connected nonempty finite
graph $G$ there is a scalar $d_G$ with

$$D(G)=d_GG.$$

Then $D=0$.

**Proof.** Write $U(G)=G/|V(G)|$ and
$L=\lim_{n\to\infty}U(C_n)$. Applying $V$ to
$D(U(C_n))=d_{C_n}U(C_n)$ shows
$d_{C_n}\to c=V(D(L))$. Hence $D(L)=cL$.

Fix a connected graph $G$ with $m=|V(G)|>1$, and set $X=U(G)L$.
The derivation identity gives

$$D(X)=(d_G+c)X.$$

Join $G\square C_n$ to $C_{mn}$ by one edge and call the connected
graph $J_n$. The two pieces have equal vertex count, and the degree
bound is independent of $n$. The weighted sparse-change estimate from
[the intrinsic-structure note](INTRINSIC_GRAPH_STRUCTURE.md) gives

$$U(J_n)\longrightarrow Y=\tfrac12(X+L).$$

Applying $V$ again shows $d_{J_n}\to d=V(D(Y))$, and thus $D(Y)=dY$.
Linearity and the formula for $D(X)$ give

$$d(X+L)=(d_G+c)X+cL.$$

The elements $X$ and $L$ are linearly independent. They have the same
vertex mass, while their edge observables are
$E(X)=1+|E(G)|/m>1=E(L)$. Comparing coefficients gives $d=c$ and
$d_G=0$. A derivation vanishes on the unit as well. It therefore
vanishes on the dense finite graph algebra and, by continuity, on $A$.
$\square$

This is an infinitesimal companion to diagonal automorphism rigidity.
It excludes a continuous $A$-valued derivation that simply assigns an
additive scalar weight to each connected graph. It does not exclude
derivations whose images mix graph elements, or assert that a scalar
point derivation must vanish.

## 3. Balance forces a root weight to be constant on a finite component

Fix a radius $r\ge1$. Let $w$ be a real or complex function of rooted
$r$-ball type. For a finite graph $G$, reweight the counting mass at
each root $u$ by $w(G,u)$. At radii $R\ge r$ this prescribes

$$b_R(B)=w(B|_r)a_R^G(B),\tag{2}$$

and at smaller radii use truncation. This produces compatible arrays;
mass-transport balance is the additional condition in question.

**Balance lemma.** If (2) is balanced for a connected finite $G$, then
$w(G,u)$ is independent of $u$.

**Proof.** Test balance with the adjacent-vertex transport

$$F(G,u,v)=\mathbf1_{\{u\sim v\}}\overline{w(G,u)-w(G,v)}.$$

For this test one may restrict to components isomorphic to this fixed
finite $G$, using a ball of radius exceeding its diameter. The transport
is then local and bounded. Complex transports mean their real and
imaginary parts. The weighted outgoing-minus-incoming sum is

$$
\sum_{u\sim v}^{\mathrm{oriented}}
(w(G,u)-w(G,v))\overline{w(G,u)-w(G,v)}
=2\sum_{\{u,v\}\in E(G)}|w(G,u)-w(G,v)|^2.
$$

Balance makes this zero. Every adjacent pair has equal weights, and
connectedness finishes the proof. $\square$

The lemma uses all the balance equations of the completion. Inspecting
only the radius-$r$ histogram would miss this obstruction.

## 4. Classification of finite-radius multiplicative weights

Assume $w$ is a semicharacter of the radius-$r$ rooted product monoid:

$$w(B\star_r C)=w(B)w(C),\qquad w(K_1)=1.$$

No growth assumption is needed for the following finite-graph result.

**Theorem.** Suppose coefficient reweighting by $w$ is balanced for
every finite graph. Exactly two choices are possible:

$$
w(B)=1\quad\text{for every }B,
\qquad\text{or}\qquad
w(B)=\mathbf1_{\{B=K_1\}}.\tag{3}
$$

They induce, respectively, the identity map and the isolated-vertex
projection $X\mapsto I(X)K_1$ on $A$.

**Proof.** Let $c$ be the weight of the common $r$-ball in a cycle of
length greater than $2r+1$. Fix a connected finite rooted graph $(G,o)$
with at least two vertices. For any integer $n>r$, the vertex
$o^n$ in $G^{\square n}$ lies at distance at least $n$ from a suitable
vertex: choose a neighbor of $o$ in each coordinate. Attach a long
cycle to that distant vertex by one new edge. The $r$-ball at $o^n$
is unchanged. A cycle root chosen more than $r$ steps from the
attachment also retains its line $r$-ball.

The balance lemma makes the two weights equal in this connected
graph. Multiplicativity therefore gives

$$w(G,o)^n=c\qquad(n>r).\tag{4}$$

If $c=0$, equation (4) forces $w(G,o)=0$ for every such graph. If
$c\ne0$, comparison of consecutive exponents gives $w(G,o)=1$, and
then $c=1$. Every rooted ball is realized by its own finite graph,
so (3) follows. Both listed choices plainly preserve balance and have
the stated continuous extensions. $\square$

In particular, a neighbor-link generating weight

$$w(B)=\prod_F z_F^{c_F(B)}$$

can define many useful scalar characters. Used as a coefficient
reweighting map on elements of $A$, it preserves balance only when all
$z_F=1$ or all $z_F=0$. To see that these parameters are forced, evaluate
at the apex of $\operatorname{cone}(F)$, whose link has the single
component $F$. For phase parameters $|z_F|=1$, only the identity
remains. This gives no prohibition on non-diagonal automorphisms.

## 5. Additive weights and the scalar/global distinction

Suppose instead that a finite-radius real or complex root statistic $q$
satisfies

$$q(B\star_r C)=q(B)+q(C).$$

Then $q(K_1)=0$. If coefficient weighting by $q$ preserves balance for
every finite graph, the same gluing argument gives

$$nq(G,o)=c\qquad(n>r),$$

with $c$ the long-cycle value. Consecutive exponents imply $q(G,o)=0$,
and hence $q=0$ on all ball types.

Thus a nonzero additive root statistic can be used in continuous scalar
point derivations when it has polynomial growth, but its direct
coefficient-weighting rule fails to give an $A$-valued derivation. In
the ambient unbalanced convolution arrays the Leibniz rule does hold;
the rule fails to land in the graph completion.

The obstruction has small explicit witnesses for every $c_F$. For
$F=K_1$, use $P_3$: the root weights are one at the leaves and two at
the center. For $|V(F)|\ge2$, attach one leaf at the apex of
$\operatorname{cone}(F)$. The apex has $c_F=1$ and the new leaf has
$c_F=0$, so the balance energy in Section 3 is positive.

## 6. Scope for the intrinsic question

The infinite-dimensional cotangent space is an algebraic and topological
statement at $V$. The particular basis of point derivations was
constructed from external rooted graph coordinates. We have not proved
that $V$, the radius-one quotient, its link-component generators, the
finite graphs, or the positive cone are intrinsically determined by
$A$.

The two rigidity statements show why scalar generating-function
symmetries cannot be promoted to symmetries of $A$ by simply
reweighting coefficients. They leave open transformations that rearrange
rooted types, mix graph elements, or change the graph while transporting
its weights. The parity-filter endomorphism in
[the reflection note](REFLECTION_EXTENSION.md) is compatible with this
restriction: it deletes edges before assigning component signs.

All claims above have proofs using the previously established local
convolution, balance representation, sparse-change estimates, and line
limit. These results add restrictions and tangent structure to the
intrinsic investigation; they do not settle the automorphism problem
or establish a literature-priority claim.
