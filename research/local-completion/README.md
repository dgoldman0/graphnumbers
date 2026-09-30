# Local completion of Cartesian graph arithmetic

Research checkpoint, 30 September 2026. Candidate v0.1.

The research goal is a complete extension of graph arithmetic containing the
ordinary reals, with meaningful graph approximation, continuous graph quantities,
and useful calculus. This directory specifies one candidate and records its
proofs and limitations. Its adoption as the final Graph Reals construction
remains open.

The [literature review](LITERATURE_REVIEW.md) identifies direct prior art for
the finite graph ring and substantial prior art for graph-number completions,
local convergence, moment control, and calculus. The exact signed weighted
local completion was not located in the sources examined; its originality is
unestablished. Read this candidate as an application of existing frameworks
with a particular topology, subject to the comparisons and open questions in
that review.

## What the candidate contains

Every finite simple undirected graph is included up to isomorphism. Addition of
ordinary graphs is disjoint union and multiplication is Cartesian product.
The empty graph is zero; the one-vertex graph K1 is the unit. An edgeless graph
on n vertices represents the scalar n. Other real numbers appear as real
multiples of K1, with their usual arithmetic and topology.

Finite real linear combinations of connected graph classes form the initial
algebra. Its topology compares signed rooted-neighborhood histograms at every
radius, weighted by every positive integer power of neighborhood size. Completing
this algebra adds limits such as normalized cycles C_n/n and their Cartesian
products. The rational-coefficient graph span is also dense.

The completion is a faithful commutative unital Frechet algebra. Vertex and edge
counts, isolated-vertex count, and every fixed connected motif count extend
continuously. Exponentials, polynomial differentiation, and real-parameter
differentiation and integration are established in the note. The analysis
supplement explains the resulting connections to ordinary real analysis.

## Files

| File | Purpose |
| --- | --- |
| [local_graph_completion.pdf](local_graph_completion.pdf) | Seven-page definition-and-proof note, including explicit counterexamples. |
| [local_graph_completion.tex](local_graph_completion.tex) | Editable source for the note. |
| [ANALYSIS.md](ANALYSIS.md) | Finite-graph and real embeddings, graph directions, and observable compatibility with calculus. |
| [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) | Prior-art assessment, 24 references, and boundaries on possible novelty. |
| [COMPARISON_LEMMAS.md](COMPARISON_LEMMAS.md) | Explicit identifications and examples distinguishing the candidate from nearby constructions. |
| [SEARCH_LOG.md](SEARCH_LOG.md) | Search scope, source access, query families, and remaining gaps. |
| [verify_local_algebra.py](verify_local_algebra.py) | Exact finite checks using only Python's standard library. |
| [verification_results.json](verification_results.json) | Recorded output: all 3,606 checks passed. |

## Reproduce the checks

From the repository root, with Python 3.10 or later:

```sh
cd research/local-completion
python3 verify_local_algebra.py --output verification_results.json
```

The verifier uses exact rational arithmetic and exact isomorphism backtracking.
Checks cover relabeling, product balls, signed histogram convolution, seminorm
inequalities, scalar behavior, observable formulas, all graph types on at most
four vertices, a cospectral pair, and cycle/grid and counterexample formulas.
These finite checks verify examples and implementation. The universal claims
have separate proofs in the note. Run Python without optimization flags so that
its assertions remain enabled.

To rebuild the PDF from this directory with a standard LaTeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error local_graph_completion.tex
pdflatex -interaction=nonstopmode -halt-on-error local_graph_completion.tex
```

The committed PDF was rendered and all seven pages were visually inspected.

## Established boundaries

- Arbitrary graph division is unavailable. Every connected finite graph with
  more than one vertex is a nonunit. Nonzero scalars and exponentials are units.
- Component count is discontinuous: C_(2n) - 2 C_n tends to zero locally while
  its linear component count remains -1.
- No single norm induces this topology.
- A convergent sequence of ordinary finite graphs is eventually constant.
  Nontrivial approximation uses normalized or signed graph combinations.
- The graph limits describe local combinatorial neighborhoods. Additional
  requirements involving global geometry or a dense-graphon metric remain open.

The historical software is independent of this verifier. General unit
classification, the structure of all completed elements, and the range of
geometrically useful graph variations remain research questions.
