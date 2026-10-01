# Code review: core library modules

- **Assigned:** `python/src/graphlocal/graphs.py`, `elements.py`, `local.py`, `heat.py`, `reconstruction.py`, `defects.py`, `prepared.py`, `interactions.py`, `__init__.py`.
- **Revision reviewed:** `d6aba96`. These modules are unchanged in `65f08bb`.
- **Evidence:** `common.py`, `oracles.py` and the `t_*.py` scripts in this folder.
- **Brief:** [BRIEFS.md](../BRIEFS.md#code-core).

This is the reviewer's final report as delivered, except that scratch-directory paths now point to this folder. It was delivered three times; the later copies differed only in using absolute repository paths. The lead reviewer reproduced findings 1 and 2 ([lead/outputs/code_findings.txt](../lead/outputs/code_findings.txt)).

---

I reviewed graphs.py, elements.py, local.py, heat.py, reconstruction.py, defects.py, prepared.py, interactions.py and __init__.py against independent oracles (networkx, mpmath, scipy, closed forms). I found **no unsound certificate on any path built only from library elements**. Heat, relative heat, prepared queries, the algebra exponential, local data and isomorphism all held. The one false certificate needs a custom source that ignores the requested weight, and `exp()` doesn't catch it. The other findings are a reconstruction contract hole, budgets that don't bound work, and test gaps. No repo files were modified (`git status` is clean). Scratch scripts are in this folder.

## Findings, most severe first

**1. VALIDATION (can produce a false certificate): `elements.py:332–347`, `Exponential.approximate`.** It never checks the `k` or radius of the inner approximation (line 337). It always labels its result with the requested `k` and adds `lipschitz*approx.error`, even when that error only holds in a weaker norm. `heat_return` and `relative_heat` do check what they receive. `Product.approximate` (308–314) raises on a k mismatch, or quietly returns the weaker `k` when both inputs are weak, which honours the label but not the request.
```python
class WeightOne(Element):   # honest k=1 certificates, ignores requested k
    degree_bound, variation_bound, mass, positive = 2, Q(1), Q(1), True
    def norm_bound(self, r, k): return Line().norm_bound(r, k)
    def approximate(self, r, k=1, epsilon="1e-6"):
        eta = Q(epsilon)
        bump = LocalHistogram(r, [(star(99), eta/100)])   # k=1 weight exactly eta
        return LocalApproximation(Line().local(r) + bump, 1, eta)
got = exp(WeightOne()/10).approximate(1, 3, "1e-3")
ref = exp(Line()/10).approximate(1, 3, Q("1e-30"))
got.k, float(got.error), float((got.histogram - ref.histogram).norm(3))
# actual: (3, 0.00064, 0.1297)  -> certificate false by ~200x
```
Expected: reject when `approx.k < k` or the radius differs, or return the honest weaker `k`.

**2. BUG/VALIDATION: `reconstruction.py:105–164`.** Disconnected catalog graphs are accepted. `cost` (159), `negative_mass` (164) and the dual describe catalog coordinates. But `Finite(zip(primal, graphs))` (162) merges components, so the returned `element` doesn't match them.
```python
r = reconstruct(Finite.from_graph(path(2)).local(1),
                [disjoint_union(path(2), graph(1)), graph(1)])
r.cost, r.negative_mass, [(c, g.rows) for c, g in r.element.terms]
# actual: (4, 1, [(1, (2, 1))])  -> element is just K2: mass 2, no negative part
```
Expected: reject disconnected catalog entries (`catalog()` never makes them), or report the element's actual mass.

**3. VALIDATION/MINOR (budget doesn't bound work): `graphs.py:207–230`.** `search_budget` counts search nodes, but each node scans every unmapped u against every v and the whole mapping, about Θ(n²·m).
- **Isomorphic pairs:** the search uses only about n nodes.
  - `isomorphic(a, b, True, search_budget=231)` succeeds on the 231-vertex L³ ball (230 raises).
  - `((L*L)*L - L*(L*L)).local(5)` takes 7.2 s; one comparison at r=6 (377 vertices) takes 54 s, using 377 of the 100,000 nodes.
- **Non-isomorphic 4-regular pairs** (refinement cannot split them):
  - n=80: about 9 ms per node.
  - n=160: 4000 nodes took 244 s, so the default budget means roughly 100 minutes before `BudgetExceeded`.
- `LocalHistogram.multiply`, `Line.local`, `CutLineDefect.local` and `Finite.local` have no overall work limit. One fuzz trial of `relative_heat` (t=2, degree 7, 52 retained moments) ran 75 s without touching any budget.

**4. MINOR: `elements.py:66, 330, 335`.** `exp_bracket` keeps its own hard-coded 512-term budget, separate from `Exponential.max_terms`.
- `exp(Finite.scalar(600), max_terms=100000).approximate(0, 1, "1e-3")` raises `BudgetExceeded("Exponential majorant exceeds max_terms")`.
- Any input whose weighted norm is above about 510 can't be approximated at any user budget.

**5. MINOR VALIDATION: `graphs.py:68–78`.** `graph()` silently collapses repeated or reversed edges, so a multigraph edge list becomes a simple graph: `graph(2, [(0,1),(1,0)]).edges == 1`. `from_networkx` does reject multigraphs.

**6. TEST-GAP**
- `run_tests.py:20–47`: the "coverage" list is hard-coded text (e.g. "160 independent brute-force isomorphism comparisons"), not measured.
- `test_graphlocal.py:68–78`: isomorphism is only checked on random 4-vertex pairs, mostly separated by degree sequence. Only C6 vs 2C3 exercises backtracking.
- `test_graphlocal.py:127–134`: the grid check only tests `norm(1) == 1+2r(r+1)`, i.e. the vertex count, not the ball's shape.
- `test_defects.py:110–119`: only checks that two library intervals overlap.
- `test_prepared_interactions.py:63–71`: compares the prepared result with `relative_heat`, i.e. the implementation against itself.
- No tests for k-mismatched sources fed to `exp`, disconnected catalogs, or adversarial inexact sources with the error on large balls.

## Checked and found correct

- **Isomorphism, rooted and unrooted:** about 10k pairs against networkx. This included 4×4 rook vs Shrikhande at all 256 root pairs, SRG, circulant and cubic families, random regular graphs and degree-preserving swaps. No class was merged or split across 1,220 histograms.
- **Local data against explicit large finite graphs**, matched with networkx: Line (r≤5), L², L³, E (r≤5), E·L, E², two-cut and I_ℓ (ℓ≤6, r≤5), `LineCutDefect` and `connected_cut_interaction` (including negative positions), random finite products (local and materialized), and the `SparseEdgeDifference` affected-root shortcut (120 random graphs with chords and mixed edits).
- **Locality threshold:**
  - Lazy-walk returns from the induced R-ball are exact through order 2R with a tight D; in 159 of 278 cases order 2R+1 already differs, so the boundary is tight.
  - The radius ceil(M/2) is used consistently in heat, relative heat and prepared queries.
  - `walk_observable` uses radius ⌊ℓ/2⌋ correctly.
- **Heat enclosures:**
  - `heat_return` matches Bessel/mpmath for Line, L², L³, signed combinations and 360 random finite, signed and product inputs (t up to 31/3, ε down to 1e-30).
  - `relative_heat` and `PreparedRelativeHeat` match path/cycle closed forms for E, E·L, multi-cuts and interactions, plus mpmath on 300 random sparse edits with and without products.
  - Also enclosed: 320 adversarial-but-valid inexact sources (error ε/2 placed on K1 or the largest ball, both signs) and 320 random expression trees.
  - I re-derived the tail and normalization algebra: the bounds are monotone in the omitted tail, |A| ≤ β(S−term), and the interval radius stays ≤ ε. The prepared-query plan is monotone in t.
- **Exponential:** certificates match closed-form star coefficients for k=1..3, and `exp_bracket` matches mpmath. The sum, scale and product error formulas are correct.
- **Reconstruction:** `reconstruct` matches scipy's LP solver (HiGHS) on 250 random LPs; 132 infeasibility witnesses were verified. `catalog` counts are 1, 1, 2, 6, 21, 112 and the degree filter matches the networkx atlas.
- **Inputs and caching:** floats, bools, negative radius, k=0, ε≤0, loops, multigraph objects and disconnected histogram keys are all rejected. No float arithmetic appears in these modules. Cache keys include all parameters and outputs are immutable.
- **Metadata:** degree, variation, mass, positivity, edit-bound and moment-profile propagation through sums, scalings and products agrees with the research notes.
