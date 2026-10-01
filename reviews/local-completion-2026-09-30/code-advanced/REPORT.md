# Code review: advanced library modules

- **Assigned:** `python/src/graphlocal/controlled.py`, `edge_interactions.py`, `interaction_moments.py`, `interaction_bounds.py`, `nonspectral.py`, `inverse.py`, `local_inverse.py`, `defect_exponential.py`.
- **Revision reviewed:** `d6aba96`. These modules are unchanged in `65f08bb`.
- **Evidence:** `oracle.py` and the `t_*.py` scripts in this folder.
- **Brief:** [BRIEFS.md](../BRIEFS.md#code-advanced).

This is the reviewer's final report as delivered, except that scratch-directory paths now point to this folder. See the [folder README](../README.md#provenance) for what was not preserved. The lead reviewer reproduced findings 1 and 3 ([lead/outputs/code_findings.txt](../lead/outputs/code_findings.txt)).

---

## Review of controlled, edge_interactions, interaction_moments, interaction_bounds, nonspectral, inverse, local_inverse, defect_exponential

**Bottom line:** I found no unsound certificate for built-in inputs. Every bound and interval I tested against independent oracles held. The findings below are validation holes, usability bugs and test gaps. Scratch scripts are in this folder (`t_*.py`, `oracle.py`).

### Findings, most severe first

**1. VALIDATION: degree-cap projection drops source mass without checking it** (`inverse.py:109-113`, `controlled.py:83-88`)
- The true marginal has zero mass on types above the degree cap D. So any removed mass Σ|c|·|B|^k must be ≤ the source error, but neither file checks this.
- A wrong `degree_bound` therefore produces wrong "certified" results with zero or tiny error.
- The sibling `heat_return` rejects the same input, and the variation and mass consistency checks at `inverse.py:81-82,112-113` show the code intends to catch contradictions.

```python
class Mis(Element):  # Line/4 (degree 2) claiming degree_bound=1
    degree_bound, variation_bound, mass, positive = 1, Q(1,4), Q(1,4), True
    def local(self, r): return (Line()/4).local(r)
NeumannInverse(Mis()).approximation_certificate(1, 1, "1e-6")
#  actual: histogram 1·[K1], error 0  (mass 1, contradicts its own inverse.mass == 4/3)
controlled_heat(Mis(), "1/2", "1e-8").interval   # actual: [-8.6e-10, 8.6e-10]; data give ≈0.1164
heat_return(Mis(), "1/2")                        # raises ValueError: degree bound exceeded
```
- **Expected:** raise when the removed weighted mass exceeds `source.error` (here 3/4 > 0). `NeumannInverse` could also check |mass(ã) − value.mass| ≤ δ.

**2. VALIDATION/MINOR: work budgets don't bound work** (`defect_exponential.py:128-135`; same pattern at `inverse.py:128-131` and `local_inverse.py:142-145`)
- `max_terms` and `max_vertices` cap the series length N and the graph size. They do not cap the number of rooted types or the isomorphism dedupe cost.
- The README (lines 160-161, 207-208) says they "limit work".
- Measured with `CutLineExponential(Q(1,10)).approximation_certificate(2, 1, eps)`:

| eps | N | Types | Time |
|---|---|---|---|
| 1e-1 | 6 | 84 | 6 s |
| 1e-2 | 7 | 120 | 18 s |
| 1e-3 | 8 | 165 | 64 s |

- Each extra term costs about 3.5× more, and the default budgets never trigger.

**3. BUG (minor): axis caching and compatibility** (`nonspectral.py:68,84` together with `179-181,271-273`)
- `lru_cache` keys on the call form, so `rooted_cliques(3)` and `rooted_cliques(size=3)` are distinct objects with the same name. The same holds for `link_components(p)` vs `link_components(p, None)`.
- Axis compatibility is by object identity, so identical laws compare unequal:
  - `joint_distribution(K4/4, (rooted_cliques(3),)) == joint_distribution(K4/4, (rooted_cliques(size=3),))` → actual `False`; both are `{(3,): 1}`.
  - Their jets also compare unequal, and `+`/convolve raise.
- **Fix:** normalize the arguments before caching, or compare axes by (name, radius, growth, function).

**4. MINOR: API holes**
- `interaction_moments`, `tree_interaction_leading` and `interaction_heat_bound` raise TypeError on `GeometricInteraction`, which is documented as the same element (`interaction_moments.py:39-40,105-106`; `interaction_bounds.py:74-75`).
- `controlled_heat(3, 1)` raises AttributeError (`controlled.py:50`, no `as_element`). The other APIs accept scalars.
- `NeumannInverse(0).local(r)` and `CutLineExponential(0).local(r)` raise ExactLocalUnavailable, so `joint_distribution` rejects these trivially exact units.

**5. TEST-GAP**
- **Missing coverage:**
  - `NeumannInverse` is tested only at r=1 (plus scalars at r=3) (`test_inverse.py:24,45`). There is no nontrivial r≥2 tail or stability check and no source of degree ≥2.
  - Inexact-source `refine_local_inverse` is tested only at k=1 (`test_local_inverse.py:87`).
  - `CutLineExponential` error validity at r≥2 is checked only indirectly, through the product with its inverse.
  - `interaction_heat_bound` value enclosure is tested on one fixture only.
- **Overlap-only checks** (two library intervals intersect; no independent truth): `test_interaction_bounds.py:126-128`, `test_controlled_branching.py:153-156`.
- **Tautological or implementation-restating assertions:**
  - `test_local_inverse.py:50` and `test_defect_exponential.py:96`: `to_data` histogram equals `histogram.to_data()`.
  - `test_local_inverse.py:111`: `assertNotIsInstance(certificate, Element)`.
  - `test_local_inverse.py:99`: error equals stability + tail.
  - `test_defect_exponential.py:71`: truncated norm ≤ majorant.
- **Verifier duplicates the library:** `examples/unbounded_inverse_verification.py:123-135` re-implements the library's tail formula verbatim. It checks only against partial sums of that same majorant (lines 242-249), never against the library's certificate.

### Checked and found correct (independent oracles)

- **NeumannInverse**
  - q is genuinely a global variation bound for every built-in constructor (Finite, Line, Sum, Scale, Product, SparseEdgeDifference, EdgeInteraction, PreparedLocal).
  - `_power_majorant` and its derivative match mpmath series; the stability bound using max(q, q̂) is valid.
  - Exact hypercube oracle, c ∈ {4, 2, 3/2}, r=0..3, k=1..3: coefficients exact and exact weighted tail ≤ error ≤ ε.
  - Adversarial ± in-support noise: true error ≤ certified.
  - `degree_bound None` for H/4 is justified, since the inverse contains hypercubes of every dimension.
- **local_inverse**
  - The residual is computed in the weighted norm ‖·‖_{r,k}, which is submultiplicative because |B⋆C| ≤ |B||C|.
  - Inexact-source handling (q = q0 + δB, B/(1−q), Bq/(1−q)) is correct.
  - The refinement error is at most ε/3 + ε/2, so it never exceeds ε; adversarial noise at k=1,2 checked against exact inverses.
- **defect_exponential**
  - At r=1, coefficients match exp(2tz−2tz²) and the truncation agrees with a 12-term-longer series.
  - At r=2,3 an independent product-of-path-atoms power-norm formula matches the library histograms exactly.
  - Exact tails ≤ the README majorant for r=2,3,4, k=1..3, t ∈ {1/50, 1/8, 1/2}, all N.
  - Local variation equals e^{4r|t|}.
- **interaction_moments:** 127 random mixed insert/delete cases (random orientation, normalization, reduction) match exact subset matrix traces, both Laplacian and uniformized.
- **tree_interaction_leading:** 200 random trees with extra branches match the brute-force onset and coefficient.
- **interaction_geometry:** |d_j| ≤ `moment_bound` ≤ profile, vanishing order correct, and the profile stays valid at D+1 and D+3.
- **interaction_heat_bound:** magnitude and `after_step` (0, 1, 3, 6) bounds hold against 60-digit eigenvalue heat traces in 153 cases.
- **controlled_heat:** all enclosures contain the true value for EdgeInteraction and GeometricInteraction (153 cases), 16 sum/product/scale compositions with L, E and E², and the infinite defects I_ℓ, TwoCut, LineCutDefect, I₂·E, I₂·L. The documented sign-reversal values are reproduced.
- **Bridge reduction:** 94 random bridge sets with cycle blocks agree with raw inclusion–exclusion in both `finite()` and `local(0..3)`.
- **Jets:** `certified_jet` intervals contain the exact moments for NeumannInverse(H/c) and CutLineExponential; MomentJet reciprocal, log, exp, product, power and truncation match sympy series.
- **Validation:** floats, bools, negative radius, k=0, ε≤0, negative t, q≥1, residual ≥1 and the budgets are all rejected correctly.
- **Examples:** `certified_arithmetic`, `unbounded_inverse_examples`, `geometry_bound_examples` and `controlled_heat_examples` reproduce their recorded JSON.
