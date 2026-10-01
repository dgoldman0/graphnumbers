# Review of the local-completion work of 30 September 2026

This review covers the 17 commits on `main` from `09c6fef` (12:26) to `65f08bb` (20:00) on 30 September 2026 (times UTC−4), about 36,000 added lines. They comprise:

- the v0.1 note `local_graph_completion.tex`;
- 26 later mathematical notes in `research/local-completion/`;
- the literature review notes;
- the `graphlocal` library (versions 0.1 through 0.8.0) with its examples, verifiers and tests.

The review was conducted on 30 September – 1 October 2026 in a Claude Code session: a lead reviewer plus 14 parallel referees.

## Verdict

The mathematics holds up. No false theorem or invalid proof was found anywhere, and every quantitative claim the referees recomputed from scratch matched. Some independent checks went far beyond the repository's own; for example, the tree onset theorem held on 310,419 cut sets against the 249 the notes check.

The library's certified bounds also held on every path built from its own elements.

The weaknesses are in how results and evidence are described:

- Several README sentences claim more than the notes prove; the most significant is the independence statement added in `65f08bb`.
- Some notes describe verification checks that do not exist, and the headline check counts include tautologies.
- The library has validation gaps for custom elements and work budgets that do not actually bound work.

The full register is in [FINDINGS.md](FINDINGS.md).

## Results at a glance

| Area | Outcome |
|---|---|
| Mathematics | 0 errors. 6 proof gaps whose conclusions are true; 18 statement-level corrections (missing r ≥ 2 qualifiers and similar). |
| Summaries | 9 documentation overclaims, D1–D9. |
| Library | No unsound certificate on built-in inputs. 8 findings: 2 checks missing against contract-violating custom elements, 2 bugs, unbounded work budgets, assert-only validation in the reconstruction tool, and API holes. |
| Evidence | 6 findings about checks that are absent, hard-coded, tautological or overcounted. |
| Literature | All 59 cited works exist with correct metadata. Missing Aldous–Lyons credit; stale TeX bibliography; 9 preprint-only citations. |
| Reproducibility | All 18 deterministic recorded results regenerate identically; 127/127 tests pass; all README code blocks run; the PDF matches its TeX source. |

## Next steps

[DEVELOPMENT_PATH.md](DEVELOPMENT_PATH.md) is a research plan building on this review (1 October 2026). It contains:
- the three directions the review identified;
- the structure problems for the rooted-ball monoids that all three depend on;
- near-term theorems with proof sketches, each labelled known, proposed, conjecture or open;
- an order of work.

## Method

Each referee received a written brief ([BRIEFS.md](BRIEFS.md)) and worked independently, with read-only access to the repository. The briefs required checking proofs line by line and recomputing claims with code written from scratch. The repository's own verifiers were not to be counted as evidence. Referees were to flag checks that do not test what they claim, and to grade findings as ERROR, GAP, OVERCLAIM or MINOR, calling something an ERROR only with a concrete counterexample or failing step.

The lead reviewer:

- checked the core construction by hand;
- reran all deterministic results, tests and documentation examples;
- read new code directly;
- reproduced the code findings and confirmed the documentation findings against the repository text.

See [lead/REPORT.md](lead/REPORT.md).

Environment: Python 3.11.15 with numpy 2.4.6, scipy 1.17.1, sympy 1.14.0, networkx 3.6.1 and mpmath 1.3.0, plus poppler-utils for the PDF comparison. The extra packages were installed outside the repository; the library itself needs none of them.

## Coverage map

| Folder | Scope | Revision |
|---|---|---|
| [lead](lead/REPORT.md) | `local_graph_completion.tex` by hand; reproducibility; README examples; check counts; PDF/TeX; `medium_defects.py`; confirmation of findings | `65f08bb` |
| [m1-representation](m1-representation/REPORT.md) | `REPRESENTATION_PROBLEM`, `REPRESENTATION_THEOREM`, `ANALYSIS`, `COMPARISON_LEMMAS` | `d6aba96` |
| [m2-units](m2-units/REPORT.md) | `MULTIPLICATION_AND_UNITS`, `CHARACTERS_AND_INVERSION`, `FOLLOWUP_IDENTIFICATION` (mathematics) | `d6aba96` |
| [m3-approx-intrinsic](m3-approx-intrinsic/REPORT.md) | `QUANTITATIVE_APPROXIMATION`, `INTRINSIC_GRAPH_STRUCTURE`, `REFLECTION_EXTENSION`, `reconstruct_local.py` | `d6aba96` |
| [m4-heat-defects](m4-heat-defects/REPORT.md) | `EFFECTIVE_ANALYSIS_AND_HEAT`, `SPARSE_DEFECTS_AND_RELATIVE_HEAT`, `DEFECT_INTERACTIONS`, `GEOMETRY_VERSUS_SPECTRUM` | `d6aba96` |
| [m5-branch-planar](m5-branch-planar/REPORT.md) | `BRANCHING_DEFECTS`, `PLANAR_DEFECTS` | `d6aba96` |
| [m6-higher-decay](m6-higher-decay/REPORT.md) | `HIGHER_INTERACTION_GEOMETRY`, `INTERACTION_DECAY` | `d6aba96` |
| [m7-arith-nonspectral](m7-arith-nonspectral/REPORT.md) | `GEOMETRIC_ARITHMETIC`, `NONSPECTRAL_CALCULUS`, `NONSPECTRAL_INTRINSIC` | `d6aba96` |
| [m8-unbounded-inversion](m8-unbounded-inversion/REPORT.md) | `UNBOUNDED_VARIATION_INVERSION`, `EFFECTIVE_LOCAL_INVERSION` | `d6aba96` |
| [n1-branching](n1-branching/REPORT.md) | `BRANCHING_ARITHMETIC` | `65f08bb` |
| [n2-planar](n2-planar/REPORT.md) | `PLANAR_ARITHMETIC` | `65f08bb` |
| [n3-mixed](n3-mixed/REPORT.md) | `MIXED_MEDIUM_ARITHMETIC` | `65f08bb` |
| [lit-review](lit-review/REPORT.md) | Citations in all notes and the TeX bibliography | `d6aba96` |
| [code-core](code-core/REPORT.md) | `graphs`, `elements`, `local`, `heat`, `reconstruction`, `defects`, `prepared`, `interactions` | `d6aba96` |
| [code-advanced](code-advanced/REPORT.md) | `controlled`, `edge_interactions`, `interaction_moments`, `interaction_bounds`, `nonspectral`, `inverse`, `local_inverse`, `defect_exponential` | `d6aba96` |

Notes and library modules reviewed at `d6aba96` are unchanged in `65f08bb`. That commit adds new files and updates the READMEs, version number, package exports, test runner and recorded test results.

## Provenance

- **Reports.** Each `REPORT.md` is the referee's final report as delivered, preceded by a short header. The only edit inside a report is that scratch-directory paths now point to the referee's folder here. The referees are model-generated reviewers; the lead reviewer reproduced or confirmed the findings marked ✓ in [FINDINGS.md](FINDINGS.md).
- **Scripts and outputs.** The files beside each report are the referee's own scripts and output logs, kept as they were run. Some hard-code paths from the review machine, such as `sys.path.insert(0, '/home/user/graphnumbers/python/src')`, and one log ends in a traceback containing scratch paths.
- **Lead outputs.** Everything in `lead/outputs/` was produced at commit `65f08bb`, and each file records the commit it ran on.
- **Not preserved**, to keep this folder small and free of duplicates:
  - copies of the repository that some referees made to run its verifiers;
  - Python bytecode caches;
  - pickled computation caches (5.4 MB in m1, 0.7 MB in n2), which the scripts regenerate;
  - outputs of the repository's own verifiers and examples that referees reran, which are identical to the committed results (see [regenerate_and_compare.txt](lead/outputs/regenerate_and_compare.txt));
  - a binary profiler dump, a text dump of the repository's own PDF, an environment file listing scratch paths, and empty log files.

## Reproducing

The lead checks run from the repository root (the heat-value check also needs mpmath on `PYTHONPATH`):

```sh
R=reviews/local-completion-2026-09-30/lead
bash $R/regenerate_and_compare.sh                # all deterministic results vs committed JSON
(cd python && PYTHONPATH=src python3 tests/run_tests.py)
python3 $R/run_readme_blocks.py
python3 $R/check_counts.py
PYTHONPATH=python/src python3 $R/reproduce_code_findings.py
bash $R/reproduce_reconstruct_local.sh
PYTHONPATH=python/src python3 $R/lattice_heat_value.py
python3 $R/benchmark_timings.py
bash $R/pdf_vs_tex.sh                            # needs poppler-utils
```

Referee scripts run from their own folder. Put the repository's `python/src` and a site directory containing numpy, scipy, sympy, networkx and mpmath on `PYTHONPATH`. Some are long exhaustive searches; for example, `m6-higher-decay/tree_exhaustive.py` enumerates every cut set of every tree with up to 11 vertices.
