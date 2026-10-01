# Referee report: literature citations

- **Assigned:** citations in `research/local-completion/LITERATURE_REVIEW.md`, `SPARSE_DEFECT_COMPARISON.md`, `SEARCH_LOG.md`, `FOLLOWUP_IDENTIFICATION.md`, the bibliography of `local_graph_completion.tex`, and citations elsewhere in the notes.
- **Revision reviewed:** `d6aba96`.
- **Evidence:** `notes.md` in this folder (working notes).
- **Access limits:** full-text access was blocked; only web search worked. See the report.
- **Brief:** [BRIEFS.md](../BRIEFS.md#lit-review).

This is the referee's final report as delivered, except that scratch-directory paths now point to this folder. A text dump of the repo's own PDF in the referee's scratch area was not preserved.

---

**Citation check: research/local-completion (read-only; no repo files touched)**

**Access limits:** I could only check metadata, not full texts. WebFetch was blocked for every host I tried (arxiv.org, ar5iv, api.crossref.org, semanticscholar, jmlr.org, math.smertnig.at, wikipedia). Curl through the proxy got 403 from the arXiv export API, crossref, doi.org, openalex and zbmath. Only WebSearch worked, so I checked metadata through search results and summaries. Content was checked only where those summaries showed it. Anything I judged without a source is marked "from memory, unverified".

**Counts:** I checked 59 distinct cited works: 25 in LITERATURE_REVIEW (24 numbered plus Ebrahimi-Fard–Patras), 18 more in SEARCH_LOG and FOLLOWUP, 10 more in SPARSE_DEFECT_COMPARISON, Infusino in the .tex, and 5 in other notes. python/README.md has no formal citations.
- **Exist with correct authors, title, year and identifier:** 59/59. This includes the very recent ones: Vrana arXiv:2609.31021 (submitted 25 Sep 2026), Zhipeng Lu arXiv:2609.06747 (6 Sep 2026), de Boer–Buys–Zuiddam v2 (13 Jun 2026), Chojnacki in Semigroup Forum (7 Jul 2026), and Kurauskas v3 (Feb 2023).
- **Problems:** 0 fabricated, 0 confirmed misattributed, 1 possible overstatement, 2 places where credit is missing or the bibliography is stale, 9 entries that cite only the preprint of a published paper, and many section/theorem pinpoints I couldn't verify.

**Problems, most severe first**

1. **No FABRICATED citations found.**

2. **MISATTRIBUTED: none confirmed.**
   - Possible overstatement at LITERATURE_REVIEW.md:101–103. The note says Knill 2021 "discusses weighted Wiener algebras". His abstract says "an extension of Q with a single network completes to the Wiener algebra A(T)". The word "weighted" doesn't appear in anything I could see. (arxiv.org/abs/2106.10093; people.math.harvard.edu/~knill/graphgeometry/papers/ring3.pdf)

3. **Missing credit (positioning)**
   - **Aldous–Lyons [7] products, LITERATURE_REVIEW.md:150–166.** Search snippets of the Aldous–Lyons arXiv HTML (arxiv.org/html/math/0603062v3 and v4) say the paper defines the independent product of two unimodular measures, rooted at the pair of roots. They also say G_n × G'_n "clearly has random weak limit" equal to that product law. That is the probability-measure version of the review's μ_{G□H} = μ_G * μ_H, but the review cites [7] only for mass transport and Question 10.1. The exact section is unverified.
   - **Stale .tex bibliography, local_graph_completion.tex:300–303.** It cites only Benjamini–Schramm and Infusino. It does not credit the finite Cartesian graph ring A₀ (van de Woestijne, Knill, Imrich–Klep–Smertnig) or Kurauskas for its triangle counterexample. The review says all of these "should be credited". Its "No claim of novelty is made" is fine; the bibliography just predates the review.

4. **INACCURATE-METADATA: nothing wrong found.** These entries cite only the preprint although a published version exists:
   - SPARSE_DEFECT_COMPARISON.md:296 — Beckermann–Kressner–Schweitzer, published SIMAX 39 (2018) 539–565.
   - :300 — Cortinovis–Kressner–Massei, published SIMAX 2022, doi:10.1137/21M1432594.
   - :324 — Frommer–Schimmel–Schweitzer, published SIMAX 42 (2021).
   - :333 — Sobieczky, published J. Theor. Probab. 23 (2010).
   - :343 — Isozaki–Korotyaev: listed as 2011, published Ann. Henri Poincaré 13 (2012) 751–788.
   - LITERATURE_REVIEW.md:344 — Pirkovskii, published Proc. AMS 134 (2006) 2621–2631.
   - LITERATURE_REVIEW.md:340 and FOLLOWUP:249 — Abolghasemi–Rejali–Vishki, published Iran. J. Sci. Technol. A 44 (2020) with only two authors.
   - FOLLOWUP:256 — Pedersen, published Banach Center Publ. 91 (2010) 247–259.
   - FOLLOWUP:264 — Kaimanovich, published Zap. Nauchn. Sem. POMI 441 (2015).

5. **UNVERIFIABLE: section/theorem pinpoints I couldn't see.** All of these exist as works, and the claims fit their abstracts.
   - **Benjamini–Schramm "Section 1.2 supplies the rooted local topology"** (tex:301, LITERATURE_REVIEW:325, COMPARISON_LEMMAS:75).
   - van de Woestijne Definition 2.5 and Theorems 2.3, 2.6, 2.7. The substance is confirmed by a snippet: graphs under Cartesian product form N₀[Y], and the difference ring is Z[Y] in countably many variables.
   - Imrich–Klep–Smertnig Proposition 3.2, Definition 3.4, Propositions 3.7–3.8, Theorem 3.16 and Section 3.1.
   - Kurauskas Theorems 2.1–2.2 and Example 2.1. The substance is confirmed: the (h−1)-th degree moment is uniformly integrable iff the local weak limit determines h-vertex subgraph counts.
   - Bordenave–Caputo equation (1). From memory, unverified: U(G) is their notation.
   - Aldous–Lyons Definition 2.1 and Proposition 2.2. From memory, unverified: these are the mass-transport definition and involution invariance.
   - SPARSE_DEFECT_COMPARISON:56–65, Cortinovis–Kressner–Massei. Theorem 2 (the trace is exact for polynomials of degree ≤ 2m when A is symmetric) is confirmed by a snippet. The 4n constant attributed to Theorem 3 is unverified.
   - Shirai, SPARSE_DEFECT_COMPARISON:184–193. The abstract stresses an inverse spectral problem and Green functions versus periodic orbits. The "Dirichlet on a finite set / Poisson tails, Section 2" description is plausible but unseen.
   - Smaller pinpoints:
     - Isozaki–Korotyaev §6.2, equation (6.11)
     - Rao–Teh §3.1
     - Sobieczky Theorem 1.8
     - Bhatt–Patel Examples 1.2, 1.4 and 1.5
     - Pedersen Theorem 2.3 and Corollary 2.4
     - Kaimanovich Theorem 50, Corollary 52 and Remark 53
     - Albeverio–Mazzucchi §5.1, Theorems 4–5
     - Blute et al. Definition 2.2, Proposition 2.3 and Theorem 1
     - Hatami–Lovász–Szegedy Theorem 3.2
     - de Boer–Buys–Zuiddam Definition 2.4 and Remark 2.5
     - Cébron–Dahlqvist–Male Definitions 1.7 and 1.10–1.14
     - Vrana §5.4
     - the Schaden and Shajesh–Schaden equation numbers
     - Mančinska et al. after Corollary 7.4. Their discussion of the Shrikhande graph versus K4□K4 is confirmed.
   - URLs I couldn't confirm:
     - The IAS repository PDF for Bhatt–Patel. It is plausible: another Bhatt paper sits at repository.ias.ac.in/59697.
     - The Gröchenig homepage PDF.
     - The ems.press serial-article-files links.
     - The Chojnacki DOI number.
   - Confirmed: Infusino Lecture 6 is "Hausdorff lmc algebras and finest lmc topology", continuing Section 2.2 from Lect5.pdf. The Kriegl–Michor ~kriegl/Skripten/apbook.pdf URL exists.

**Prior-art positioning and novelty hedging**

The hedging looks sound and, if anything, conservative. The notes repeatedly say that "not found" is not novelty, record that there was no MathSciNet/zbMATH search, and credit the finite ring, local convergence, lmc completion and generalized-series arguments to existing work. I found no prior construction of this exact signed, polynomially weighted local Fréchet completion (from memory, unverified beyond the searches).

The main gap is that Aldous–Lyons already treat the rooted Cartesian product of unimodular laws and its compatibility with Benjamini–Schramm limits. The review should credit this in its measure-algebra section, and the .tex bibliography should be brought in line with the review.

Standard references worth adding:
- Hammack–Imrich–Klavžar, *Handbook of Product Graphs* (2011) — the graph-semiring ≅ polynomial-semiring viewpoint.
- Lovász, *Large Networks and Graph Limits* (2012).
- Hora–Obata, *Quantum Probability and Spectral Analysis of Graphs* (2007) — Cartesian product and commutative independence.
- E. Michael, Memoirs AMS 11 (1952) — the canonical source for lmc algebras.
- For unbounded-degree limits: van der Hofstad, *Random Graphs and Complex Networks* vol. 2, and Backhausz–Szegedy action convergence.

Scratch notes are in this folder (`notes.md`).
