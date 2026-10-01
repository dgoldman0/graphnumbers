# Lit review check notes (scratch)
Access: curl via proxy -> 403 for arxiv export API, crossref, s2, doi.org, openalex, zbmath, google.
WebFetch -> EGRESS_BLOCKED for arxiv.org, ar5iv, crossref, smertnig.at, jmlr.org, semanticscholar, wikipedia.
WebSearch works (titles/URLs + snippet summaries only).

## Findings
- vdWoestijne: AMC 5(2) 2012 303-319, arXiv 1103.0709 OK. Thm numbers unverified.
- IKS: ADAM 8(1) P1.11 2025; v2 13 Feb 2025 OK. Venue unnamed in review (minor). DOI prefix consistent.
- Knill 2017/2019/2021 exist; 2019 abstract confirms real/complex Banach completion with join. 2021 abstract confirms Banach algebra R. Wiener? check.
- BS: EJP 6 (2001) paper 23 pp1-13, DOI 10.1214/EJP.v6-96; v4 Oct 2001. Section 1.2 unverified.
- AL: EJP 12 (2007) paper 54 1454-1508; v6 1 Nov 2018 OK; Question 10.1 confirmed via BCLV snippet.
- BC: PTRF 163 (2015) 149-222 DOI OK.
- Kurauskas: JAP 59 (2022) 755-776; v3 4 Feb 2023 OK. Sidorenko inequality in abstract.
- ARV: arXiv 2008; ALSO published Iran J Sci Technol Trans A 2020 (DOI 10.1007/s40995-020-00818-2) -- not mentioned (minor).
- Pirkovskii 2004 OK (published PAMS 2006? not checked).
- Brown-George 2505.04882 v1 title "On the Roots of Degree Polynomials", now "The Degree Polynomial" OK.
- Infusino: Lect5.pdf = "2.2 Seminorm characterization of lmc algebras"; Lecture 6 = "More on seminorm characterization: Hausdorff lmc algebras and finest lmc topology" -> consistent.
- FLS JAMS 20 (2007) 37-51 OK. Razborov JSL 72(4) 1239-1282 DOI OK. Austin 2008 OK. LS JCTB 96 (2006) 933-957 DOI OK.
- CDM Doc Math 29 (2024) 39-114 OK. ALS IDAQP 10(3) 2007 303-334 OK.
- dBBZ v2 13 June 2026 OK. Vrana 2609.31021 exists, submitted 25 Sep 2026, 58pp, tensors & graphs semiring examples OK.
- BCLV 2408.00110 OK; BCV 2501.00173 OK.
- HLS GAFA 24 (2014) 269-296; arXiv title "local-global" OK.
- EF-P: Proc R Soc A 471 (2015) 20140843 OK.
- Artemenko: MSc thesis Ottawa 2013 OK.
- Albeverio-Mazzucchi RMP 28(2) 2016 1630001 OK.
- HIK 1303.6803 (MCS 7 2013 255-273) OK; 1308.2101 (AMC 9 2015) OK.
- BCJS 1805.09836 ENTCS 341 (2018) 5-22 OK.
- Bhatt-Patel BAMS 66(1) 2002 135-148 OK; IAS URL unverified.
- Pedersen 0909.2749 OK (Banach Center Publ 91 (2010) 247-259). weights on R+.
- Kaimanovich 1512.08479 OK (Zap. POMI 441 2015).
- Bravo-Hermsdorff JMLR 24 (2023) paper 187 pp1-27 OK.
- Foissy 2301.09449 OK (v final Sep 2024).
- DPR Banach Center Publ 91 (2010) 123-158 OK. Patel 1410.1695 OK.
- Ribenboim J Algebra 168 (1994) 71-89, DOI 10.1006/jabr.1994.1221 OK.
- Zhipeng Lu 2609.06747 exists (6 Sep 2026); abstract matches "vertex-deleted regular graph" description OK.
- Chojnacki Semigroup Forum 7 July 2026 exists. DOI number unverified.
- Groechenig-Leinert TAMS 358 (2006) 2695-2711 OK.
- Bordenave-Lelarge RSA 37 (2010) OK.
- Rao-Teh JMLR 14 (2013) 3295-3320 OK.
- BKS SIMAX 39 (2018) 539-565 (note says 2017 preprint; minor).
- CKM: Theorem 2 = trace exactness Pi_2m confirmed by snippet. SIMAX 2022 not mentioned.
- Shuman et al OK (IEEE TSIPN 2018). Ubaru SIMAX 38(4) OK. Tsitsulin, Munkhoeva, Perozzi WWW 2020 OK.
- Benzi-Razouk ETNA 28 (2007) 16-39 OK. FSS SIMAX 42(3) 2021 OK. Sobieczky JTP 23 (2010) OK.
- Shirai PRIMS 34(1) 1998 27-41 DOI OK; abstract: inverse spectral problem, Green fn & periodic orbits; Dirichlet-on-finite-set claim unverified.
- Isozaki-Korotyaev AHP 13 (2012) 751-788 OK; SSF for trace class potentials confirmed.
- Schaden EPL 94 (2011) 41001 OK; Shajesh-Schaden PRD 83 125032 OK.
- Shrikhande AMS 30(3) 1959 781-798 DOI OK. MPRR EJC 87 2020 OK; discusses Shrikhande vs K4xK4 cospectral.
- Speed AJS 25(2) 1983 378-388 OK.
- Kriegl-Michor URL ~kriegl/Skripten/apbook.pdf exists OK.

## Snippet-level content confirmations
- vdW: M ≅ N0[Y], D(M) ≅ Z[Y] polynomial ring countably many vars (snippet) -> supports LR:78-83.
- IKS: countably many components, generalized power series rings (abstract).
- Knill 2019: real/complex Banach completion, join addition (abstract). Knill 2021: "single network completes to Wiener algebra A(T)"; "weighted" not seen.
- Kurauskas: D^(h-1) UI iff local limit determines h-vertex subgraph counts (snippet).
- AL: Question 10.1 (via BCLV); AL define independent product of unimodular measures rooted at pair, G_n x G'_n -> product law (snippets of arxiv html v3/v4) -> uncredited in LR:150-166.
- CKM Theorem 2: tr X_m(p) exact for p in Pi_2m, A symmetric (snippet). 4n constant unverified.
- Infusino Lect5 = "2.2 Seminorm characterization"; Lecture 6 = Hausdorff lmc + finest lmc topology.
- Foissy: disjoint union is product (abstract). Kaimanovich: modular cocycle (abstract). IK: SSF trace-class potentials (abstract). Speed: Mobius partition lattice. Ubaru: analytic f, SPD. MPRR: Shrikhande vs K4xK4 cospectral.
- Shirai abstract: inverse spectral problem; Green fn and periodic orbits. Dirichlet/Poisson-tail details unverified.
## Incomplete metadata (published versions not cited): BKS SIMAX39(2018); CKM SIMAX(2022); FSS SIMAX42(2021); Sobieczky JTP23(2010); IK AHP13(2012); Pirkovskii PAMS134(2006); ARV IJST-A 44(2020, 2 authors); Pedersen BCP91(2010); Kaimanovich ZapPOMI441(2015).
## TeX bib: only BS + Infusino; no credit to graph ring (vdW/Knill/IKS) or Kurauskas.
