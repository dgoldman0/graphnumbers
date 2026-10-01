#!/usr/bin/env bash
# Compare the committed PDF of the v0.1 note with its TeX source (needs poppler-utils).
# Run from the repository root.
cd research/local-completion || exit 1
pdfinfo local_graph_completion.pdf | grep -E "Title|Producer|CreationDate|Pages"
for phrase in "not normable" "Ordinary finite graphs remain discrete" \
              "Measure representation remains unclassified" "cospectral regression pair" \
              "All 3,606 recorded checks passed"; do
    printf "%-46s pdf=%s tex=%s\n" "$phrase" \
        "$(pdftotext local_graph_completion.pdf - | tr '\n' ' ' | grep -o "$phrase" | wc -l)" \
        "$(tr '\n' ' ' < local_graph_completion.tex | grep -o "$phrase" | wc -l)"
done
pdftotext local_graph_completion.pdf - | grep -cE "^(Lemma|Proposition|Theorem) [0-9]" | sed 's/^/numbered results in PDF: /'
grep -cE "\\\\begin\{(lemma|proposition|theorem)\}" local_graph_completion.tex | sed 's/^/numbered results in TeX: /'
