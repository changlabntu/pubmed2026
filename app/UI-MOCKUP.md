# Module 1 v4: UI mockup

Text sketch of the Streamlit page. `[ ]` = button, `( )` = radio,
`______` = text input, `▸` = collapsed expander.

```
┌─ sidebar ───────────────────────────┐  ┌─ main ───────────────────────────────────────────────┐
│ Question set                        │  │ Module 1 v4: argue a research gap                     │
│   [AI for lung cancer screening ▾]  │  │                                                       │
│                                     │  │ 1. The claim                                          │
│                                     │  │   Topic question (the territory) __________________   │
│                                     │  │   The gap (the niche)      What would refute it       │
│ Model          (teacher mode only)  │  │   ____________________     ____________________       │
│   (•) Haiku  $0.10 / $0.50 per M    │  │                                                       │
│   ( ) Sonnet (greyed out)           │  │ 2. Start from three papers                            │
│                                     │  │   Corpus: (•) my own PubMed query ( ) shipped pool    │
│ Cost cap       (teacher mode only)  │  │   PubMed query ______________________________________ │
│   [=====|---------]  10 cents       │  │   Anchor + two companions: 41308576, 42229468, ...    │
│                                     │  │   Max records  [====|-------]  200                    │
│ Spend          (teacher mode only)  │  │   Scoring prompt. EDIT THIS        [ Reset all ]      │
│   Model calls   15                  │  │     BACKGROUND  ______________________________        │
│   Cost          1.6 cents           │  │     WHAT TO DO  ______________________________        │
│   ▸ Per move                        │  │     OUTPUT FORMAT ____________________________        │
│                                     │  │   [ Fetch and score the three ]                       │
│ [ Download report (.md) ]           │  │   PubMed matched 183 records; 178 form the corpus.    │
└─────────────────────────────────────┘  │   ┌ Thesis: for the gap (1) ┐ ┌ Antithesis (0) ┐       │
                                         │   │ Ledda et al. 2026 (anchor)│ │ none          │       │
                                         │   └───────────────────────────┘ └───────────────┘       │
                                         │   Neutral: Damiani et al. 2026  Zhang et al. 2021      │
                                         │   (t-SNE map: grey dots, the three outlined + named)   │
                                         │   ▸ The three records in full                         │
                                         │   Comment: ___________________________________________ │
                                         │                                                       │
                                         │ 3. Gather evidence                                    │
                                         │   [ Find evidence ]                                   │
                                         │   ┌ Thesis: for the gap (6) ┐ ┌ Antithesis: against (4)┐│
                                         │   │ Gebremichael et al. 2026│ │ Peeters et al. 2026   ││
                                         │   │ Benzaquen et al. 2024   │ │ Jiang et al. 2026     ││
                                         │   │ ...  (hover: title,     │ │ ...                   ││
                                         │   │       score, why)       │ │                       ││
                                         │   └─────────────────────────┘ └───────────────────────┘│
                                         │   18 papers scored: the three plus 15 found ...        │
                                         │   (map above recoloured: green / red / grey)           │
                                         │   ▸ The two searches and every record in full         │
                                         │   Comment: ___________________________________________ │
                                         │                                                       │
                                         │ 4. The argument                                       │
                                         │   ▾ Argument prompt. EDIT THIS   (red-edged boxes)    │
                                         │     BACKGROUND ______  WHAT TO DO ______  OUTPUT ____ │
                                         │   ▸ Verifier prompt. EDIT THIS   (red-edged boxes)    │
                                         │   [ Write the argument and verify ]                   │
                                         │   Thesis. ...  [PMID n]                               │
                                         │   Antithesis. ...  [PMID n]                           │
                                         │   Synthesis. ...                                      │
                                         │   Verdict: gap narrowed                               │
                                         │   12 claim checks: 9 supported, 2 not supported, ...  │
                                         │   ┌ verdict ─ pmid ─── claim ─── issue ┐              │
                                         │   └────────────────────────────────────┘              │
                                         │   Comment: ___________________________________________ │
                                         └───────────────────────────────────────────────────────┘
```

## Behaviour notes

Every button first checks the ledger against the cost cap. Editing the
claim, the PMIDs, the corpus source or a prompt clears that move and
everything below it, while comments survive. A reply that does not parse is
shown as raw text with a yellow note, and the note says when the reply was
cut off at the output limit. The stance of a paper is computed from the
model's relevance field and its score, and the table in the expander shows
every key the model returned. The download button writes one Markdown file
in the order of the argument: the claim, the starting papers, the evidence
for, against and neutral, how the corpus was searched, the argument with its
verdict, the verification, and an appendix.
