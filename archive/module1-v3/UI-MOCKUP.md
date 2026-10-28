# Module 1 v3: UI mockup

Text mockup of the Streamlit page, drawn before building; the word-limit
slider was dropped (edit the number in the prompt) and the anchor dropdown
moved to Q2. `[ ]` = button, `( )` = radio, `[x]` = toggle,
`______` = text input, `▸` = collapsed expander, `▾` = expanded.

```
┌─ sidebar ───────────────────────────┐  ┌─ main ───────────────────────────────────────────────┐
│ Pool                                │  │ Module 1 v3: build your own literature review          │
│   knee_mri                          │  │                                                       │
│                                     │  │ Q1. Build corpus                                      │
│ Model                               │  │   (•) use a shipped pool  ( ) my own PubMed query     │
│   (•) Haiku  $0.10 / $0.50 per M    │  │   Pool: [knee_mri ▾]                                  │
│   ( ) Sonnet $3 / $15 per M (30x)   │  │   PubMed query ______________________________________ │
│                                     │  │   Max records  [====|-------]  200                    │
│ Cost cap                            │  │   [x] exclude reviews, editorials, letters            │
│   [=====|---------]  5 cents        │  │   [ Build corpus ]                                    │
│                                     │  │   119 documents, 119 x 1536 vectors, from cache       │
│ Spend this session                  │  │   Report: document count                              │
│   Model calls   14                  │  │   My comment: _______________________________________ │
│   Cost          $0.0071             │  │                                                       │
│   ▸ Per step                        │  │ Q2. Grade three fixed papers                          │
│                                     │  │   Research question ________________________________  │
│ [ Download report (.md) ]           │  │   Anchor:    32055951 Chang 2020 ...                  │
└─────────────────────────────────────┘  │   Companion: 40885690 ...   Companion: 33337584 ...   │
                                         │   ▾ System prompt (edit me)            [ Reset ]      │
                                         │   ┌─────────────────────────────────────────────────┐ │
                                         │   │ You score each paper 1-10 against the gap:     │ │
                                         │   │ question. Return a JSON array, one object per   │ │
                                         │   │ pmid, bears (yes|no), score (1-10),             │ │
                                         │   │ cohort, n, headline_result, evidence_quote, why │ │
                                         │   └─────────────────────────────────────────────────┘ │
                                         │   [ Grade ]                                           │
                                         │   ┌ pmid ─── stance ──── score ─ gap_relation ─ flag ┐│
                                         │   │ 32055951  thesis       8     partial             ││
                                         │   │ 40885690  neutral      -     -                   ││
                                         │   │ 33337584  antithesis   2     counterexample      ││
                                         │   └──────────────────────────────────────────────────┘│
                                         │   ▸ 32055951 thesis: why and evidence                 │
                                         │   ▸ raw model reply                                   │
                                         │   Report: which PMID scores lowest and why           │
                                         │   My comment: _______________________________________ │
                                         │                                                       │
                                         │ Q3. Search the corpus                                 │
                                         │   Strategy: (•) embedding ( ) anchor's neighbours     │
                                         │   [ Search ]                                          │
                                         │   Anchor rank: 2   (or: 10 nearest to the anchor)     │
                                         │   ┌ rank ─ sim ─── pmid ─── paper ─── stance ┐        │
                                         │   │ 1  0.61  41308576  Ledda 2026   antithesis│        │
                                         │   │ ...                                     │         │
                                         │   └─────────────────────────────────────────┘         │
                                         │   Report: stance counts, anchor rank                  │
                                         │   My comment: _______________________________________ │
                                         │                                                       │
                                         │ Q4. Synthesis                                         │
                                                                                  │   ▸ Synthesis prompt            [ Reset ]             │
                                         │   ▸ Verifier prompt             [ Reset ]             │
                                         │   [ Summarize and verify ]                            │
                                         │   THESIS / ANTITHESIS / SYNTHESIS with [PMID n] cites │
                                         │   18 claim checks: 11 supported, 7 partial, 0 unsup.  │
                                         │   ┌ verdict ─ pmid ─── claim ─── issue ┐              │
                                         │   │ partial  42383283  All thesis ...  │              │
                                         │   └────────────────────────────────────┘              │
                                         │   Report: claims checked, how many supported          │
                                         │   My comment: _______________________________________ │
                                         └───────────────────────────────────────────────────────┘
```

## Behaviour notes

- Every button first checks the ledger against the cost cap. Over the cap: the
  button shows "Cap reached, raise the slider to continue" and does nothing.
- Editing the research question, prompt, strategy, or k clears that step and
  everything below it. Comments survive.
- If a reply does not parse as JSON, the table is skipped and the raw reply
  expander opens by default, with a yellow note: "Your prompt did not produce
  JSON. Fix the prompt or read the answer as text."
- The table shows whatever keys the model returned, in its order. Only `pmid`
  and `stance` matter downstream. Missing `stance`: Q4 uses all papers and says so.
- Download report writes one Markdown file: pool or query, every prompt as
  edited, every setting, every table, the review, the verification rows, the
  five comments, and the cost. That file is the homework submission.
