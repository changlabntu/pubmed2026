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
                                         │   │ You grade how well each abstract answers the    │ │
                                         │   │ question. Return a JSON array, one object per   │ │
                                         │   │ paper with keys: pmid, grade (HIGH|MEDIUM|LOW), │ │
                                         │   │ cohort, n, headline_result, evidence_quote, why │ │
                                         │   └─────────────────────────────────────────────────┘ │
                                         │   [ Grade ]                                           │
                                         │   ┌ pmid ─── grade ── cohort ── n ── headline ── ... ┐│
                                         │   │ 32055951  MEDIUM  OAI       1.2k  AUC 0.808  ... ││
                                         │   │ 40885690  LOW     ...                            ││
                                         │   │ 33337584  MEDIUM  ...                            ││
                                         │   └──────────────────────────────────────────────────┘│
                                         │   ▸ 32055951 MEDIUM: why and evidence                 │
                                         │   ▸ raw model reply                                   │
                                         │   Report: which PMID is HIGH                          │
                                         │   My comment: _______________________________________ │
                                         │                                                       │
                                         │ Q3. Find the anchor's neighbours                      │
                                         │   k [==|----] 10                                      │
                                         │   Embed: (•) title + abstract  ( ) title only         │
                                                                                  │   [ Find similar ]                                    │
                                         │   0.889  41234567 Walstra 2025 ...                    │
                                         │   ...                                                 │
                                         │   grades table for anchor + two nearest               │
                                         │   Report: did neighbours grade higher than Q2         │
                                         │   My comment: _______________________________________ │
                                         │                                                       │
                                         │ Q4. Search with the question                          │
                                         │   Strategy: (•) embedding ( ) keyword ( ) hybrid      │
                                         │   [ ] rewrite the question with Haiku first           │
                                         │   k [===|---] 10                                      │
                                                                                  │   [ Search ]                                          │
                                         │   Anchor rank: 2                                      │
                                         │   ┌ rank ─ sim ─── pmid ─── paper ─── grade ┐         │
                                         │   │ 1  0.61  41308576  Ledda 2026   HIGH    │         │
                                         │   │ ...                                     │         │
                                         │   └─────────────────────────────────────────┘         │
                                         │   Report: HIGH count, anchor rank                     │
                                         │   My comment: _______________________________________ │
                                         │                                                       │
                                         │ Q5. Summarize and verify                              │
                                         │   Include: [x] HIGH full abstract [x] MEDIUM [ ] LOW  │
                                                                                  │   ▸ Synthesis prompt            [ Reset ]             │
                                         │   ▸ Verifier prompt             [ Reset ]             │
                                         │   [ Summarize and verify ]                            │
                                         │   (review text with [PMID n] citations)               │
                                         │   18 claim checks: 11 supported, 7 partial, 0 unsup.  │
                                         │   ┌ verdict ─ pmid ─── claim ─── issue ┐              │
                                         │   │ partial  42383283  All HIGH ...    │              │
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
  and `grade` matter downstream. Missing `grade`: Q5 uses all papers and says so.
- Download report writes one Markdown file: pool or query, every prompt as
  edited, every setting, every table, the review, the verification rows, the
  five comments, and the cost. That file is the homework submission.
