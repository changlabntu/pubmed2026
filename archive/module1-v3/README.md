# Module 1 v3: build your own literature review

A Streamlit app that walks through a small literature review over PubMed
abstracts. Embeddings from OpenAI `text-embedding-3-small`; grading, writing
and fact-checking by Claude Haiku 5.5 (Sonnet 5.5 optional). This version
exposes the knobs: the pool, the question, the anchor paper, every system
prompt, the search strategy and the model. The homework is the report the
app writes for you.

Two pools ship with it:

| Pool | Default question | Abstracts | Anchor |
|---|---|---|---|
| `lung_ct` | Who has validated an AI system for lung cancer screening in a real screening cohort? | 181 | Ledda 2026, PMID 41308576 |
| `knee_mri` | Can AI detect or predict pain symptoms from knee MRI images? | 119 | Chang 2020, PMID 32055951 |

Or type your own PubMed query in Q1.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # paste ANTHROPIC_API_KEY and OPENAI_API_KEY
streamlit run app.py
```

On Streamlit Community Cloud the main file is `app/app.py`; put the same two
keys in the app's Secrets panel instead of `.env`.

## The five steps

| Step | What the student controls | What happens | Report |
|---|---|---|---|
| Q1 Build corpus | own PubMed query + max records (default; the most relevant records are kept, not the newest), or a shipped pool; reviews and editorials always excluded; three PMIDs for the anchor and two companions, added to the pool and drawn red on the map | fetch abstracts, embed title + abstract, cache under `pools/`; t-SNE map of the pool, one dot per paper sized by iCite citation count (capped at 100), hover for the title | document count |
| Q2 Grade | **grading prompt** (three boxes) | the model scores the anchor + two companions named in Q1 (or a chosen anchor + two seed-42 companions if none were given) from 1 to 10 for what each says about the gap, 1 = does the work or shows the premise wrong, 10 = names the gap as undone, and says whether the paper bears on the question at all; code derives the stance (**antithesis** at 5 or below, **thesis** above, **neutral** when it does not bear), the gap relation from the condition checks, and a flag where score and checks disagree; the table shows whatever keys the prompt asks for | which paper scores lowest and why |
| Q3 Search the corpus | strategy: **embedding** (the question's vector retrieves) or **anchor's neighbours** (cosine from the anchor) | the top 10 are scored in one call | stance counts, anchor rank (embedding) or the nearest papers (neighbours) |
| Q4 Synthesis | **synthesis prompt**, **verifier prompt** | the model argues the gap as a dialectic with cited THESIS, ANTITHESIS and SYNTHESIS sections from the thesis and antithesis papers, neutral ones left out (antithesis papers go in with their full abstract); each cited sentence is checked against its abstract | claims checked, claims supported, whether the synthesis narrowed or conceded the gap |

Under every step there is a comment box. The sidebar has the model switch
(Haiku; Sonnet shown greyed out, enable with `ALLOW_SONNET` in `app.py`), a cost cap slider that stops
the buttons when the session has spent that much, the running total, and
**Download report (.md)**: every setting, prompt, table, the review, the
verification rows, the comments and the cost in one file.

## What happens when a prompt breaks

If the grading prompt no longer produces a JSON array, the app shows the raw
reply with a yellow note instead of a table. Q4 waits for a Q3 run that did
return JSON. If the records carry neither `score` nor `bears`, Q4 uses every Q3 paper and
says so. Editing a prompt, the question, or a search setting clears that step
and the steps after it; comments survive.

## Adding a shipped pool

```bash
python build_pool.py my_pool '<PubMed query>'          # fetch, filter, embed
python pick_anchor.py my_pool '<the review question>' '<gap>' '<disproof>'  # scores the top 10, suggests an anchor
```

Then write `pools/my_pool/papers.json` with the question, anchor and
companions. Student-built pools are cached under `pools/custom/` and ignored
by git.

## Files

| File | What it is |
|---|---|
| `app.py` | the Streamlit app |
| `lit_helper.py` | corpus, retrieval (question embedding, anchor neighbours), stance grading, dialectic synthesis, verification, gap verdict |
| `pubmed_ingest.py` | PubMed E-utilities fetch and parse; citation counts come from NIH iCite in `lit_helper.fetch_citations` |
| `build_pool.py`, `pick_anchor.py` | instructor tools |
| `pools/<name>/` | query, PMIDs, meta.json (match count, fetch date), abstracts, vectors, papers.json (question, gap, anchor) |
| `homework/homework.md` | what to submit |
| `UI-MOCKUP.md` | the page layout as a text sketch |

## Cost

A run is a few cents on Haiku, about 30 model calls, most of them Q4's
one-call-per-sentence verification. The default cap is 10 cents.

## The gap loop

Q0 asks for the research question, the proposed gap (what nobody has done)
and what evidence would disprove it. All three go into every grading and
synthesis call as context, never into the editable prompts. Q1 reports the
retrieval funnel (PubMed matches, fetched, with abstract, fetch date) so
"not retrieved" is never mistaken for "no study exists", and only the 10
papers the search returned are ever graded. Every paper is scored 1 to 10 for
what it says about the gap, low against and high for, with a separate yes or
no for whether it bears on the question. The stance is computed from those two
fields (`STANCE_CUTOFF` in `lit_helper.py`): neutral when it does not bear,
antithesis at 5 or below, thesis above. The gap relation (counterexample,
partial, no, unclear) is computed from the model's per-condition checks, and a
flag marks any paper where the score and the checks disagree, so a reader
trusts the quotes over the number. Q4 argues the gap as a dialectic: thesis,
antithesis, synthesis. Q5 ends the run
with a verdict on the gap: contradicted, partly addressed, provisionally
supported, or insufficient evidence, with the closest counterexamples.

## Two things to know

- Haiku 5.5 emits a thinking block before its text even at low effort. Every
  call uses `max_tokens` of 2048 or more, or the answer comes back empty.
- The verifier checks one sentence against one abstract. Sentences that
  summarize several papers often come back `partial`, because no single
  abstract can confirm a generalization. That is a limitation of the method,
  not necessarily an error in the review.
