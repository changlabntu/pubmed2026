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
| Q1 Build corpus | own PubMed query + max records (default), or a shipped pool; reviews and editorials always excluded | fetch abstracts, embed title + abstract, cache under `pools/`; t-SNE map of the pool, one dot per paper, hover for the title | document count |
| Q2 Grade | research question, anchor paper, **grading prompt** (full text) | the model grades anchor + two seed-42 companions; the table shows whatever keys the prompt asks for | highest-graded PMID and why |
| Q3 Find similar | k, embed title+abstract or title only | cosine from the anchor; two nearest replace the companions and are graded | did the neighbours grade higher |
| Q4 Search | embedding / keyword (BM25) / hybrid (rank fusion), rewrite toggle, k | the question retrieves; every hit is graded | HIGH count, anchor rank |
| Q5 Summarize and verify | grades to include, full-abstract toggle, **synthesis prompt**, **verifier prompt** | the model writes a cited review; each cited sentence is checked against its abstract | claims checked, claims supported |

Under every step there is a comment box. The sidebar has the model switch
(Haiku; Sonnet shown greyed out, enable with `ALLOW_SONNET` in `app.py`), a cost cap slider that stops
the buttons when the session has spent that much, the running total, and
**Download report (.md)**: every setting, prompt, table, the review, the
verification rows, the comments and the cost in one file.

## What happens when a prompt breaks

If the grading prompt no longer produces a JSON array, the app shows the raw
reply with a yellow note instead of a table. Q5 waits for a Q4 run that did
return JSON. If the records carry no `grade` key, Q5 uses every Q4 paper and
says so. Editing a prompt, the question, or a search setting clears that step
and the steps after it; comments survive.

## Adding a shipped pool

```bash
python build_pool.py my_pool '<PubMed query>'          # fetch, filter, embed
python pick_anchor.py my_pool '<the review question>'  # grades the top 10, suggests an anchor
```

Then write `pools/my_pool/papers.json` with the question, anchor and
companions. Student-built pools are cached under `pools/custom/` and ignored
by git.

## Files

| File | What it is |
|---|---|
| `app.py` | the Streamlit app |
| `lit_helper.py` | corpus, search (embedding / BM25 / hybrid), grading, synthesis, verification |
| `pubmed_ingest.py` | PubMed E-utilities fetch and parse |
| `build_pool.py`, `pick_anchor.py` | instructor tools |
| `pools/<name>/` | query, PMIDs, abstracts, vectors, papers.json |
| `homework/homework.md` | what to submit |
| `UI-MOCKUP.md` | the page layout as a text sketch |

## Cost

A default run is about 1.1 cents on Haiku, about 30 model calls, most of them
Q5's one-call-per-sentence verification. The default cap is 5 cents.

## Two things to know

- Haiku 5.5 emits a thinking block before its text even at low effort. Every
  call uses `max_tokens` of 2048 or more, or the answer comes back empty.
- The verifier checks one sentence against one abstract. Sentences that
  summarize several papers often come back `partial`, because no single
  abstract can confirm a generalization. That is a limitation of the method,
  not necessarily an error in the review.
