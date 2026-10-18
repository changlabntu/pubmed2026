# Module 1 v2: a literature review in five steps

A Streamlit app that walks through a small literature review over a pool of
PubMed abstracts. Embeddings from OpenAI, grading and writing by Claude Haiku 5.5.
Two pools ship with it, and a dropdown switches between them:

| Pool | Question | Abstracts | Anchor |
|---|---|---|---|
| `lung_ct` | Who has validated an AI system for lung cancer screening in a real screening cohort? | 181 | Ledda 2026, PMID 41308576 |
| `knee_mri` | Can AI detect or predict pain symptoms from knee MRI images? | 119 | Chang 2020, PMID 32055951 |

Version 1 (`../module1-rag/`) taught keyword search and a single RAG call.
This version replaces keyword search with embeddings, makes the model grade
papers with evidence before it writes, and checks every sentence of the
review against the abstract it cites.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # paste ANTHROPIC_API_KEY and OPENAI_API_KEY
streamlit run app.py
```

On Streamlit Community Cloud, put the same two keys in the app's Secrets
panel instead of `.env`.

## The five steps

| Step | Button | What happens | Report |
|---|---|---|---|
| Q1 | Build corpus | Fetch the frozen PMIDs, keep records with an abstract, embed title plus abstract. Cached in `pools/`. | document count |
| Q2 | Grade | Haiku grades the anchor paper and two random companions HIGH / MEDIUM / LOW, with an evidence quote each. | which PMID is HIGH |
| Q3 | Find similar | Cosine similarity from the anchor to every paper. The two nearest replace the random pair and are graded. | did the neighbours grade higher |
| Q4 | Search | Embed the question itself, take its ten nearest, grade all ten. | HIGH count, anchor rank |
| Q5 | Summarize and verify | Haiku writes a short review from the HIGH and MEDIUM papers, then every cited sentence is checked against its abstract. | claims checked, claims supported |

Each pool's anchor and companions are fixed in `pools/<name>/papers.json` so
every student sees the same Q2. The companions are `random.Random(42).sample(...)`.

## Adding a pool

```bash
python build_pool.py my_pool '<PubMed query>'          # fetch, filter, embed
python pick_anchor.py my_pool '<the review question>'  # grades the top 10, suggests an anchor
```

Then write `pools/my_pool/papers.json` with the question, anchor, and the
companions the second script printed. Reviews, editorials, letters and
meta-analyses are excluded from every pool automatically.

## Files

| File | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `lit_helper.py` | build_corpus, nearest, search, grade_relevance, synthesize, verify_claims |
| `pubmed_ingest.py` | PubMed E-utilities fetch and parse |
| `build_pool.py`, `pick_anchor.py` | instructor tools for making a new pool |
| `pools/<name>/query.txt` | the PubMed query |
| `pools/<name>/pmids.txt` | frozen PMIDs |
| `pools/<name>/abstracts.json` | parsed records |
| `pools/<name>/embeddings.npy` | vectors, text-embedding-3-small |
| `pools/<name>/papers.json` | question, anchor, companions |
| `homework/homework.md` | what students report |
| `review_pipeline.py`, `REVIEW-TEMPLATE.md` | a scripted top-k review of the lung pool and its trace, instructor reference |
| `fable_literature_pipeline.md`, `reference/independent_review.md` | an agent's full-corpus review of the lung pool and its trace, instructor reference |

## The pools

Both built 2026-10-18. The queries are in each pool's `query.txt`; this
exclusion is appended to every query:

```
NOT (Review[pt] OR "Systematic Review"[pt] OR Meta-Analysis[pt]
     OR Editorial[pt] OR Comment[pt] OR Letter[pt] OR News[pt])
```

## Cost

One full run is about 1.3 cents, most of it in Q5's verification, which makes
one Haiku call per cited sentence. Thirty students with a few reruns each is
under a dollar. The sidebar shows the running total.

## Two things to know

- Haiku 5.5 emits a thinking block before its text even at low effort. Every
  call uses `max_tokens` of 2048 or more, or the answer comes back empty.
- The verifier checks one sentence against one abstract. Sentences that
  summarize several papers ("most studies were retrospective") often come
  back `partial`, because no single abstract can confirm a generalization.
  That is a limitation of the method, not necessarily an error in the review.
