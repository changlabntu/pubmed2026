# Module 1 Demo Plan: RAG Literature Review on AI for Early Lung Cancer Detection from CT

Scope: one working demo covering the RAG half of LLM Zoomcamp Module 1
(lessons 1-10). No agents, no vector search, no later modules.
Teaching notes live inside the demo notebook as markdown cells.

## Goal

Students run a notebook top to bottom and end with a working RAG system
that answers literature-review questions over ~280 PubMed abstracts,
citing PMIDs. The homework repeats the same flow on the frozen corpus
with five multiple-choice questions.

## Corpus

PubMed query (278 hits on 2026-10-12):

    "lung cancer" AND ("deep learning" OR "artificial intelligence")
    AND ("low-dose CT" OR "low-dose computed tomography") AND screening

- Fetch once, freeze the PMID list in `data/pmids.txt`, and ship the
  parsed result as `data/abstracts.json`. Students re-fetch by PMID so
  everyone has identical data and homework answers stay fixed.
- E-utilities: `esearch` (PMIDs) then `efetch` in XML, 200 IDs per call,
  with `tool` and `email` parameters and an NCBI API key in `.env`.
- Document schema:

      pmid, title, abstract, authors (list), first_author, journal,
      year, mesh (list), keywords (list), doi

- Cleaning: join labeled abstract sections, drop records with no
  abstract, de-duplicate by PMID.

## Model

Claude Haiku 5.5 (`claude-haiku-5-5`) through the Anthropic Python SDK.
Key in `.env` as `ANTHROPIC_API_KEY`. Expected spend per student: cents.

## Deliverables

```
module1-rag/
  README.md                  how to set up and run
  pyproject.toml             uv project: anthropic, minsearch,
                             sqlitesearch, requests, python-dotenv
  .env.example
  pubmed_ingest.py           search_pmids, fetch_abstracts,
                             parse_pubmed_xml, load_corpus, build_index
  rag_helper.py              RAGBase with PubMed context builder and
                             Claude llm() that returns text + usage
  demo.ipynb                 the teaching demo (sections below)
  data/pmids.txt             frozen PMID list
  data/abstracts.json        frozen parsed corpus
  homework/homework.md       five questions
  homework/solution.ipynb    instructor-only, verifies the answer keys
```

## demo.ipynb sections

Each section: one markdown cell (concept, why this step exists, what
to notice) followed by a few code cells.

1. **The problem.** Ask Claude "What deep learning approaches are
   used for lung nodule detection on low-dose CT?" with no context.
   Generic answer. Teaches: the model does not know this corpus.

2. **Get the corpus.** `esearch` to show the real workflow, then
   `efetch` by the frozen PMID list. Parse XML, inspect one record,
   count missing abstracts, save JSON. Teaches: ingestion, messy real
   data, reproducibility.

3. **Search.** minsearch index with `title` and `abstract` as text
   fields, `pmid`, `year`, `journal` as keyword fields. Boost `title`.
   Run a query, print top 5 with PMID, year, first author, title.
   Filter by year range. Teaches: similarity scoring, text vs keyword
   fields, boosting, filtering.

4. **Prompt.** Instructions: answer only from context, cite PMIDs in
   brackets after each claim, say "not covered in the retrieved
   abstracts" otherwise. Context block per hit: PMID, year, first
   author, journal, title, abstract. Teaches: system vs user prompt,
   grounding, provenance.

5. **LLM call.** Send the prompt with `client.messages.create`, print
   the answer, read `usage.input_tokens` / `output_tokens`, compute
   cost from Haiku pricing. Teaches: the API surface, token budgets.

6. **Wrap it in RAGBase.** Import from `rag_helper.py`, run the three
   review questions and discuss which plain RAG handles:
   - "Who has validated AI nodule detection in a screening cohort?"
     (fits: retrieval + citation)
   - "What is the most common dataset used?" (does not fit: top-5
     cannot aggregate; name the limitation, point to later modules)
   - "What is the overall clinical implication?" (partially fits)
   Teaches: the honest limits of top-k retrieval.

7. **Citation check.** Extract PMIDs cited in the answer, verify each
   exists in the retrieved context. Teaches: hallucinated citations
   and how to catch them programmatically.

8. **Persist.** Write the index with sqlitesearch, reload in a new
   cell, query again. Teaches: ingestion vs querying, why in-memory
   breaks on restart.

## rag_helper.py

Same `RAGBase` structure as the course:

- `search(query, num_results=5, year_from=None, year_to=None)`
- `build_context(results)` using the PubMed fields
- `build_prompt(query, results)`
- `llm(prompt)` returning `(text, usage)` via the Anthropic SDK
- `rag(query)` returning `(answer, usage)`

## homework.md (five questions, order-of-magnitude options)

1. Load `data/pmids.txt`, fetch and parse. How many documents have a
   non-empty abstract?
2. Index as in the demo. For the fixed query, what is the PMID of the
   first result?
3. Run RAG on that query. How many input tokens were sent?
4. Restrict to 2022-2024. How many documents remain, and is the top
   PMID the same?
5. Put every abstract into one prompt and ask the "overall
   implication" question. Roughly how many times more input tokens
   than Q3?

Answer keys come from `solution.ipynb`. Q3 and Q5 are read from
`usage.input_tokens`; options are spaced 10x apart so provider or
model drift does not change the answer.

## Build order

1. `pubmed_ingest.py` and the frozen corpus (everything depends on it)
2. `rag_helper.py`
3. `demo.ipynb`
4. `homework.md` and `solution.ipynb`
5. `README.md`

## Out of scope

Agents and function calling, chunking, vector search, evaluation,
monitoring, Kestra. Each can become its own demo later using the same
corpus.
