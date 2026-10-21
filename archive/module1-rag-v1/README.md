# Module 1: RAG for a literature review

A working demo of Retrieval-Augmented Generation over ~270 PubMed abstracts
on **AI for early lung cancer detection from low-dose CT**, using Claude
Haiku 5.5. Adapted from Module 1 of the
[LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

No GPU, no Docker. A laptop, Python 3.10+, and an API key.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then paste your ANTHROPIC_API_KEY
jupyter notebook demo.ipynb
```

An NCBI API key is optional. Without one PubMed allows 3 requests per
second, which is plenty here.

## Files

| File | What it is |
|---|---|
| `demo.ipynb` | The teaching demo. Explanations in markdown, code below each. Run top to bottom. |
| `pubmed_ingest.py` | Fetch and parse PubMed records (E-utilities), build the search index |
| `rag_helper.py` | `RAGBase`: search, prompt, LLM call, citation check |
| `data/pmids.txt` | Frozen list of 278 PMIDs so everyone uses identical data |
| `data/abstracts.json` | The parsed corpus (272 records with abstracts), for reference |
| `homework/homework.md` | Five questions |
| `homework/answer_key.py` | Instructor only |

## The corpus

PubMed query, run 2026-10-12:

```
"lung cancer" AND ("deep learning" OR "artificial intelligence")
AND ("low-dose CT" OR "low-dose computed tomography") AND screening
```

To rebuild the frozen files from a fresh search run
`python pubmed_ingest.py`. Note that answers in the homework are tied to the
current `data/pmids.txt`.

## Cost

Haiku 5.5 is $0.10 per million input tokens and $0.50 per million output.
Running the demo and homework once costs a few cents per student. The one
exception is homework Q5, which sends the whole corpus in a single prompt,
about 200K tokens. Prompts over 100K tokens are billed at the long-context
rate ($0.50 per million input), so that call is about 10 cents.
