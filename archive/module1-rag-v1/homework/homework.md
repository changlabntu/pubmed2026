# Homework 1: RAG over PubMed abstracts

Same pipeline as `demo.ipynb`, same frozen corpus. Run the steps and report
what you see. If your number does not match an option exactly, pick the
closest one.

Model: `claude-haiku-5-5`. Put your key in `.env`.

## Setup

```python
from dotenv import load_dotenv
load_dotenv()

from pubmed_ingest import read_pmids, load_corpus, build_index
from rag_helper import RAGBase
```

## Q1. Loading the corpus

Read `data/pmids.txt` and fetch the records with `load_corpus(pmids)`. It
drops records that have no abstract and de-duplicates by PMID.

How many documents do you get?

* 27
* 272
* 2,720
* 27,200

## Q2. Searching

Build the index with `build_index(documents)` and search with this query,
boosting the title by 2.0 and returning 5 results:

> Who has validated an AI system for lung cancer screening in a real screening cohort?

What is the PMID of the first result?

* 33240975
* 34164287
* 36563698
* 39165498

## Q3. RAG

Create `RAGBase(index)` and run `rag.rag(query)` with the same query. The
result has `input_tokens` from the API response.

How many input tokens were sent to the model?

* 500
* 5,000
* 50,000
* 500,000

We count input tokens instead of dollars because the price depends on the
model, but the prompt we send is the same for everyone.

## Q4. Filtering by year

Create `RAGBase(index, year_from=2022, year_to=2024)` and search again with
the same query. Also count how many documents in the corpus have a year
between 2022 and 2024 inclusive.

How many documents are in that range, and is the first result the same PMID
as in Q2?

* about 80, same PMID
* about 80, different PMID
* about 200, same PMID
* about 200, different PMID

## Q5. Everything in the context

The whole corpus fits in Claude's context window. Instead of retrieving 5
abstracts, put all of them in:

```python
rag_all = RAGBase(index)
prompt = rag_all.build_prompt(
    "What is the overall clinical implication of this literature for lung cancer screening programs?",
    documents,          # every document, not a search result
)
answer, usage = rag_all.llm(prompt)
```

Compare `usage.input_tokens` with your Q3 number. Roughly how many times
more input tokens does the all-in-context version use?

* about the same
* 5x more
* 50x more
* 500x more

This one call costs about 10 cents, roughly 100 times the Q3 call, because
prompts over 100K tokens are billed at a higher rate.

The corpus fits, and the answer is probably good. So why retrieve at all?
Compare the cost of the two calls and think about what happens with 27,200
documents instead of 272.

## Bonus (ungraded): check the citations

Run `rag.rag(query)` from Q3 and pass the result to `cited_pmids` from
`rag_helper`. Verify every cited PMID is among `result.search_results`. Any
PMID that is not was invented by the model.

## Submit

Submit your five answers in the course form.
