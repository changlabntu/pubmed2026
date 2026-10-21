"""
Instructor answer key. Run from the module1-rag folder:

    .venv/bin/python homework/answer_key.py            # all questions (needs ANTHROPIC_API_KEY)
    .venv/bin/python homework/answer_key.py --no-llm   # Q1, Q2, Q4 only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from pubmed_ingest import read_pmids, load_corpus, build_index
from rag_helper import RAGBase, cited_pmids

NO_LLM = "--no-llm" in sys.argv
QUERY = "Who has validated an AI system for lung cancer screening in a real screening cohort?"

pmids = read_pmids("data/pmids.txt")
documents = load_corpus(pmids)
print(f"Q1: {len(documents)} documents")

index = build_index(documents)
rag = RAGBase(index, llm_client=None if not NO_LLM else object())
top = rag.search(QUERY)[0]
print(f"Q2: first result PMID {top['pmid']} ({top['first_author']} {top['year']})")

rag_recent = RAGBase(index, llm_client=rag.llm_client, year_from=2022, year_to=2024)
top_recent = rag_recent.search(QUERY)[0]
n_recent = sum(1 for d in documents if d["year"] and 2022 <= d["year"] <= 2024)
same = "same" if top_recent["pmid"] == top["pmid"] else "different"
print(f"Q4: {n_recent} documents in 2022-2024, first result {top_recent['pmid']} ({same} PMID as Q2)")

if NO_LLM:
    print("Q3, Q5: skipped (--no-llm)")
    sys.exit()

r = rag.rag(QUERY)
print(f"Q3: {r.input_tokens} input tokens ({r.output_tokens} out, ${r.cost_usd:.5f})")
shown = {d["pmid"] for d in r.search_results}
bad = [p for p in cited_pmids(r.answer) if p not in shown]
print(f"    bonus: cited {cited_pmids(r.answer)}, not in context: {bad}")

prompt = rag.build_prompt(
    "What is the overall clinical implication of this literature for lung cancer screening programs?",
    documents,
)
answer, usage = rag.llm(prompt)
print(f"Q5: {usage.input_tokens} input tokens, {usage.input_tokens / r.input_tokens:.0f}x Q3")
