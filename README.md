# pubmed2026

Course material for a beginner LLM module: a mini literature review over
PubMed abstracts that argues for or against a research gap, built as a
Streamlit app that students run and report on.

| Folder | What it is |
|---|---|
| `app/` | the current app (v4): claim, three starting papers, evidence, argument with verdict. Streamlit Community Cloud main file: `app/app.py` |
| `archive/module1-rag-v1/` | v1: keyword search with minsearch and one RAG call, Jupyter notebook |
| `archive/module1-rag-v2/` | v2: embeddings, fixed anchor + companions, grading, verification |
| `archive/module1-v3/` | v3: student-editable prompts, search strategies, HIGH/MEDIUM/LOW then thesis/antithesis grading, gap verdict as its own step |

Keys needed: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. See `app/README.md`.
