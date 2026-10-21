import json, re, sys, numpy as np, anthropic
from dotenv import load_dotenv; load_dotenv(".env")
from openai import OpenAI
from minsearch import Index
docs = json.load(open("pools/lung_ct/abstracts.json")); by = {d["pmid"]: d for d in docs}; pm = [d["pmid"] for d in docs]
V = np.load("pools/lung_ct/embeddings.npy"); oa = OpenAI(); cl = anthropic.Anthropic()
Q = "Who has validated an AI system for lung cancer screening in a real screening cohort?"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-haiku-5-5"
OUT = sys.argv[2] if len(sys.argv) > 2 else "reference/review_trace.json"
ANCHOR = "41308576"; LOG = {"question": Q, "model": MODEL, "steps": []}
def log(step, **kw): LOG["steps"].append({"step": step, **kw}); print(f"\n### {step}"); [print(f"  {k}: {v}" if not isinstance(v, list) else f"  {k}: {len(v)} items") for k, v in kw.items()]
def embed(t): v = np.array(oa.embeddings.create(model="text-embedding-3-small", input=[t]).data[0].embedding, dtype=np.float32); return v / np.linalg.norm(v)
def haiku(system, user, max_tokens=1500):
    r = cl.messages.create(model=MODEL, max_tokens=max_tokens, system=system, messages=[{"role":"user","content":user}], output_config={"effort":"low"})
    return "".join(b.text for b in r.content if b.type=="text").strip(), r.usage.input_tokens, r.usage.output_tokens
def ctx(pmids): return "\n\n".join(f"PMID {p} | {by[p]['first_author']} et al., {by[p]['year']} | {by[p]['journal']}\nTITLE: {by[p]['title']}\nABSTRACT: {by[p]['abstract']}" for p in pmids)
tok_in = tok_out = 0

# 1 embedding search with the raw question
qo = np.argsort(-(V @ embed(Q))); emb_top = [pm[j] for j in qo[:15]]
log("1. embedding search, raw question, k=15", top=emb_top, anchor_rank=[pm[j] for j in qo].index(ANCHOR)+1)

# 2 query rewrite -> hypothetical abstract -> embedding search
hyp, i, o = haiku("Write a 3-sentence abstract of a hypothetical primary study that would answer the user's question. Methods, cohort, metrics. No preamble.", Q, 300); tok_in += i; tok_out += o
ho = np.argsort(-(V @ embed(hyp))); hyp_top = [pm[j] for j in ho[:15]]
log("2. query rewrite (Haiku) then embedding search, k=15", hypothetical_abstract=hyp, top=hyp_top, anchor_rank=[pm[j] for j in ho].index(ANCHOR)+1)

# 3 keyword search
idx = Index(text_fields=["title","abstract"], keyword_fields=["pmid"]).fit(docs)
kw_top = [d["pmid"] for d in idx.search(Q, boost_dict={"title": 2.0}, num_results=15)]
log("3. keyword search (minsearch TF-IDF), k=15", top=kw_top, anchor_rank=(kw_top.index(ANCHOR)+1 if ANCHOR in kw_top else ">15"))

# 4 union of candidates
cands = list(dict.fromkeys(emb_top + hyp_top + kw_top))
log("4. union of candidates", candidates=cands, count=len(cands))

# 5 grade every candidate, 10 per call
GRADE = """You are grading papers for a literature review.
For EACH paper in the context, rate how well it answers the question:
HIGH = directly answers it (an AI system evaluated on scans from an actual screening program or trial, with performance results),
MEDIUM = partly relevant, LOW = not relevant.
Output exactly one line per paper, in this format and nothing else:
PMID <pmid> | <HIGH|MEDIUM|LOW> | <one sentence why>"""
grades = {}
for b in range(0, len(cands), 10):
    txt, i, o = haiku(GRADE, f"QUESTION: {Q}\n\nCONTEXT:\n{ctx(cands[b:b+10])}", 1200); tok_in += i; tok_out += o
    for line in txt.splitlines():
        m = re.match(r"PMID\s+(\d+)\s*\|\s*(HIGH|MEDIUM|LOW)\s*\|\s*(.*)", line.strip())
        if m: grades[m[1]] = {"grade": m[2], "why": m[3]}
high = [p for p in cands if grades.get(p, {}).get("grade") == "HIGH"]
med = [p for p in cands if grades.get(p, {}).get("grade") == "MEDIUM"]
log("5. grade candidates with Haiku, batches of 10", graded=len(grades), HIGH=high, MEDIUM=med, LOW=[p for p in cands if grades.get(p,{}).get("grade")=="LOW"])
for p in high: print(f"  HIGH {p} {by[p]['first_author']} {by[p]['year']}: {grades[p]['why'][:120]}")

# 6 synthesize from HIGH papers
SYN = """You are writing a short literature review paragraph set for a clinician audience.
Use only the abstracts in the context. Cite every claim with [PMID n]. Structure:
1) What has been validated and by whom (one bullet per study: cohort, size, AI task, headline result).
2) What recurs across studies (cohorts, AI tasks, comparison design).
3) Gaps: what no study in the context did.
Be concrete with numbers. Under 400 words."""
review, i, o = haiku(SYN, f"QUESTION: {Q}\n\nCONTEXT:\n{ctx(high)}", 4096); tok_in += i; tok_out += o
log("6. synthesize from HIGH papers only", papers=high, input_tokens=i, output_tokens=o)

# 7 verify citations
cited = sorted(set(re.findall(r"PMID\s*(\d{7,8})", review)))
log("7. verify citations", cited=cited, not_in_context=[c for c in cited if c not in high])
LOG["review"] = review; LOG["grades"] = grades; LOG["tokens"] = {"in": tok_in, "out": tok_out, "usd": (tok_in*0.1e-6 + tok_out*0.5e-6) if "haiku" in MODEL else (tok_in*3e-6 + tok_out*15e-6)}
json.dump(LOG, open(OUT,"w"), indent=1)
print("\n" + "="*70 + "\n" + review + "\n" + "="*70); print("total tokens:", LOG["tokens"])
