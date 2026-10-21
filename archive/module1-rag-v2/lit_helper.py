"""
Helpers for the Module 1 v2 literature-review app.

Pipeline pieces, in the order the app uses them:
  build_corpus      fetch abstracts from PubMed and embed them (cached on disk)
  nearest           papers most similar to one paper (vector to vectors)
  search            papers most similar to a question (text to vectors)
  grade_relevance   Haiku grades each paper HIGH / MEDIUM / LOW with evidence
  synthesize        Haiku writes a short review from the graded papers
  verify_claims     Haiku checks every cited sentence against its abstract

Every model call returns a Usage so the app can show running cost.
"""

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from pubmed_ingest import load_corpus, read_pmids

# ---------------------------------------------------------------- models

CHAT_MODEL = "claude-haiku-5-5"
EMBED_MODEL = "text-embedding-3-small"

# USD per million tokens
PRICE = {
    CHAT_MODEL: {"input": 0.10, "output": 0.50},
    EMBED_MODEL: {"input": 0.02, "output": 0.0},
}

HERE = os.path.dirname(os.path.abspath(__file__))
POOLS_DIR = os.path.join(HERE, "pools")   # absolute, so the app works from any cwd
DEFAULT_POOL = "lung_ct"


def pool_path(pool, name):
    return os.path.join(POOLS_DIR, pool, name)


def list_pools():
    """Pools are subfolders of pools/ that contain a papers.json."""
    if not os.path.isdir(POOLS_DIR):
        return []
    return sorted(d for d in os.listdir(POOLS_DIR) if os.path.exists(pool_path(d, "papers.json")))


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def cost_usd(self):
        p = PRICE[self.model]
        return (self.input_tokens * p["input"] + self.output_tokens * p["output"]) / 1e6


@dataclass
class Ledger:
    """Running total of every call made in a session."""
    entries: list = field(default_factory=list)

    def add(self, step, usage):
        self.entries.append((step, usage))

    @property
    def cost_usd(self):
        return sum(u.cost_usd for _, u in self.entries)

    @property
    def calls(self):
        return sum(u.calls for _, u in self.entries)


# ---------------------------------------------------------------- clients

def anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def openai_client():
    from openai import OpenAI
    return OpenAI()


# ---------------------------------------------------------------- Q1 corpus

def embed_texts(texts, client=None, batch_size=100):
    """Embed a list of strings. Returns (unit-normalised array, Usage)."""
    client = client or openai_client()
    vectors, usage = [], Usage(EMBED_MODEL)
    for i in range(0, len(texts), batch_size):
        r = client.embeddings.create(model=EMBED_MODEL, input=texts[i:i + batch_size])
        vectors.extend(e.embedding for e in r.data)
        usage.input_tokens += r.usage.total_tokens
        usage.calls += 1
    V = np.asarray(vectors, dtype=np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V, usage


def build_corpus(pool=DEFAULT_POOL, force=False, progress=None):
    """Fetch abstracts for a pool and embed them. Uses the cached files when
    present. Returns (documents, vectors, usage_or_None, source) where source
    is "cache" or "network".
    """
    abstracts, embeddings = pool_path(pool, "abstracts.json"), pool_path(pool, "embeddings.npy")
    if not force and os.path.exists(abstracts) and os.path.exists(embeddings):
        with open(abstracts) as f:
            documents = json.load(f)
        vectors = np.load(embeddings)
        if len(documents) == vectors.shape[0]:
            return documents, vectors, None, "cache"

    if progress:
        progress("Fetching abstracts from PubMed ...")
    documents = load_corpus(read_pmids(pool_path(pool, "pmids.txt")))
    if progress:
        progress(f"Embedding {len(documents)} abstracts ...")
    vectors, usage = embed_texts([d["title"] + "\n" + d["abstract"] for d in documents])
    with open(abstracts, "w") as f:
        json.dump(documents, f, indent=1, ensure_ascii=False)
    np.save(embeddings, vectors)
    return documents, vectors, usage, "network"


def load_papers(pool=DEFAULT_POOL):
    """The question, anchor and companion PMIDs for a pool."""
    with open(pool_path(pool, "papers.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- Q3 / Q4 similarity

def nearest(anchor_pmid, documents, vectors, k=10):
    """Top-k papers by cosine similarity to one paper, excluding itself."""
    pmids = [d["pmid"] for d in documents]
    i = pmids.index(anchor_pmid)
    sims = vectors @ vectors[i]
    order = [j for j in np.argsort(-sims) if j != i][:k]
    return [(documents[j], float(sims[j])) for j in order]


def search(question, documents, vectors, k=10, client=None):
    """Top-k papers by cosine similarity to a question. Returns (hits, Usage)."""
    q, usage = embed_texts([question], client=client)
    sims = vectors @ q[0]
    order = np.argsort(-sims)[:k]
    return [(documents[j], float(sims[j])) for j in order], usage


def rank_of(pmid, hits):
    for r, (d, _) in enumerate(hits, 1):
        if d["pmid"] == pmid:
            return r
    return None


# ---------------------------------------------------------------- prompts

def context_block(docs):
    return "\n\n".join(
        f"PMID {d['pmid']} | {d['first_author']} et al., {d['year']} | {d['journal']}\n"
        f"TITLE: {d['title']}\nABSTRACT: {d['abstract']}"
        for d in docs
    )


GRADE_INSTRUCTIONS = """You are screening papers for a literature review.

For EACH paper in the context, judge how well it answers the question.
  HIGH   = directly answers it: an AI system was evaluated on scans from an
           actual screening program or trial, with performance results.
  MEDIUM = partly relevant: related task, unclear or mixed cohort, or no results.
  LOW    = not relevant.

Return ONLY a JSON array, one object per paper, in the order given, with keys:
  pmid            string
  grade           "HIGH" | "MEDIUM" | "LOW"
  cohort          the screening cohort or data source named in the abstract, or null
  n               number of participants or scans in the evaluation, as written, or null
  headline_result the key quantitative result with numbers exactly as written, or null
  evidence_quote  one verbatim sentence from the abstract that justifies the grade
  why             one sentence
No prose before or after the JSON."""


SYNTHESIS_INSTRUCTIONS = """You are writing a short literature review for a clinician audience.
Use only the material in the context. Cite every claim as [PMID n]; if a sentence
rests on several papers, cite each: [PMID n; PMID m].

Structure, under 350 words:
1) Who validated what: one bullet per HIGH paper with cohort, size, AI task, headline result.
2) What recurs across the studies: cohorts, AI tasks, comparison design.
3) Gaps: what no paper in the context did.
Be concrete with numbers. No preamble."""


VERIFY_INSTRUCTIONS = """You are a strict fact-checker. You get one claim and ONE of the abstracts it cites.
Judge ONLY from the abstract. Every number and named cohort that the claim attributes
to this paper must appear in, or be directly derivable from, the abstract.

A claim may summarize several papers ("most studies ...", "cohorts ranged from ..."). Then
judge only the part that concerns this paper: supported if the abstract is consistent with
its share of the claim, partial if some detail is missing or loosely stated, unsupported
only if the abstract contradicts the claim or contains nothing relevant to it.

Return ONLY a JSON object:
{"verdict": "supported" | "partial" | "unsupported",
 "quote": "verbatim abstract text that supports or contradicts the claim",
 "issue": "what the abstract does not state, or null"}"""


def _chat(client, system, user, max_tokens=4096):
    """One Haiku call. max_tokens must stay large: the model emits a thinking
    block before its text, and a small budget returns an empty answer."""
    r = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low"},
    )
    text = "".join(b.text for b in r.content if b.type == "text").strip()
    return text, Usage(CHAT_MODEL, r.usage.input_tokens, r.usage.output_tokens, calls=1)


def _parse_json(text):
    """Haiku occasionally wraps JSON in a code fence. Strip it and parse."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


# ---------------------------------------------------------------- Q2 / Q3 / Q4 grading

def grade_relevance(question, docs, client=None):
    """Grade each document. Returns (records, Usage). A record is the JSON
    object from the model plus the document itself under "doc"."""
    client = client or anthropic_client()
    user = f"QUESTION: {question}\n\nCONTEXT:\n{context_block(docs)}"
    text, usage = _chat(client, GRADE_INSTRUCTIONS, user)
    try:
        items = _parse_json(text)
    except json.JSONDecodeError:
        text2, usage2 = _chat(client, GRADE_INSTRUCTIONS, user)   # one retry
        usage.input_tokens += usage2.input_tokens
        usage.output_tokens += usage2.output_tokens
        usage.calls += usage2.calls
        items = _parse_json(text2)
    by_pmid = {d["pmid"]: d for d in docs}
    records = []
    for item in items:
        pmid = str(item.get("pmid", "")).strip()
        if pmid in by_pmid:
            item["doc"] = by_pmid[pmid]
            item["quote_found"] = quote_in_abstract(item.get("evidence_quote"), by_pmid[pmid]["abstract"])
            records.append(item)
    return records, usage


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def quote_in_abstract(quote, abstract, min_words=6):
    """Fuzzy check: is the quote (or a 6-word window of it) in the abstract?
    Haiku sometimes trims or lightly rewords a sentence."""
    q, a = _norm(quote), _norm(abstract)
    if not q:
        return False
    if q in a:
        return True
    words = q.split()
    for i in range(0, max(1, len(words) - min_words + 1)):
        if " ".join(words[i:i + min_words]) in a:
            return True
    return False


# ---------------------------------------------------------------- Q5 synthesis

def record_block(records, attach_full_high=True):
    """Context for the synthesis: HIGH papers in full (optional), MEDIUM as
    the grader's structured record only."""
    parts = []
    for r in records:
        d = r["doc"]
        head = f"PMID {d['pmid']} | {d['first_author']} et al., {d['year']} | grade {r['grade']}"
        if r["grade"] == "HIGH" and attach_full_high:
            parts.append(f"{head}\nTITLE: {d['title']}\nABSTRACT: {d['abstract']}")
        else:
            parts.append(
                f"{head}\nTITLE: {d['title']}\nCOHORT: {r.get('cohort')}\nN: {r.get('n')}\n"
                f"HEADLINE RESULT: {r.get('headline_result')}\nEVIDENCE: {r.get('evidence_quote')}"
            )
    return "\n\n".join(parts)


def synthesize(question, records, attach_full_high=True, client=None):
    """Write the review from HIGH and MEDIUM records. Returns (text, Usage)."""
    client = client or anthropic_client()
    kept = [r for r in records if r["grade"] in ("HIGH", "MEDIUM")]
    user = f"QUESTION: {question}\n\nCONTEXT:\n{record_block(kept, attach_full_high)}"
    return _chat(client, SYNTHESIS_INSTRUCTIONS, user)


# ---------------------------------------------------------------- Q5 verification

CITE_RE = re.compile(r"\[(PMID[^\]]*)\]")
PMID_RE = re.compile(r"\d{7,8}")


def split_claims(review):
    """Break the review into (claim_text, [pmids]) pairs. A claim is the text
    on a line up to and including one citation bracket."""
    claims = []
    for line in review.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        pos, carry = 0, ""
        for m in CITE_RE.finditer(line):
            text = line[pos:m.start()].replace("**", "").strip(" .;:")
            pmids = PMID_RE.findall(m.group(1))
            pos = m.end()
            if carry:
                text = (carry + " " + text).strip()
                carry = ""
            if not text or not pmids:
                continue
            if len(text.split()) < 5:
                # a bare name or label like "Chang et al., 2020" cited on its
                # own: not a checkable claim, fold it into the next clause
                carry = text
                continue
            claims.append((text, pmids))
        tail = line[pos:].replace("**", "").strip(" .;:")
        if carry and tail and claims:
            # label followed by uncited detail on the same line: check the
            # detail against the label's citation
            claims.append((carry + " " + tail, claims[-1][1]))
    return claims


def _verify_one(client, text, pmid, d):
    if d is None:
        return {"claim": text, "pmid": pmid, "verdict": "unsupported",
                "quote": "", "issue": "PMID is not in the corpus"}, Usage(CHAT_MODEL)
    user = (f"CLAIM: {text}\n\nABSTRACT (PMID {pmid}, {d['first_author']} {d['year']}):\n"
            f"TITLE: {d['title']}\n{d['abstract']}")
    out, usage = _chat(client, VERIFY_INSTRUCTIONS, user, max_tokens=2048)
    try:
        v = _parse_json(out)
    except json.JSONDecodeError:
        v = {"verdict": "parse_error", "quote": out[:200], "issue": "model did not return JSON"}
    return {"claim": text, "pmid": pmid, **v}, usage


def verify_claims(review, documents, client=None, progress=None, workers=4):
    """Check every cited claim against each abstract it cites, a few calls in
    parallel. Returns (rows in review order, Usage)."""
    from concurrent.futures import ThreadPoolExecutor

    client = client or anthropic_client()
    by_pmid = {d["pmid"]: d for d in documents}
    pairs = [(text, pmid) for text, pmids in split_claims(review) for pmid in pmids]
    rows, total, done = [None] * len(pairs), Usage(CHAT_MODEL), 0

    def job(i):
        text, pmid = pairs[i]
        return i, _verify_one(client, text, pmid, by_pmid.get(pmid))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (row, usage) in ex.map(job, range(len(pairs))):
            rows[i] = row
            total.input_tokens += usage.input_tokens
            total.output_tokens += usage.output_tokens
            total.calls += usage.calls
            done += 1
            if progress:
                progress(done, len(pairs))
    return rows, total
