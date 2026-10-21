"""
Helpers for the Module 1 literature-review app (v3).

Pipeline pieces, in the order the app uses them:
  build_corpus        load a shipped pool (abstracts + vectors, cached on disk)
  build_custom_pool   same for a student's own PubMed query
  nearest             papers most similar to one paper (vector to vectors)
  search              papers most similar to a question (embedding)
  keyword_search      papers most similar to a question (BM25 over words)
  hybrid_search       the two rankings fused
  rewrite_query       the model turns the question into a search string
  grade_relevance     the model grades each paper; the prompt owns the format
  synthesize          the model writes a short review from the graded papers
  verify_claims       the model checks every cited sentence against its abstract

Every model call takes the system prompt and model as arguments, because the
app lets students edit both, and returns a Usage so the app can show cost.
"""

import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from pubmed_ingest import load_corpus, read_pmids

# ---------------------------------------------------------------- models

CHAT_MODEL = "claude-haiku-5-5"
EMBED_MODEL = "text-embedding-3-small"

# label shown in the app -> model id
MODELS = {
    "Haiku 5.5": "claude-haiku-5-5",
    "Sonnet 5.5": "claude-sonnet-5-5",
}

# USD per million tokens
PRICE = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 3.00, "output": 15.00},
    EMBED_MODEL: {"input": 0.02, "output": 0.0},
}

EXCLUDE_CLAUSE = (' NOT (Review[pt] OR "Systematic Review"[pt] OR Meta-Analysis[pt] '
                  'OR Editorial[pt] OR Comment[pt] OR Letter[pt] OR News[pt])')

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


ICITE_URL = "https://icite.od.nih.gov/api/pubs"


LAST_ICITE_ERROR = None


def fetch_citations(pmids, batch_size=200, attempts=3):
    """Citation counts from NIH iCite (free, no key). Returns {pmid: count}.
    PubMed itself does not carry cited-by counts. A failed batch is retried;
    whatever was fetched is kept, and the last error is stored in
    LAST_ICITE_ERROR so the app can show why counts are missing."""
    import time

    import requests

    global LAST_ICITE_ERROR
    LAST_ICITE_ERROR = None
    counts = {}
    headers = {"User-Agent": "pubmed2026-course-app (contact: course instructor)"}
    for i in range(0, len(pmids), batch_size):
        batch = ",".join(pmids[i:i + batch_size])
        for attempt in range(attempts):
            try:
                r = requests.get(ICITE_URL, params={"pmids": batch, "fl": "pmid,citation_count"},
                                 headers=headers, timeout=60)
                r.raise_for_status()
                for row in r.json().get("data", []):
                    counts[str(row["pmid"])] = int(row.get("citation_count") or 0)
                break
            except Exception as e:  # noqa: BLE001
                LAST_ICITE_ERROR = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(2 ** attempt)
    return counts


def ensure_citations(documents, save_path=None, progress=None):
    """Stamp a "citations" count on every record that lacks one. Writes the
    updated records back to save_path when given."""
    missing = [d["pmid"] for d in documents if "citations" not in d]
    if not missing:
        return documents
    if progress:
        progress(f"Fetching citation counts for {len(missing)} papers from iCite ...")
    counts = fetch_citations(missing)
    if not counts:
        return documents
    for d in documents:
        if "citations" not in d and d["pmid"] in counts:
            d["citations"] = counts[d["pmid"]]
    if save_path:
        try:
            with open(save_path, "w") as f:
                json.dump(documents, f, indent=1, ensure_ascii=False)
        except OSError:
            pass
    return documents


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
            return ensure_citations(documents, abstracts, progress), vectors, None, "cache"

    if progress:
        progress("Fetching abstracts from PubMed ...")
    documents = load_corpus(read_pmids(pool_path(pool, "pmids.txt")))
    ensure_citations(documents, progress=progress)
    if progress:
        progress(f"Embedding {len(documents)} abstracts ...")
    vectors, usage = embed_texts([d["title"] + "\n" + d["abstract"] for d in documents])
    with open(abstracts, "w") as f:
        json.dump(documents, f, indent=1, ensure_ascii=False)
    np.save(embeddings, vectors)
    return documents, vectors, usage, "network"


def tsne_2d(vectors, seed=42):
    """Project the embedding vectors to two dimensions for a scatter plot.
    Close dots are similar papers; the overall layout is arbitrary."""
    from sklearn.manifold import TSNE

    n = vectors.shape[0]
    perplexity = max(2, min(30, (n - 1) // 4))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(vectors)


def load_papers(pool=DEFAULT_POOL):
    """The question, anchor and companion PMIDs for a pool."""
    with open(pool_path(pool, "papers.json")) as f:
        return json.load(f)


def pool_query(pool):
    try:
        with open(pool_path(pool, "query.txt")) as f:
            return f.read().strip()
    except OSError:
        return ""


def companions(documents, anchor, seed=42, k=2):
    """Two random companions for an anchor, reproducible across students."""
    pool = [d["pmid"] for d in documents if d["pmid"] != anchor]
    return random.Random(seed).sample(pool, k)


def build_custom_pool(query, retmax=200, exclude=True, progress=None, force=False):
    """A student's own PubMed query, fetched and embedded like a shipped pool.
    Cached under pools/custom/<hash>/ so a repeat build is free. Returns
    (documents, vectors, usage_or_None, source, pool_id)."""
    from pubmed_ingest import search_pmids

    term = query.strip() + (EXCLUDE_CLAUSE if exclude else "")
    pool_id = "custom/" + hashlib.sha1(f"{term}|{retmax}".encode()).hexdigest()[:10]
    folder = os.path.join(POOLS_DIR, pool_id)
    abstracts, embeddings = os.path.join(folder, "abstracts.json"), os.path.join(folder, "embeddings.npy")
    if not force and os.path.exists(abstracts) and os.path.exists(embeddings):
        with open(abstracts) as f:
            documents = json.load(f)
        return ensure_citations(documents, abstracts, progress), np.load(embeddings), None, "cache", pool_id

    if progress:
        progress("Searching PubMed ...")
    pmids, total = search_pmids(term, retmax=retmax)
    pmids = sorted(pmids, key=int)
    if not pmids:
        raise ValueError("PubMed returned no records for this query.")
    if progress:
        progress(f"PubMed reports {total} hits; fetching {len(pmids)} abstracts ...")
    documents = load_corpus(pmids)
    if not documents:
        raise ValueError("Records were found but none had an abstract.")
    ensure_citations(documents, progress=progress)
    if progress:
        progress(f"Embedding {len(documents)} abstracts ...")
    vectors, usage = embed_texts([d["title"] + "\n" + d["abstract"] for d in documents])
    try:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "query.txt"), "w") as f:
            f.write(term + "\n")
        with open(abstracts, "w") as f:
            json.dump(documents, f, indent=1, ensure_ascii=False)
        np.save(embeddings, vectors)
    except OSError:
        pass   # read-only disk: still usable for this session
    return documents, vectors, usage, "network", pool_id


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


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return _TOKEN_RE.findall(text.lower())


def keyword_search(question, documents, k=10, k1=1.5, b=0.75):
    """BM25 over title + abstract: the classic word-overlap ranking. No
    embeddings, no model call. A word that appears in few papers counts
    more; long papers are penalised. Returns [(doc, score)]."""
    toks = [_tokens(d["title"] + " " + d["abstract"]) for d in documents]
    n = len(documents)
    avgdl = sum(len(t) for t in toks) / max(1, n)
    df = Counter()
    for t in toks:
        df.update(set(t))
    q = _tokens(question)
    scores = []
    for t in toks:
        tf, dl, s = Counter(t), len(t), 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    order = sorted(range(n), key=lambda i: -scores[i])[:k]
    return [(documents[i], scores[i]) for i in order]


def hybrid_search(question, documents, vectors, k=10, client=None, depth=50, c=60):
    """Reciprocal rank fusion of the embedding and keyword rankings. A paper
    scores sum(1 / (c + rank)) over the lists it appears in. Returns (hits, Usage)."""
    emb, usage = search(question, documents, vectors, k=depth, client=client)
    kw = keyword_search(question, documents, k=depth)
    fused = {}
    for hits in (emb, kw):
        for r, (d, _) in enumerate(hits, 1):
            fused[d["pmid"]] = fused.get(d["pmid"], 0.0) + 1.0 / (c + r)
    by = {d["pmid"]: d for d in documents}
    top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    return [(by[p], s) for p, s in top], usage


REWRITE_INSTRUCTIONS = """Rewrite the research question as a search string for a literature
database: the key concepts, each with one or two synonyms, under 30 words,
no boolean operators, no quotes. Return only the search string."""


def rewrite_query(question, client=None, model=CHAT_MODEL, system=REWRITE_INSTRUCTIONS):
    client = client or anthropic_client()
    text, usage = _chat(client, system, question, max_tokens=2048, model=model)
    return text.strip().strip('"'), usage


# ---------------------------------------------------------------- prompts

def context_block(docs):
    return "\n\n".join(
        f"PMID {d['pmid']} | {d['first_author']} et al., {d['year']} | {d['journal']}\n"
        f"TITLE: {d['title']}\nABSTRACT: {d['abstract']}"
        for d in docs
    )


GRADE_INSTRUCTIONS = """You are screening papers for a literature review.

For EACH paper in the context, judge how well it answers the question.
  HIGH   = directly answers it: the paper's own study addresses the question
           and reports results.
  MEDIUM = partly relevant: related task or population, unclear design, or no results.
  LOW    = not relevant.

Return ONLY a JSON array, one object per paper, in the order given, with keys:
  pmid            string
  grade           "HIGH" | "MEDIUM" | "LOW"
  cohort          the cohort or data source named in the abstract, or null
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


def _chat(client, system, user, max_tokens=4096, model=CHAT_MODEL):
    """One model call. max_tokens must stay large: the model emits a thinking
    block before its text, and a small budget returns an empty answer."""
    r = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low"},
    )
    text = "".join(b.text for b in r.content if b.type == "text").strip()
    return text, Usage(model, r.usage.input_tokens, r.usage.output_tokens, calls=1)


def _parse_json(text):
    """Haiku occasionally wraps JSON in a code fence. Strip it and parse."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


# ---------------------------------------------------------------- Q2 / Q3 / Q4 grading

def _add(total, usage):
    total.input_tokens += usage.input_tokens
    total.output_tokens += usage.output_tokens
    total.calls += usage.calls


def grade_relevance(question, docs, client=None, system=GRADE_INSTRUCTIONS, model=CHAT_MODEL):
    """Grade each document with whatever format the prompt asks for.
    Returns (records_or_None, Usage, raw_text). records is None when the
    reply is not a JSON array of objects, so the app can show the raw text.
    Each record is the model's object plus "doc" (matched by pmid, else by
    position) and, when an evidence_quote key exists, "quote_found"."""
    client = client or anthropic_client()
    user = f"QUESTION: {question}\n\nCONTEXT:\n{context_block(docs)}"
    text, usage = _chat(client, system, user, model=model)
    items = None
    for attempt in range(2):
        try:
            items = _parse_json(text)
            break
        except json.JSONDecodeError:
            if attempt == 0:
                text, u2 = _chat(client, system, user, model=model)   # one retry
                _add(usage, u2)
    if not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
        return None, usage, text
    by_pmid = {d["pmid"]: d for d in docs}
    records = []
    for i, item in enumerate(items):
        pmid = str(item.get("pmid", "")).strip()
        d = by_pmid.get(pmid) or (docs[i] if i < len(docs) else None)
        if d is None:
            continue
        item["pmid"] = d["pmid"]
        item["doc"] = d
        if "evidence_quote" in item:
            item["quote_found"] = quote_in_abstract(item.get("evidence_quote"), d["abstract"])
        records.append(item)
    return records, usage, text


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

_SKIP_KEYS = {"doc", "quote_found", "pmid"}


def grade_of(record):
    return str(record.get("grade", "")).upper()


def record_block(records, attach_full_high=True):
    """Context for the synthesis: HIGH papers in full (optional), the others
    as the grader's own record, whatever keys the prompt produced."""
    parts = []
    for r in records:
        d = r["doc"]
        head = f"PMID {d['pmid']} | {d['first_author']} et al., {d['year']} | grade {grade_of(r) or 'n/a'}"
        if grade_of(r) == "HIGH" and attach_full_high:
            parts.append(f"{head}\nTITLE: {d['title']}\nABSTRACT: {d['abstract']}")
        else:
            fields = "\n".join(f"{k.upper()}: {v}" for k, v in r.items() if k not in _SKIP_KEYS)
            parts.append(f"{head}\nTITLE: {d['title']}\n{fields}")
    return "\n\n".join(parts)


def select_records(records, include=("HIGH", "MEDIUM")):
    """Records whose grade is in `include`. If no record carries a grade
    (the student's prompt dropped it), every record is kept."""
    if not any(grade_of(r) for r in records):
        return list(records), False
    return [r for r in records if grade_of(r) in include], True


def synthesize(question, records, attach_full_high=True, client=None,
               system=SYNTHESIS_INSTRUCTIONS, model=CHAT_MODEL):
    """Write the review from the selected records. Returns (text, Usage)."""
    client = client or anthropic_client()
    user = f"QUESTION: {question}\n\nCONTEXT:\n{record_block(records, attach_full_high)}"
    return _chat(client, system, user, model=model)


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


def _verify_one(client, text, pmid, d, system=VERIFY_INSTRUCTIONS, model=CHAT_MODEL):
    if d is None:
        return {"claim": text, "pmid": pmid, "verdict": "unsupported",
                "quote": "", "issue": "PMID is not in the corpus"}, Usage(model)
    user = (f"CLAIM: {text}\n\nABSTRACT (PMID {pmid}, {d['first_author']} {d['year']}):\n"
            f"TITLE: {d['title']}\n{d['abstract']}")
    out, usage = _chat(client, system, user, max_tokens=2048, model=model)
    try:
        v = _parse_json(out)
    except json.JSONDecodeError:
        v = {"verdict": "parse_error", "quote": out[:200], "issue": "model did not return JSON"}
    return {"claim": text, "pmid": pmid, **v}, usage


def verify_claims(review, documents, client=None, progress=None, workers=4,
                  system=VERIFY_INSTRUCTIONS, model=CHAT_MODEL):
    """Check every cited claim against each abstract it cites, a few calls in
    parallel. Returns (rows in review order, Usage)."""
    from concurrent.futures import ThreadPoolExecutor

    client = client or anthropic_client()
    by_pmid = {d["pmid"]: d for d in documents}
    pairs = [(text, pmid) for text, pmids in split_claims(review) for pmid in pmids]
    rows, total, done = [None] * len(pairs), Usage(model), 0

    def job(i):
        text, pmid = pairs[i]
        return i, _verify_one(client, text, pmid, by_pmid.get(pmid), system=system, model=model)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (row, usage) in ex.map(job, range(len(pairs))):
            rows[i] = row
            _add(total, usage)
            done += 1
            if progress:
                progress(done, len(pairs))
    return rows, total
