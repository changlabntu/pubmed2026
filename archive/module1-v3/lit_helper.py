"""
Helpers for the Module 1 literature-review app (v3).

Pipeline pieces, in the order the app uses them:
  build_corpus        load a shipped pool (abstracts + vectors, cached on disk)
  build_custom_pool   same for a student's own PubMed query
  nearest             papers most similar to one paper (vector to vectors)
  search              papers most similar to a question (embedding)
  grade_relevance     the model scores each paper 1-10 against the gap; code derives thesis/antithesis/neutral
  synthesize          the model writes the dialectic (thesis, antithesis, synthesis) from the graded papers
  verify_claims       the model checks every cited sentence against its abstract
  assess_gap          the model judges whether the proposed gap is still open

Every model call takes the system prompt and model as arguments, because the
app lets students edit both, and returns a Usage so the app can show cost.
"""

import datetime as dt
import hashlib
import html
import json
import math
import os
import random
import re
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
    truncated: bool = False   # the model hit max_tokens, so the text is cut off

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


def add_papers(documents, vectors, pmids, progress=None):
    """Append specific papers (the student's anchor and companions) to a
    corpus. Papers already in the pool are left where they are; the rest are
    fetched, given a citation count, embedded, and appended. Returns
    (documents, vectors, usage_or_None, missing_pmids). Inputs are not mutated."""
    have = {d["pmid"] for d in documents}
    wanted = [p for p in pmids if p not in have]
    documents, vectors = list(documents), vectors
    if not wanted:
        return documents, vectors, None, []
    if progress:
        progress(f"Fetching {len(wanted)} chosen paper(s) from PubMed ...")
    extra = load_corpus(wanted)
    ensure_citations(extra)
    missing = [p for p in wanted if p not in {d["pmid"] for d in extra}]
    if not extra:
        return documents, vectors, None, missing
    ev, usage = embed_texts([d["title"] + "\n" + d["abstract"] for d in extra])
    return documents + extra, np.vstack([vectors, ev]), usage, missing


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


def pool_meta(pool):
    """The retrieval funnel for a pool: how many records PubMed matched, how
    many were fetched, and when. Reads meta.json; older pools fall back to
    hits.txt, pmids.txt and the file date of abstracts.json. Missing values
    are None."""
    meta = {"total": None, "fetched": None, "fetch_date": None}
    try:
        with open(pool_path(pool, "meta.json")) as f:
            meta.update({k: v for k, v in json.load(f).items() if k in meta})
            return meta
    except (OSError, ValueError):
        pass
    try:
        with open(pool_path(pool, "hits.txt")) as f:
            meta["total"] = int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    try:
        meta["fetched"] = len(read_pmids(pool_path(pool, "pmids.txt")))
    except OSError:
        pass
    try:
        mtime = os.path.getmtime(pool_path(pool, "abstracts.json"))
        meta["fetch_date"] = dt.date.fromtimestamp(mtime).isoformat() + " (file date)"
    except OSError:
        pass
    return meta


def write_pool_meta(folder, term, total, pmids):
    """meta.json and pmids.txt for a freshly built pool."""
    with open(os.path.join(folder, "pmids.txt"), "w") as f:
        f.write("\n".join(pmids) + "\n")
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump({"term": term, "total": total, "fetched": len(pmids),
                   "fetch_date": dt.date.today().isoformat()}, f, indent=1)


def companions(documents, anchor, seed=42, k=2):
    """Two random companions for an anchor, reproducible across students."""
    pool = [d["pmid"] for d in documents if d["pmid"] != anchor]
    return random.Random(seed).sample(pool, k)


def build_custom_pool(query, retmax=200, exclude=True, progress=None, force=False):
    """A student's own PubMed query, fetched and embedded like a shipped pool.
    Cached under pools/custom/<hash>/ so a repeat build is free. Returns
    (documents, vectors, usage_or_None, source, pool_id). The pool's
    query.txt also records how many records PubMed matched in total."""
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
    pmids, total = search_pmids(term, retmax=retmax, sort="relevance")
    pmids = sorted(pmids, key=int)
    if not pmids:
        raise ValueError("PubMed returned no records for this query.")
    if progress:
        progress(f"PubMed matched {total} records; fetching the {len(pmids)} most relevant ...")
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
        with open(os.path.join(folder, "hits.txt"), "w") as f:
            f.write(f"{total}\n")
        write_pool_meta(folder, term, total, pmids)
        with open(abstracts, "w") as f:
            json.dump(documents, f, indent=1, ensure_ascii=False)
        np.save(embeddings, vectors)
    except OSError:
        pass   # read-only disk: still usable for this session
    return documents, vectors, usage, "network", pool_id


# ---------------------------------------------------------------- Q3 retrieval

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

def question_block(question, gap=None):
    """The research question, plus the student's proposed gap and what would
    disprove it when given. Goes in the user message, so the editable system
    prompts stay clean."""
    s = f"QUESTION: {question}"
    if gap and (gap.get("gap") or "").strip():
        s += f"\nPROPOSED GAP: {gap['gap'].strip()}"
        if (gap.get("disproof") or "").strip():
            s += f"\nEVIDENCE THAT WOULD DISPROVE THE GAP: {gap['disproof'].strip()}"
    return s


def context_block(docs):
    return "\n\n".join(
        f"PMID {d['pmid']} | {d['first_author']} et al., {d['year']} | {d['journal']}\n"
        f"TITLE: {d['title']}\nABSTRACT: {d['abstract']}"
        for d in docs
    )


# The grading prompt is three sections so the app can show each in its own
# box; the heading of each is the box label. GRADE_INSTRUCTIONS is the sections
# joined, which is what gets sent, after {research_question} and {proposed_gap}
# are filled in (see fill_prompt).
GRADE_SECTIONS = [
    ("background", "BACKGROUND", """You are screening scientific papers for a literature review that argues for or against a proposed research gap. A reviewer will read the papers you keep and verify what you extract, so state nothing the abstract does not support. When unsure whether a paper bears on the question, say it does; when unsure between two scores, give the lower one. A reviewer reads every low-scoring paper in full and can discard it, but never sees one you screened out.

Research_question:
{research_question}

Proposed research gap:
{proposed_gap}"""),

    ("todo", "WHAT TO DO", """For each paper in the provided context:

1. Read only its title and abstract, and count only the paper's own work, not studies it cites. If title and abstract disagree, follow the abstract. Treat anything unstated as unknown, never as absent.
2. Decide whether the paper's own work bears on the research question (bears: yes or no). If no, set score and gap_checks to null.
3. Check the gap for every paper that bears on the question. Split the gap into conditions, worded the same way for every paper so results can be compared, and mark each yes, no, or not reported. Use no only when the abstract shows the work differs.
4. Score the paper from 1 to 10 for what it says about the proposed gap. Low scores undermine the gap as worth doing; high scores support it.
   1-2  the paper's own study does what the gap says nobody has done, or shows the gap's premise is wrong
   3-4  it does most of that work and the rest is not reported, or it shows that filling the gap would matter little (the problem is solved another way, is minor, or the approach is known not to work)
   5-6  mixed: some evidence each way, or the paper's work sits beside the gap without settling it
   7-8  it bears on the question but is shown to stop short of a named condition
   9-10 its abstract names the missing work as not yet done, or shows the problem matters and is unsolved"""),

    ("output", "OUTPUT FORMAT", """Return only a valid JSON array, one object per paper in the original order, starting with [ and ending with ]. Each object has these fields:

    pmid            string
    bears           "yes" | "no"
    score           integer 1-10, or null when bears is "no"
    evidence_quote  one verbatim sentence from the abstract that justifies the score (when bears is "no", one showing what the paper actually studies), or null if none fits
    why             one sentence naming what decided the score: the condition met or unmet, the premise undermined, or why the gap would matter little
    gap_checks      [{"condition": string, "met": "yes" | "no" | "not reported", "quote": string | null}] | null
                    quote is one verbatim sentence supporting met; null when met is "not reported\""""),
]


def join_sections(items):
    """(heading, text) pairs to one system prompt: each heading on its own
    line above its text, blank-line separated, sections with no text dropped."""
    return "\n\n".join(f"{title}\n\n{text.strip()}" for title, text in items if text and text.strip())


GRADE_INSTRUCTIONS = join_sections((title, text) for _, title, text in GRADE_SECTIONS)
PLACEHOLDERS = ("{research_question}", "{proposed_gap}")


def fill_prompt(system, question, gap=None):
    """Put the question and gap into the system prompt's placeholders. Returns
    (filled_system, had_placeholders): when the student removed the
    placeholders, the caller puts the question block in the user message."""
    had = any(p in system for p in PLACEHOLDERS)
    g = (gap or {}).get("gap", "").strip() if gap else ""
    d = (gap or {}).get("disproof", "").strip() if gap else ""
    proposed = g or "(none given)"
    if g and d:
        proposed += f"\nEvidence that would disprove it: {d}"
    return system.replace("{research_question}", question).replace("{proposed_gap}", proposed), had


SYNTHESIS_INSTRUCTIONS = """You are writing the argument for a proposed research gap as a dialectic,
for a clinician audience. Use only the material in the context. Cite every claim
as [PMID n]; if a sentence rests on several papers, cite each: [PMID n; PMID m].

Structure, under 400 words, with these three headings:

THESIS. The gap as proposed, and the papers that point at it. For each thesis
paper, one sentence on what its own study did and which condition its design
stopped short of, or where its abstract names the missing work.

ANTITHESIS. The strongest case against the gap. For each antithesis paper, what
its own study did that meets the conditions, with cohort, size and headline
result. State the counterexample at its strongest before weighing it. If there
is no antithesis paper, say so and name the thesis papers that come closest.

SYNTHESIS. What survives. Restate the gap as the narrowest claim the antithesis
leaves standing, or concede that the gap as proposed is closed. Name the
condition the disagreement turns on.

Each record carries SCORE (1 = strongest against the gap, 10 = strongest for)
and STANCE derived from it; a low score with weak quotes is a weak antithesis,
say so. Where a record carries FLAG, the score and the grader's own checks
disagree: trust the quotes over the number.

Write "none of these abstracts", never "no study": absence from the context is
not absence from the literature. Be concrete with numbers. No preamble."""


VERIFY_INSTRUCTIONS = """You are a strict fact-checker. You get one claim and ONE of the abstracts it cites.
Judge ONLY from the abstract. Every number and named cohort that the claim attributes
to this paper must appear in, or be directly derivable from, the abstract.

A claim may summarize several papers ("most studies ...", "cohorts ranged from ..."). Then
judge only the part that concerns this paper: supported if the abstract is consistent with
its share of the claim, partial if some detail is missing or loosely stated, insufficient
if the abstract says nothing about this paper's share of the claim, unsupported only if
the abstract contradicts the claim. "Not reported in this abstract" is not "not studied".

Return ONLY a JSON object:
{"verdict": "supported" | "partial" | "insufficient" | "unsupported",
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
    return text, Usage(model, r.usage.input_tokens, r.usage.output_tokens, calls=1,
                       truncated=r.stop_reason == "max_tokens")


def _parse_json(text):
    """Haiku occasionally wraps JSON in a code fence. Strip it and parse."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


# ---------------------------------------------------------------- Q2 / Q3 grading

def _add(total, usage):
    total.input_tokens += usage.input_tokens
    total.output_tokens += usage.output_tokens
    total.calls += usage.calls
    total.truncated = total.truncated or usage.truncated


GRADE_MAX_TOKENS = 16384


def _grade_call(question, docs, client, system, model, gap=None, by_position=True):
    """One grading call over `docs`. Returns (records_or_None, Usage, raw).
    Records match the model's objects to documents by pmid; with by_position
    a record whose pmid is unknown falls back to its position in the list."""
    system, filled = fill_prompt(system, question, gap)
    user = ("" if filled else f"{question_block(question, gap)}\n\n") + f"CONTEXT:\n{context_block(docs)}"
    # 20 records with quotes and gap checks run well past 4096 output tokens,
    # and the thinking block counts against the same budget
    text, usage = _chat(client, system, user, max_tokens=GRADE_MAX_TOKENS, model=model)
    items = None
    for attempt in range(2):
        try:
            items = _parse_json(text)
            break
        except json.JSONDecodeError:
            if attempt == 0 and not usage.truncated:   # a cut-off reply would only be cut off again
                text, u2 = _chat(client, system, user, max_tokens=GRADE_MAX_TOKENS, model=model)
                _add(usage, u2)
    if not isinstance(items, list) or not all(isinstance(x, dict) for x in items):
        return None, usage, text
    by_pmid = {d["pmid"]: d for d in docs}
    records, seen = [], set()
    for i, item in enumerate(items):
        pmid = str(item.get("pmid", "")).strip()
        d = by_pmid.get(pmid) or (docs[i] if by_position and i < len(docs) else None)
        if d is None or d["pmid"] in seen:
            continue
        seen.add(d["pmid"])
        item["pmid"] = d["pmid"]
        item["doc"] = d
        if "evidence_quote" in item:
            item["quote_found"] = quote_in_abstract(item.get("evidence_quote"), d["abstract"])
        if isinstance(item.get("gap_checks"), list):
            item["gap_relation"], item["gap_why"] = gap_relation(item["gap_checks"])
        stance = derive_stance(item)
        if stance:
            item = {"pmid": item["pmid"], "stance": stance, **item}
        item["flag"] = stance_flag(item)
        records.append(item)
    return records, usage, text


def gap_relation(checks):
    """Summarise the grader's per-condition checks as one label, by a fixed
    rule rather than by asking the model: every condition met = counterexample
    (the paper does what the gap says nobody has done); any condition shown
    unmet = no; some met and none unmet = partial; nothing decidable = unclear.
    Returns (label, one sentence naming the deciding conditions)."""
    checks = [c for c in checks if isinstance(c, dict)]
    met = {m: [c.get("condition", "?") for c in checks if str(c.get("met", "")).strip().lower() == m]
           for m in ("yes", "no", "not reported")}
    if not checks:
        return "unclear", "no conditions were checked"
    if met["no"]:
        return "no", "not met: " + "; ".join(met["no"])
    if len(met["yes"]) == len(checks):
        return "counterexample", "every condition met"
    if met["yes"]:
        return "partial", "not reported: " + "; ".join(met["not reported"])
    return "unclear", "nothing reported for any condition"


STANCE_CUTOFF = 5   # score 1..5 = antithesis, 6..10 = thesis; bears "no" = neutral


def score_of(record):
    try:
        return int(record.get("score"))
    except (TypeError, ValueError):
        return None


def derive_stance(record):
    """The stance is computed from the model's two numbers, never asked of it:
    neutral when the paper does not bear on the question, else antithesis at
    or below STANCE_CUTOFF and thesis above. None when the prompt returned
    neither field (the student dropped them)."""
    bears = str(record.get("bears", "")).strip().lower()
    if bears == "no":
        return "neutral"
    score = score_of(record)
    if score is None:
        return None
    return "antithesis" if score <= STANCE_CUTOFF else "thesis"


def stance_flag(record):
    """Where the score and the model's own condition checks clearly disagree,
    say so, so the synthesis and the reader trust the quotes over the number.
    Every condition met but scored as thesis is a contradiction. A 1 or 2 with
    a condition shown unmet is allowed (the paper may undermine the gap's
    premise instead) but is marked so the reader checks the why. None
    otherwise, or when there are no checks or no score."""
    score, relation = score_of(record), record.get("gap_relation")
    if relation is None or score is None:
        return None
    if score > STANCE_CUTOFF and relation == "counterexample":
        return f"score {score}, but every condition is met"
    if score <= 2 and relation == "no":
        return f"score {score} with a condition shown unmet: read the why"
    return None


def grade_relevance(question, docs, client=None, system=GRADE_INSTRUCTIONS, model=CHAT_MODEL, gap=None):
    """Grade each document with whatever format the prompt asks for.
    Returns (records_or_None, Usage, raw_text). records is None when the
    reply is not a JSON array of objects, so the app can show the raw text.
    Each record is the model's object plus "doc" (matched by pmid, else by
    position) and, when an evidence_quote key exists, "quote_found"."""
    client = client or anthropic_client()
    return _grade_call(question, docs, client, system, model, gap=gap, by_position=True)


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


# ---------------------------------------------------------------- Q4 synthesis

_SKIP_KEYS = {"doc", "quote_found", "pmid"}


STANCES = ("thesis", "antithesis", "neutral")


def stance_of(record):
    return str(record.get("stance", "")).strip().lower()


def record_block(records, attach_full_antithesis=True):
    """Context for the synthesis: every paper as the grader's own record,
    whatever keys the prompt produced, plus the full abstract for antithesis
    papers, which are the ones the argument turns on."""
    parts = []
    for r in records:
        d = r["doc"]
        head = f"PMID {d['pmid']} | {d['first_author']} et al., {d['year']} | stance {stance_of(r) or 'n/a'}"
        fields = "\n".join(f"{k.upper()}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}"
                           for k, v in r.items() if k not in _SKIP_KEYS)
        body = f"{head}\nTITLE: {d['title']}\n{fields}"
        if stance_of(r) == "antithesis" and attach_full_antithesis:
            body += f"\nABSTRACT: {d['abstract']}"
        parts.append(body)
    return "\n\n".join(parts)


def select_records(records, include=("thesis", "antithesis")):
    """Records whose stance is in `include`. If no record carries a stance
    (the student's prompt dropped it), every record is kept."""
    if not any(stance_of(r) for r in records):
        return list(records), False
    return [r for r in records if stance_of(r) in include], True


def synthesize(question, records, client=None, system=SYNTHESIS_INSTRUCTIONS, model=CHAT_MODEL, gap=None):
    """Write the dialectic from the selected records. Returns (text, Usage)."""
    client = client or anthropic_client()
    user = f"{question_block(question, gap)}\n\nCONTEXT:\n{record_block(records)}"
    return _chat(client, system, user, model=model)


# ---------------------------------------------------------------- Q3 stance panels

PANEL_CSS = """
body { margin: 0; font-family: "Source Sans Pro", sans-serif; font-size: 14px; color: #262730; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
h4 { margin: 0 0 6px 0; font-size: 14px; }
.p { position: relative; border-radius: 6px; padding: 6px 10px; margin: 4px 0; cursor: help; }
.t { background: #e6f4ea; border: 1px solid #8fd19e; }
.a { background: #fdecea; border: 1px solid #f1a9a0; }
.n { background: #f0f0f0; border: 1px solid #cccccc; display: inline-block; margin: 2px 6px 2px 0; }
.tip { display: none; position: absolute; left: 0; top: 100%; z-index: 10; width: 420px; max-width: 90vw;
       background: #262730; color: #fff; padding: 8px 10px; border-radius: 6px; font-size: 13px;
       line-height: 1.35; white-space: normal; box-shadow: 0 2px 8px rgba(0,0,0,.3); }
.p:hover .tip { display: block; }
.r .tip { left: auto; right: 0; }
.tip b { color: #ffd66b; }
.none { color: #888; font-style: italic; }
"""


def surname(author):
    """'Paul R' -> 'Paul', 'van der Berg JM' -> 'van der Berg': drop a trailing
    initials token (PubMed gives LastName Initials)."""
    parts = (author or "").split()
    if len(parts) > 1 and parts[-1].isupper() and len(parts[-1]) <= 3:
        parts = parts[:-1]
    return " ".join(parts) or "?"


def stance_panels_html(records, anchor=None):
    """HTML for two columns of small panels, one per paper: thesis (for the
    gap) on the left in green, antithesis (against) on the right in red,
    strongest first. The panel shows only author and year; hovering shows the
    title, score and the grader's why. Neutral papers sit underneath in grey.
    Returns (html, height_px)."""
    def panel(r, cls, right=False):
        d = r["doc"]
        label = f"{surname(d['first_author'])} et al. {d['year']}" + (" (anchor)" if d["pmid"] == anchor else "")
        tip = f"<b>{html.escape(d['title'])}</b><br>score {html.escape(str(r.get('score', '?')))}: {html.escape(r.get('why') or '')}"
        if r.get("flag"):
            tip += f"<br>flag: {html.escape(r['flag'])}"
        return f'<div class="p {cls}{" r" if right else ""}">{html.escape(label)}<div class="tip">{tip}</div></div>'

    by = {k: [r for r in records if stance_of(r) == k] for k in STANCES}
    thesis = sorted(by["thesis"], key=lambda r: -(score_of(r) or 0))
    anti = sorted(by["antithesis"], key=lambda r: score_of(r) or 11)
    left = "".join(panel(r, "t") for r in thesis) or '<div class="none">none</div>'
    right = "".join(panel(r, "a", right=True) for r in anti) or '<div class="none">none</div>'
    body = (f'<div class="cols"><div><h4>Thesis: for the gap ({len(thesis)})</h4>{left}</div>'
            f'<div><h4>Antithesis: against the gap ({len(anti)})</h4>{right}</div></div>')
    if by["neutral"]:
        body += ('<div style="margin-top:8px">Neutral, no bearing on the question: '
                 + "".join(panel(r, "n") for r in by["neutral"]) + "</div>")
    rows = max(len(thesis), len(anti), 1)
    height = 30 + rows * 38 + (44 if by["neutral"] else 0)
    return f"<html><head><style>{PANEL_CSS}</style></head><body>{body}</body></html>", height


# ---------------------------------------------------------------- Q4 verification

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


# ---------------------------------------------------------------- Q5 gap assessment

GAP_INSTRUCTIONS = """You judge whether a proposed research gap is still open, using ONLY the
material given: the graded papers, the review with its verification table, and
the search funnel (how many records PubMed matched, how many were fetched and
how many the grader saw).

Rules:
- "Not reported in this abstract" is not "not studied".
- "Not retrieved" is not "no study exists": the corpus is only the fetched records.
- A paper counts as a counterexample only if its OWN study does what the gap says
  nobody has done. Name the closest ones even when they fall short, and say why.
- Each graded paper carries SCORE (1 = strongest against the gap, 10 = strongest
  for), STANCE derived from it (antithesis at 5 or below, thesis above, neutral
  when the paper does not bear on the question), GAP_CHECKS (the grader's test of
  each disproof condition, with quotes), GAP_RELATION, and a FLAG where score and
  checks disagree. Start from those, but trust a quote over a number, and the
  abstract over both where you have it.
- The review is a dialectic: THESIS, ANTITHESIS, SYNTHESIS. Weigh the antithesis
  yourself from the quotes and the verification table; do not take the synthesis
  on trust.

Return ONLY a JSON object:
{"verdict": "contradicted" | "partly addressed" | "provisionally supported" | "insufficient evidence",
 "counterexamples": [{"pmid": "string", "why": "one sentence"}],   (up to 3, closest first)
 "uncertainty": "what would change the verdict, one or two sentences",
 "search_limits": "what the funnel numbers imply for this verdict, one or two sentences"}"""

GAP_VERDICTS = ("contradicted", "partly addressed", "provisionally supported", "insufficient evidence")


def funnel_line(funnel, n_docs=None, screened=None):
    """One sentence of retrieval numbers for prompts and the report."""
    if not funnel:
        return "Search funnel: not recorded."
    parts = []
    if funnel.get("total") is not None:
        parts.append(f"PubMed matched {funnel['total']} records")
    if funnel.get("fetched") is not None:
        parts.append(f"{funnel['fetched']} fetched")
    if n_docs is not None:
        parts.append(f"{n_docs} with an abstract (the corpus)")
    if screened is not None:
        parts.append(f"{screened} graded")
    s = "Search funnel: " + ", ".join(parts) if parts else "Search funnel: not recorded"
    if funnel.get("fetch_date"):
        s += f"; fetched {funnel['fetch_date']}"
    return s + "."


def assess_gap(question, gap, records, review, rows, funnel_text, client=None,
               system=GAP_INSTRUCTIONS, model=CHAT_MODEL):
    """One call that weighs the proposed gap against the strongest papers.
    Returns (verdict_dict_or_None, Usage, raw_text)."""
    client = client or anthropic_client()
    table = "\n".join(f"{r['verdict']} | {r['pmid']} | {r['claim']}" for r in rows) or "(no claims checked)"
    user = (f"{question_block(question, gap)}\n\n{funnel_text}\n\n"
            f"GRADED PAPERS:\n{record_block(records)}\n\n"
            f"REVIEW:\n{review}\n\nVERIFICATION (verdict | pmid | claim):\n{table}")
    text, usage = _chat(client, system, user, model=model)
    out = None
    for attempt in range(2):
        try:
            out = _parse_json(text)
            break
        except json.JSONDecodeError:
            if attempt == 0:
                text, u2 = _chat(client, system, user, model=model)
                _add(usage, u2)
    if not isinstance(out, dict):
        return None, usage, text
    return out, usage, text
