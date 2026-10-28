"""
Helpers for the Module 1 app: argue a research gap (v4).

Pipeline pieces, in the order the app uses them:
  build_corpus        load a shipped pool (abstracts + vectors, cached on disk)
  build_custom_pool   same for a student's own PubMed query
  nearest             papers most similar to one paper (vector to vectors)
  search              papers most similar to a question (embedding)
  grade_relevance     the model scores each paper 1-10 against the gap; code derives thesis/antithesis/neutral
  argue               the model writes the dialectic (thesis, antithesis, synthesis) and the verdict
  verify_claims       the model checks every cited sentence against its abstract

Every model call takes the system prompt and model as arguments, because the
app lets students edit both, and returns a Usage so the app can show cost.
"""

import datetime as dt
import hashlib
import html
import json
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

EMBED_MAX_CHARS = 24000   # the model takes 8192 tokens; a few PubMed abstracts run past that


def embed_texts(texts, client=None, batch_size=100):
    """Embed a list of strings. Returns (unit-normalised array, Usage)."""
    client = client or openai_client()
    vectors, usage = [], Usage(EMBED_MODEL)
    texts = [t[:EMBED_MAX_CHARS] for t in texts]
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
   1-2  the paper's own study does what the gap says nobody has done (every condition met), or shows the gap's premise is wrong
   3-4  it does the deciding work and only a secondary condition is not reported, or it shows that filling the gap would matter little (the problem is solved another way, is minor, or the approach is known not to work)
   5-6  mixed: some evidence each way, or the paper's work sits beside the gap without settling it
   7-8  it bears on the question but a condition is shown unmet (met: no), however much of the rest it does
   9-10 its abstract names the missing work as not yet done, or shows the problem matters and is unsolved
   A condition marked no keeps the score at 5 or above unless the paper undermines the premise or the worth of the gap; say which in why. A paper on a different disease, task or outcome is neutral, not a low score."""),

    ("output", "OUTPUT FORMAT", """Return only a valid JSON array, one object per paper in the original order, starting with [ and ending with ]. Each object has these fields:

    pmid            string
    bears           "yes" | "no"
    score           integer 1-10, or null when bears is "no"
    evidence_quote  one verbatim sentence from the abstract that justifies the score (when bears is "no", one showing what the paper actually studies), or null if none fits
    why             always one sentence: what decided the score (the condition met or unmet, the premise undermined, why the gap would matter little), or for neutral why the paper is off the question
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


VERDICTS = ("gap stands", "gap narrowed", "gap closed", "can't tell")

ARGUE_SECTIONS = [
    ("background", "BACKGROUND", """You are writing the argument for a proposed research gap as a dialectic, for a clinician audience. A reviewer will check every cited sentence against the abstract it cites, so use only the material in the context, and cite every claim as [PMID n], or [PMID n; PMID m] when a sentence rests on several papers.

Research question:
{research_question}

Proposed research gap:
{proposed_gap}"""),

    ("todo", "WHAT TO DO", """Write three parts, under 400 words in all, then give a verdict.

1. Thesis: the gap as proposed, and the papers whose own limits point at it. For each one, one sentence on what its study did and which condition it stopped short of, or where its abstract names the missing work.
2. Antithesis: the strongest case against the gap. For each antithesis paper, what its own study did that meets the conditions, with cohort, size and headline result, stated at its strongest before it is weighed. If there is none, say so and name the thesis papers that come closest.
3. Synthesis: what survives. Restate the gap as the narrowest claim the antithesis leaves standing, or concede that the gap as proposed is closed. Name the condition the disagreement turns on, and what the search funnel means for how far the verdict can be trusted.
4. Verdict, one of:
   "gap stands"    nothing in the context does what the gap says nobody has done
   "gap narrowed"  something does part of it; the gap survives only in the narrower form you stated
   "gap closed"    a paper's own study does it, or shows the premise wrong, and nothing worth stating survives
   "can't tell"    the abstracts do not settle the deciding condition
   The verdict must match the synthesis: a narrower surviving claim means "gap narrowed", not "gap closed".
5. Rules. "Not reported in this abstract" is not "not studied". "Not retrieved" is not "no study exists": write "none of these abstracts", never "no study". Each record carries SCORE (1 = strongest against the gap, 10 = strongest for) and STANCE derived from it; a low score with weak quotes is a weak antithesis, say so. Where a record carries FLAG, the score and the grader's own checks disagree: trust the quotes over the number, and the abstract over both."""),

    ("output", "OUTPUT FORMAT", """Return ONLY a JSON object with keys "thesis", "antithesis", "synthesis" (each a string of prose with the citations) and "verdict" (one of the four strings)."""),
]

# What a student starts from: one line per box. The model gets exactly what
# is in the boxes, so adding a rule here is the exercise.
ARGUE_SECTIONS_STUDENT = [
    ("background", "BACKGROUND", "You are writing the argument for a proposed research gap as thesis, antithesis and synthesis, for clinicians, using only the papers in the context and citing every claim as [PMID n]."),
    ("todo", "WHAT TO DO", "Write the thesis (the papers that show the gap is real), the antithesis (the strongest papers against it), the synthesis (what survives), and a verdict: gap stands, gap narrowed, gap closed, or can't tell."),
    ("output", "OUTPUT FORMAT", "Return ONLY a JSON object with keys thesis, antithesis, synthesis and verdict."),
]
ARGUE_INSTRUCTIONS = join_sections((title, text) for _, title, text in ARGUE_SECTIONS)

VERIFY_SECTIONS = [
    ("background", "BACKGROUND", """You are a strict fact-checker. You get one claim and ONE of the abstracts it cites, and you judge only from that abstract. Every number and named cohort that the claim attributes to this paper must appear in, or follow directly from, the abstract."""),

    ("todo", "WHAT TO DO", """1. Find this paper's share of the claim. A sentence that summarizes or counts several papers ("most studies ...", "five records each report ...") is judged only for its part about this paper; that the other papers are absent from this abstract is not a contradiction.
2. Check every number and cohort in that share against the abstract.
3. Choose the verdict:
   "supported"      the abstract is consistent with this paper's share of the claim, even if loosely stated; put any looseness in "issue"
   "not supported"  the abstract contradicts the claim, or the claim attributes to this paper a number or cohort the abstract gives differently
   "can't tell"     the abstract says nothing about this paper's share of the claim. "Not reported in this abstract" is not "not studied".
4. Quote the abstract sentence that decides it."""),

    ("output", "OUTPUT FORMAT", """Return ONLY a JSON object:
{"verdict": "supported" | "not supported" | "can't tell",
 "quote": "the abstract sentence that decides it, verbatim, or empty",
 "issue": "what is loose, missing or contradicted, one short sentence, or empty"}"""),
]

VERIFY_SECTIONS_STUDENT = [
    ("background", "BACKGROUND", "You are a fact-checker who gets one claim and one abstract it cites, and judges from that abstract alone."),
    ("todo", "WHAT TO DO", "Decide whether the abstract supports the claim, contradicts it, or says nothing about it, and quote the sentence that decides it."),
    ("output", "OUTPUT FORMAT", "Return ONLY a JSON object with keys verdict (supported, not supported, or can't tell), quote and issue."),
]
VERIFY_INSTRUCTIONS = join_sections((title, text) for _, title, text in VERIFY_SECTIONS)

CLAIM_VERDICTS = ("supported", "not supported", "can't tell")



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
    if score <= 4 and relation == "no":
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


# ---------------------------------------------------------------- Q3 stance panels

PANEL_CSS = """
body { margin: 0; font-family: "Source Sans Pro", sans-serif; font-size: 14px; color: #262730; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
h4 { margin: 0 0 6px 0; font-size: 14px; }
.p { display: block; border-radius: 6px; padding: 6px 10px; margin: 4px 0; cursor: pointer;
     color: inherit; text-decoration: none; }
.p:hover { filter: brightness(0.94); outline: 2px solid #262730; }
.t { background: #e6f4ea; border: 1px solid #8fd19e; }
.a { background: #fdecea; border: 1px solid #f1a9a0; }
.n { background: #f0f0f0; border: 1px solid #cccccc; display: inline-block; margin: 2px 6px 2px 0; }
.none { color: #888; font-style: italic; }
#detail { margin-top: 10px; min-height: 84px; background: #262730; color: #fff; padding: 8px 12px;
          border-radius: 6px; font-size: 13px; line-height: 1.4; }
#detail b { color: #ffd66b; }
#detail .hint { color: #aab; font-style: italic; }
"""

PANEL_JS = """
var detail = document.getElementById('detail');
var hint = detail.innerHTML;
document.querySelectorAll('.p').forEach(function (el) {
  el.addEventListener('mouseenter', function () { detail.innerHTML = el.dataset.detail; });
});
document.body.addEventListener('mouseleave', function () { detail.innerHTML = hint; });
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
    strongest first. The panel shows only author and year; hovering fills a
    detail box under the columns with the title, score and the grader's why;
    clicking opens the paper on PubMed. Neutral papers sit between the columns
    and the detail box in grey. Returns (html, height_px)."""
    def panel(r, cls, right=False):
        d = r["doc"]
        label = f"{surname(d['first_author'])} et al. {d['year']}" + (" (anchor)" if d["pmid"] == anchor else "")
        score = r.get("score")
        tip = (f"<b>{html.escape(d['title'])}</b><br>"
               + (f"Score {html.escape(str(score))} out of 10. " if score is not None else "")
               + html.escape(r.get("why") or ""))
        if r.get("flag"):
            tip += f"<br>Flag: {html.escape(r['flag'])}."
        url = f"https://pubmed.ncbi.nlm.nih.gov/{d['pmid']}/"
        return (f'<a class="p {cls}" href="{url}" target="_blank" rel="noopener" '
                f'data-detail="{html.escape(tip, quote=True)}">{html.escape(label)}</a>')

    by = {k: [r for r in records if stance_of(r) == k] for k in STANCES}
    thesis = sorted(by["thesis"], key=lambda r: -(score_of(r) or 0))
    anti = sorted(by["antithesis"], key=lambda r: score_of(r) or 11)
    left = "".join(panel(r, "t") for r in thesis) or '<div class="none">none</div>'
    right = "".join(panel(r, "a") for r in anti) or '<div class="none">none</div>'
    body = (f'<div class="cols"><div><h4>Papers for the gap, thesis ({len(thesis)})</h4>{left}</div>'
            f'<div><h4>Papers against the gap, antithesis ({len(anti)})</h4>{right}</div></div>')
    if by["neutral"]:
        body += ('<div style="margin-top:8px">Neutral papers, with no bearing on the question '
                 + "".join(panel(r, "n") for r in by["neutral"]) + "</div>")
    body += ('<div id="detail"><span class="hint">Hover over a panel to see the title, the score and the reason. '
             'Click a panel to open the paper on PubMed.</span></div>')
    rows = max(len(thesis), len(anti), 1)
    height = 30 + rows * 38 + (44 if by["neutral"] else 0) + 110
    return (f"<html><head><style>{PANEL_CSS}</style></head><body>{body}<script>{PANEL_JS}</script></body></html>",
            height)


# ---------------------------------------------------------------- Q4 verification

CITE_RE = re.compile(r"\[(PMID[^\]]*)\]")
PMID_RE = re.compile(r"\d{7,8}")


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(])")
_LABEL_TOKENS = {"et", "al", "al.", "and", "&", ",", ";", "colleagues"}


def _label_only(text):
    """'Bai et al', 'Chang and Lee, 2020', 'the UKLS trial'? Only the first two:
    a fragment that names authors and nothing else."""
    words = [w for w in re.split(r"[\s,;]+", text.strip(" ,;:()")) if w]
    rest = [w for w in words if w.lower() not in _LABEL_TOKENS and not re.fullmatch(r"\(?\d{4}\)?", w)]
    return 0 < len(rest) <= 2


def split_claims(review):
    """Break the review into (claim_text, [pmids]) pairs. A claim is the
    sentence that ends with a citation bracket. Three repairs on top of that:
    an uncited sentence between two citations belongs to the earlier one; a
    bare author label cited on its own ("Bai et al [PMID n]") attaches its
    PMID to the sentence it sits in; a label followed by uncited detail on the
    same line is checked as label + detail."""
    claims = []
    for line in review.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        pos, carry, carry_pmids, first = 0, "", [], len(claims)
        for m in CITE_RE.finditer(line):
            text = line[pos:m.start()].replace("**", "").strip(" .;:")
            pmids = PMID_RE.findall(m.group(1))
            pos = m.end()
            if not pmids:
                continue
            sentences = [s for s in _SENT_RE.split(text) if s.strip()] if text else []
            if len(sentences) > 1 and len(claims) > first:
                claims[-1][0] += " " + " ".join(s.strip(" .;:") for s in sentences[:-1])
                text = sentences[-1].strip(" .;:")
            if carry:
                if _label_only(text) and len(claims) > first:
                    # a list of labels: each PMID belongs to the sentence before the list
                    claims[-1][1].extend(p for p in carry_pmids if p not in claims[-1][1])
                    carry, carry_pmids = text, pmids
                    continue
                text, pmids = (carry + " " + text).strip(), carry_pmids + pmids
                carry, carry_pmids = "", []
            if not text:
                if len(claims) > first:
                    claims[-1][1].extend(p for p in pmids if p not in claims[-1][1])
                continue
            if _label_only(text) or len(text.split()) < 5:
                carry, carry_pmids = text, pmids
                continue
            claims.append([text, list(pmids)])
        tail = line[pos:].replace("**", "").strip(" .;:,")
        if carry:
            if tail:
                claims.append([carry + " " + tail, carry_pmids])
            elif len(claims) > first:
                claims[-1][1].extend(p for p in carry_pmids if p not in claims[-1][1])
            else:
                claims.append([carry, carry_pmids])
        elif tail and len(claims) > first and len(tail.split()) >= 5:
            claims[-1][0] += " " + tail
    return [(t, p) for t, p in claims]


def _verify_one(client, text, pmid, d, system=VERIFY_INSTRUCTIONS, model=CHAT_MODEL):
    if d is None:
        return {"claim": text, "pmid": pmid, "verdict": "not supported",
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


# ---------------------------------------------------------------- Q4 argument

def funnel_line(funnel, n_docs=None, screened=None):
    """One or two sentences of retrieval numbers for prompts and the report."""
    if not funnel or funnel.get("total") is None:
        return "The search funnel was not recorded."
    s = f"PubMed matched {funnel['total']} records"
    s += f" on {funnel['fetch_date']}." if funnel.get("fetch_date") else "."
    parts = []
    if funnel.get("fetched") is not None:
        parts.append(f"{funnel['fetched']} were fetched")
    if n_docs is not None:
        parts.append(f"{n_docs} have an abstract and form the corpus")
    if screened is not None:
        parts.append(f"{screened} were scored")
    if parts:
        s += " " + (", ".join(parts[:-1]) + (", and " if len(parts) > 1 else "") + parts[-1]).capitalize() + "."
    return s


def argue(question, gap, records, funnel_text, client=None, system=ARGUE_INSTRUCTIONS, model=CHAT_MODEL):
    """One call that writes thesis, antithesis, synthesis and the verdict from
    the thesis and antithesis records. Returns (dict_or_None, Usage, raw).
    None when the reply is not a JSON object with the four keys."""
    client = client or anthropic_client()
    system, filled = fill_prompt(system, question, gap)
    user = ("" if filled else f"{question_block(question, gap)}\n\n") + f"{funnel_text}\n\nCONTEXT:\n{record_block(records)}"
    text, usage = _chat(client, system, user, max_tokens=8192, model=model)
    out = None
    for attempt in range(2):
        try:
            out = _parse_json(text)
            break
        except json.JSONDecodeError:
            if attempt == 0 and not usage.truncated:
                text, u2 = _chat(client, system, user, max_tokens=8192, model=model)
                _add(usage, u2)
    if not isinstance(out, dict) or not all(isinstance(out.get(k), str) for k in ("thesis", "antithesis", "synthesis")):
        return None, usage, text
    out["verdict"] = str(out.get("verdict", "")).strip().lower()
    if out["verdict"] not in VERDICTS:
        out["verdict"] = "can't tell"
    return out, usage, text


def argument_text(out):
    """The three parts as one cited text, for verification and the report."""
    return "\n\n".join(f"{k.upper()}\n{out.get(k, '')}" for k in ("thesis", "antithesis", "synthesis"))
