"""
Module 1 v4: argue a research gap.

Four moves, the way a paper's introduction argues: state the claim (topic
question, the gap, what would refute it), start from three papers, gather
evidence for and against, write the argument as thesis / antithesis /
synthesis with a verdict, and verify every cited sentence. Every prompt is
editable. A cost cap in the sidebar keeps the bill small. A comment box under
each move and a Download button turn the session into the homework report.

Run:  streamlit run app.py
"""

import datetime as dt
import json
import os
import re
import sys

import streamlit as st
from dotenv import load_dotenv

import lit_helper as L

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Community Cloud keeps secrets in st.secrets; mirror them into the environment
# so the SDKs pick them up the same way as from .env.
for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NCBI_API_KEY", "NCBI_EMAIL", "TEACHER"):
    try:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]
    except Exception:
        pass

st.set_page_config(page_title="Module 1: argue a research gap", layout="wide")
st.title("Argue a research gap")

# Every free-text field (text_input and text_area) gets a light green edge so
# students can see at a glance what they are expected to type into.
st.markdown(
    """<style>
    div[data-baseweb="input"], div[data-baseweb="textarea"] {
        border: 2px solid #8fd19e !important;
        border-radius: 6px;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
        border-color: #3fa65b !important;
    }
    /* the argument and verifier prompt boxes: the ones the homework asks students to change */
    [class*="st-key-w_argue_"] div[data-baseweb="textarea"],
    [class*="st-key-w_verify_"] div[data-baseweb="textarea"] {
        border: 2px solid #e57373 !important;
    }
    [class*="st-key-w_argue_"] div[data-baseweb="textarea"]:focus-within,
    [class*="st-key-w_verify_"] div[data-baseweb="textarea"]:focus-within {
        border-color: #c62828 !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

ss = st.session_state
ss.setdefault("ledger", L.Ledger())
ss.setdefault("client", None)

# Teacher mode shows the cost controls, the spend, the Sonnet option and the
# argument and verifier prompts in full. Students see Haiku only, a hidden cap,
# and an outline of those two prompts. Start it with
#     streamlit run app.py -- --teacher
# or set TEACHER=1 in the environment (Community Cloud: in the Secrets panel).
TEACHER = "--teacher" in sys.argv or os.environ.get("TEACHER", "") == "1"
STUDENT_CAP_CENTS = 10

# q1 = the claim (no result to clear), q2 = corpus + the three starting
# papers, q3 = the evidence, q4 = the argument
EVIDENCE_K = 6   # papers per search in move 3; the two searches are unioned
STEPS = ["q1", "q2", "q3", "q4"]
RESULT_KEYS = {
    "q1": [],
    "q2": ["docs", "V", "corpus_meta", "q2"],
    "q3": ["q3"],
    "q4": ["q4"],
}


# ---------------------------------------------------------------- small helpers

def client():
    if ss.client is None:
        ss.client = L.anthropic_client()
    return ss.client


def clear_from(step):
    """Drop the results of `step` and every step after it."""
    for s in STEPS[STEPS.index(step):]:
        for k in RESULT_KEYS[s]:
            ss.pop(k, None)


def inputs_changed(step, signature):
    """Clear a step when any input it depends on changed since it last ran."""
    key = f"sig_{step}"
    if ss.get(key) != signature:
        if key in ss:
            clear_from(step)
        ss[key] = signature


def run_button(label, key):
    """A run button that refuses once the session cost cap is reached."""
    clicked = st.button(label, key=key, type="primary")
    if not clicked:
        return False
    spent = ss.ledger.cost_usd
    if spent >= CAP_USD:
        st.error(f"You have reached the cost cap. This session has spent {100 * spent:.1f} cents of the "
                 f"{CAP_CENTS} cent cap. " + ("Raise the slider in the sidebar to continue." if TEACHER else
                                              "Download your report; ask your instructor if you need more runs."))
        return False
    return True


def sectioned_prompt_editor(name, sections, label):
    """A prompt in several boxes, always open, one per section, so students can
    change one part at a time. Each box takes its text from a seed through the
    widget's own value= (a value put in session state in an earlier run is never
    shown by the browser). "Reset all" and a question set replace the seeds and
    bump a version so fresh boxes are created. The joined text is kept in
    ss[prompt_<name>] and returned; each box's text in ss[prompt_<name>_<key>]."""
    seed = ss.setdefault(f"{name}_seed", {key: default for key, _, default in sections})
    ver = ss.setdefault(f"{name}_ver", 0)
    st.markdown(f"**{label}**")
    edited = [key for key, _, default in sections
              if ss.get(f"prompt_{name}_{key}", seed[key]).strip() != default.strip()]
    c1, c2 = st.columns([5, 1])
    c1.caption("The boxes below are joined in order into the system prompt. "
               + (f"You have edited {', '.join(edited)}." if edited else "All boxes hold the default text."))
    if c2.button("Reset all", key=f"reset_{name}"):
        ss[f"{name}_seed"] = {key: default for key, _, default in sections}
        ss[f"{name}_ver"] = ver + 1
        st.rerun()
    for key, title, default in sections:
        # label and height must not depend on the text: a keyed widget whose
        # parameters change is a new widget to Streamlit, and starts empty
        ss[f"prompt_{name}_{key}"] = st.text_area(
            title, value=seed[key], key=f"w_{name}_{key}_{ver}",
            height=max(136, 48 * (default.count("\n") + 2)))
    combined = L.join_sections((title, ss[f"prompt_{name}_{key}"]) for key, title, _ in sections)
    ss[f"prompt_{name}"] = combined
    with st.expander("The combined prompt, as it is sent", expanded=False):
        st.code(combined, language="text")
    return combined


COMMENT_ASKS = {
    "q2": "Did the three papers fall where you expected, and why did the anchor get the score it did? "
          "Then change one thing in the scoring prompt and score again. What did you change, and what moved?",
    "q3": "Which of the two searches found the paper you would have chosen yourself? "
          "Which red panel do you disagree with, and why?",
    "q4": "Did the synthesis narrow the gap, concede it, or keep it as written, and do you agree with the verdict? "
          "Add one rule to the argument prompt and run again. What changed? "
          "Which verification row do you dispute, after reading that abstract? "
          "Finally, which single change you made had the largest effect on the argument, and how do you know?",
}


def comment_box(step):
    st.markdown(f"**Your comment.** {COMMENT_ASKS[step]}")
    st.text_area("Your comment", key=f"comment_{step}", height=140, label_visibility="collapsed")


def funnel_sentence(meta):
    """Plain-language retrieval funnel for the corpus caption."""
    f = meta.get("funnel") or {}
    n_pool = meta.get("n_pool")
    if f.get("total") is None or n_pool is None:
        return ""
    date = f" on {f['fetch_date']}" if f.get("fetch_date") else ""
    fetched = f.get("fetched")
    s = f"PubMed matched **{f['total']:,}** records{date}."
    if fetched is not None:
        s += f" Fetched **{fetched:,}**; **{n_pool:,}** have abstracts and form the corpus."
        unseen = f["total"] - fetched
        if unseen > 0:
            s += f" The other **{unseen:,}** were never seen."
    else:
        s += f" **{n_pool:,}** have abstracts and form the corpus."
    extra = len(meta.get("fixed", []))
    if extra and len(ss.docs) > n_pool:
        s += f" Plus the {len(ss.docs) - n_pool} paper(s) you added."
    return s


def paper_label(d):
    return f"{d['pmid']} · {d['first_author']} {d['year']} · {d['title'][:90]}"


def paper_line(d, score=None, tag=""):
    s = f"**{d['pmid']}** · {d['first_author']} {d['year']} · {d['title']}"
    if score is not None:
        s = f"`{score:.3f}` " + s
    return s + (f" **{tag}**" if tag else "")


def record_rows(records):
    """Flatten the model's records into table rows, keeping its own keys."""
    rows = []
    for r in records:
        row = {"pmid": r["pmid"]}
        for k, v in r.items():
            if k in ("doc", "pmid", "quote_found"):
                continue
            # everything as text: a column that mixes ints and strings (n = 160 vs
            # "160 scans") otherwise trips Arrow when the table is drawn
            if k == "gap_checks" and isinstance(v, list):   # one line per condition, quotes in the expander
                v = "\n".join(f"{c.get('met')}: {c.get('condition')}" for c in v if isinstance(c, dict))
            row[k] = v if v is None or isinstance(v, str) else (
                json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v))
        if "quote_found" in r:
            row["quote found"] = "yes" if r["quote_found"] else "NO"
        rows.append(row)
    return rows


def embed_html(doc, height):
    """st.iframe where this Streamlit has it; the older components call otherwise."""
    (getattr(st, "iframe", None) or st.components.v1.html)(doc, height=height)


def stance_panels(records, anchor=None):
    html_doc, height = L.stance_panels_html(records, anchor)
    embed_html(html_doc, height)


def grading_failed(result):
    """The warning for a scoring call that returned no usable records."""
    st.warning("The reply was cut off at the model's output limit, so the JSON is incomplete. "
               "Shorten what the grading prompt asks for per paper and score again."
               if result.get("truncated") else
               "The reply was not a JSON array of objects. Read the answer as text, or fix the prompt and score again.")
    st.code(result["raw"])


def show_records(result):
    """The audit trail for a scoring result: the full table, one expander per
    paper with the record and the abstract, and the raw reply."""
    records = result["records"]
    st.caption(f"Scored by {result['model']}")
    st.dataframe(record_rows(records), width="stretch", hide_index=True)
    for r in records:
        d = r["doc"]
        with st.expander(f"{d['first_author']} {d['year']}, PMID {d['pmid']}, scored {L.stance_of(r) or 'without a stance'}. Full record and abstract"):
            for k, v in r.items():
                if k not in ("doc", "pmid", "quote_found"):
                    st.write(f"**{k}:** {v}")
            if "quote_found" in r and not r["quote_found"]:
                st.warning("The evidence quote was not found in the abstract. Read it and judge for yourself.")
            st.caption(d["abstract"])
    with st.expander("Raw model reply"):
        st.code(result["raw"])


def md_table(rows):
    if not rows:
        return "_empty_\n"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(out) + "\n"


def stance_line(r):
    """One report line for a scored paper."""
    d = r["doc"]
    stance = L.stance_of(r)
    head = f"- **{d['first_author']} {d['year']}**, PMID {d['pmid']}. {d['title']} "
    if stance == "neutral" or stance == "":
        head += f"Scored {stance or 'without a stance'}. "
    else:
        head += f"Scored {stance}, {r.get('score', '?')} out of 10. "
    return (head + (r.get("why") or "").rstrip(". ") + "."
            + (f" Quote: \"{r['evidence_quote']}\"" if r.get("evidence_quote") else "")
            + (f" Flag: {r['flag']}." if r.get("flag") else ""))


def verification_sentence(rows):
    counts = {v: sum(r["verdict"] == v for r in rows) for v in L.CLAIM_VERDICTS}
    return (f"The verifier checked {len(rows)} claims. {counts['supported']} were supported, "
            f"{counts['not supported']} were not supported, and for {counts[chr(99) + 'an' + chr(39) + 't tell']} "
            f"it could not tell.")


# ---------------------------------------------------------------- question sets and the sidebar

ALLOW_SONNET = TEACHER   # Sonnet 5.5 is offered in teacher mode only

# A question set prefills the own-query mode so the demo runs with no typing:
# question, gap, disproof, PubMed query, the three PMIDs, and optionally edited
# prompts. One JSON file per set in question_sets/; pick one in the sidebar.
QSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "question_sets")
DEFAULT_QSET = "tkr_prediction.json"
# every prompt is edited in sections; a question set saves the edited sections per prompt
SECTIONS = {"grade": L.GRADE_SECTIONS,
            "argue": L.ARGUE_SECTIONS if TEACHER else L.ARGUE_SECTIONS_STUDENT,
            "verify": L.VERIFY_SECTIONS if TEACHER else L.VERIFY_SECTIONS_STUDENT}
DEFAULTS = {name: {key: text for key, _, text in secs} for name, secs in SECTIONS.items()}


def list_qsets():
    if not os.path.isdir(QSETS_DIR):
        return []
    return sorted(f for f in os.listdir(QSETS_DIR) if f.endswith(".json"))


def load_qset(name):
    with open(os.path.join(QSETS_DIR, name)) as f:
        return json.load(f)


def apply_qset(qs):
    """Copy a question set into the widgets. Prompts the set leaves out go back to default."""
    ss.question = qs.get("question", "")
    ss.gap = qs.get("gap", "")
    ss.disproof = qs.get("disproof", "")
    ss.own_query = qs.get("query", "")
    pmids = qs.get("pmids", [])
    ss.fixed_custom = ", ".join(pmids) if isinstance(pmids, list) else str(pmids)
    ss.source = "My own PubMed query"
    prompts = qs.get("prompts") or {}
    for name, secs in SECTIONS.items():
        given = prompts.get(name) or {}
        if isinstance(given, str):   # a whole prompt as one string: first box gets it, the rest go blank
            given = {key: (given if key == secs[0][0] else "") for key in DEFAULTS[name]}
        ss[f"{name}_seed"] = {key: given.get(key, default) for key, default in DEFAULTS[name].items()}
        ss[f"{name}_ver"] = ss.get(f"{name}_ver", 0) + 1
        for key, text in ss[f"{name}_seed"].items():
            ss[f"prompt_{name}_{key}"] = text
        ss[f"prompt_{name}"] = L.join_sections((title, ss[f"{name}_seed"][key]) for key, title, _ in secs)


def on_pick_qset():
    apply_qset(load_qset(ss.qset))


if "qset" not in ss:
    sets = list_qsets()
    ss.qset = DEFAULT_QSET if DEFAULT_QSET in sets else (sets[0] if sets else None)
    if ss.qset:
        apply_qset(load_qset(ss.qset))

with st.sidebar:
    st.subheader("Question set")
    st.selectbox("Load a set", list_qsets(), key="qset", on_change=on_pick_qset,
                 format_func=lambda f: f"{load_qset(f).get('name', f)} ({f})")
    if TEACHER:
        st.subheader("Model")
        model_label = st.radio(
            "Chat model", list(L.MODELS), key="model_label",
            captions=["$0.10 in / $0.50 out per M tokens", "$3 in / $15 out per M tokens, 30x Haiku"],
        )
        st.subheader("Cost cap")
        CAP_CENTS = st.slider("Stop making model calls after this many cents", 1, 50, STUDENT_CAP_CENTS, key="cap_cents")
    else:
        model_label = "Haiku 5.5"
        CAP_CENTS = STUDENT_CAP_CENTS
    MODEL = L.MODELS[model_label]
    CAP_USD = CAP_CENTS / 100
    totals_box = st.container()

# ---------------------------------------------------------------- report and totals

def build_report():
    gap = ss.get("active_gap", {})
    lines = [f"# Argument for a research gap", f"Date: {dt.date.today()}",
             f"Question set: {ss.get('qset') or 'none'}", "",
             "## 1. The claim", "",
             f"- Topic question: {ss.get('active_question', '')}",
             f"- The gap: {gap.get('gap', '')}",
             f"- What would refute it: {gap.get('disproof', '')}", ""]
    meta = ss.get("corpus_meta")
    lines += ["## 2. Starting papers", ""]
    r2 = ss.get("q2")
    if r2 and r2["records"]:
        anchor = (meta or {}).get("fixed", [None])[0]
        lines += [stance_line(r) + (" (anchor)" if r["pmid"] == anchor else "") for r in r2["records"]] + [""]
    elif r2:
        lines += ["Scoring returned no usable records.", "", "```", r2["raw"], "```", ""]
    lines += [f"Comment: {ss.get('comment_q2', '')}", ""]
    lines += ["## 3. Evidence", ""]
    r3 = ss.get("q3")
    if r3 and r3["records"]:
        groups = {k: [r for r in r3["records"] if L.stance_of(r) == k] for k in L.STANCES}
        lines += ["### Papers for the gap (thesis)", ""] + ([stance_line(r) for r in groups["thesis"]] or ["- none"]) + [""]
        lines += ["### Papers against the gap (antithesis)", ""] + ([stance_line(r) for r in groups["antithesis"]] or ["- none"]) + [""]
        lines += ["### Papers with no bearing (neutral)", ""] + ([stance_line(r) for r in groups["neutral"]] or ["- none"]) + [""]
        lines += ["### How it was searched", "",
                  f"The corpus `{meta['pool']}` was built from the PubMed query `{meta['query']}` and has "
                  f"{len(ss.docs)} documents. {L.funnel_line(meta.get('funnel'), meta.get('n_pool'), len(r3['records']))} "
                  f"The two searches, the {EVIDENCE_K} papers closest to the topic question and the {EVIDENCE_K} closest to the anchor, "
                  f"added {r3['n_new']} new papers, which were scored.", ""]
    elif r3:
        lines += ["Scoring returned no usable records.", "", "```", r3["raw"], "```", ""]
    lines += [f"Comment: {ss.get('comment_q3', '')}", ""]
    lines += ["## 4. The argument", ""]
    r4 = ss.get("q4")
    if r4 and r4["argument"]:
        out = r4["argument"]
        for part in ("thesis", "antithesis", "synthesis"):
            lines += [f"### {part.capitalize()}", "", out[part], ""]
        lines += [f"**Verdict: {out['verdict']}**", ""]
        rows = r4["rows"]
        lines += ["## 5. Verification", "", verification_sentence(rows), "",
                  md_table([{"verdict": r["verdict"], "pmid": r["pmid"], "claim": r["claim"],
                             "issue": r.get("issue")} for r in rows])]
    elif r4:
        lines += ["The argument call returned no usable JSON.", "", "```", r4["raw"], "```", ""]
    lines += [f"Comment: {ss.get('comment_q4', '')}", ""]
    lines += ["## Appendix", "", "### Prompts used", ""]
    for name, default, label in (("grade", L.GRADE_INSTRUCTIONS, "Scoring"),
                                 ("argue", L.ARGUE_INSTRUCTIONS, "Argument"),
                                 ("verify", L.VERIFY_INSTRUCTIONS, "Verifier")):
        text = ss.get(f"prompt_{name}", default)
        tag = "default" if text.strip() == default.strip() else "EDITED"
        lines += [f"#### {label} prompt ({tag})", "", "```", text, "```", ""]
    if r3 and r3["records"]:
        lines += ["### Every scored record", "", md_table(record_rows(r3["records"]))]
        for title, hits in (("Papers closest to the topic question", r3["hits_question"]),
                            ("Papers closest to the anchor", r3["hits_anchor"])):
            lines += [f"### {title}", "", md_table([{"rank": i, "similarity": f"{sim:.3f}", "pmid": d["pmid"],
                                                    "paper": f"{d['first_author']} {d['year']}", "title": d["title"]}
                                                   for i, (d, sim) in enumerate(hits, 1)])]
    lines += ["### Model and cost", "", f"- Model: {MODEL}", f"- Model calls: {ss.ledger.calls}",
              f"- Cost: {100 * ss.ledger.cost_usd:.2f} cents", ""]
    return "\n".join(lines)


def render_totals():
    with totals_box:
        if TEACHER:
            st.subheader("Spend this session")
            st.metric("Model calls", ss.ledger.calls)
            st.metric("Cost", f"{100 * ss.ledger.cost_usd:.2f} cents")
            st.progress(min(1.0, ss.ledger.cost_usd / CAP_USD), text=f"of the {CAP_CENTS} cent cap")
            with st.expander("Per move"):
                for step, u in ss.ledger.entries:
                    st.caption(f"{step} ({u.model}): {u.input_tokens:,} tokens in, {u.output_tokens:,} out, {100 * u.cost_usd:.3f} cents")
        ss.report_md = build_report()   # also readable by the headless tests
        st.download_button("Download report (.md)", ss.report_md,
                           file_name=f"module1_report_{dt.date.today()}.md", mime="text/markdown")



# ---------------------------------------------------------------- how this works

OVERVIEW = """
   1. THE CLAIM
      Start with your question, then say what you think nobody has done
      yet, and what kind of paper would prove you wrong.
            |
            v
   2. START FROM THREE PAPERS
      Pick the paper that comes closest to the gap and two others to
      compare it with. The app fetches the literature and scores those
      three first, so you see where they stand before anything else.
            |
            v
   3. GATHER EVIDENCE
      Now the app goes looking, both near your question and near your
      anchor paper, and sorts what it finds into papers that support the
      gap and papers that undermine it.
            |
            v
   4. THE ARGUMENT
      The model argues the gap in three moves. The thesis makes the case
      for it: the papers that come close but stop short of what you say is
      missing. The antithesis makes the strongest case against it: the
      papers that already do the thing, or show it would not matter. The
      synthesis says what survives the clash, usually a narrower version
      of the gap, and ends with a verdict. Then every sentence the model
      cites is checked against the abstract behind it.
            |
            v
   REPORT
      Download the whole thing. It reads like the introduction you would
      write, with your comments alongside, and that is what you hand in.
"""

with st.expander("How this works", expanded=True):
    st.code(OVERVIEW, language=None)

# ---------------------------------------------------------------- 1. Claim

st.header("1. The claim")
st.markdown(
    "A paper's introduction makes three moves. It maps the territory, it names a niche that "
    "nobody has filled, and it occupies that niche. In this move you write the first two. "
    "You also describe the paper that would prove the niche is already taken. "
    "Every later step reads these three lines."
)
for k in ("question", "gap", "disproof"):
    ss.setdefault(k, "")
QUESTION = st.text_area("Topic question (the territory)", key="question", height=80).strip()
c1, c2 = st.columns(2)
GAP = c1.text_area("The gap (the niche: what nobody has done)", key="gap", height=180).strip()
DISPROOF = c2.text_area("What would refute it (the paper that would close the gap)", key="disproof", height=180).strip()
ss.active_question = QUESTION
ss.active_gap = {"gap": GAP, "disproof": DISPROOF}
GAPCTX = ss.active_gap

# ---------------------------------------------------------------- 2. Start from three papers

st.header("2. Start from three papers")
st.markdown(
    "The writing starts from three papers. The anchor is the paper you think comes closest to "
    "the gap. The two companions are papers you want to compare it with. The app fetches the "
    "corpus, scores these three first, and outlines them on the map, so you can see where "
    "every other paper sits in relation to them."
)
source = st.radio("Corpus", ["My own PubMed query", "Shipped pool"], horizontal=True, key="source")
if source == "Shipped pool":
    pools = L.list_pools()
    if not pools:
        st.error("No pools found. Run build_pool.py first.")
        st.stop()
    POOL = st.selectbox("Pool", pools, index=pools.index(L.DEFAULT_POOL) if L.DEFAULT_POOL in pools else 0, key="pool")
    PAPERS = L.load_papers(POOL)
    st.caption(f"`{L.pool_query(POOL)}`")
    q2_sig = ("pool", POOL)
    POOL_KEY = POOL
else:
    POOL, PAPERS = None, None
    ss.setdefault("own_query", "")
    OWN_QUERY = st.text_input("PubMed query", key="own_query",
                              placeholder='("total knee arthroplasty") AND ("machine learning") AND predict*')
    OWN_EXCLUDE = True   # reviews, editorials, letters, meta-analyses are always excluded
    POOL_KEY = "custom"
default_fixed = ", ".join([PAPERS["anchor"]] + PAPERS["companions"]) if PAPERS else ""
ss.setdefault(f"fixed_{POOL_KEY}", default_fixed)
FIXED_TEXT = st.text_input(
    "Anchor paper and two companion papers (three PMIDs, comma separated; the first is the anchor)",
    key=f"fixed_{POOL_KEY}",
    placeholder="32055951, 40885690, 33337584",
)
FIXED = re.findall(r"\d{7,8}", FIXED_TEXT)[:3]
if FIXED_TEXT.strip() and len(FIXED) != 3:
    st.warning("Give exactly three PMIDs.")
if POOL is None:
    OWN_RETMAX = st.slider("Max records", 50, 300, 200, 50, key="own_retmax")
    q2_sig = ("custom", OWN_QUERY.strip(), OWN_RETMAX, OWN_EXCLUDE)
GRADE_PROMPT = sectioned_prompt_editor("grade", L.GRADE_SECTIONS, "Scoring prompt, used here and in move 3. EDIT THIS")
inputs_changed("q2", q2_sig + (tuple(FIXED), QUESTION, GAP, DISPROOF, GRADE_PROMPT))


@st.cache_resource(show_spinner=False)
def cached_corpus(pool):
    return L.build_corpus(pool)


def build_corpus_and_papers():
    """Move 2's fetch: the corpus plus the three papers. Returns True on success."""
    if POOL:
        with st.spinner("Loading corpus ..."):
            docs, V, usage, src = cached_corpus(POOL)
            if docs and not all("citations" in d for d in docs):
                L.ensure_citations(docs, L.pool_path(POOL, "abstracts.json"))
        meta = {"pool": POOL, "query": L.pool_query(POOL), "source": src, "funnel": L.pool_meta(POOL)}
    else:
        status = st.status("Building your pool ...", expanded=True)
        try:
            docs, V, usage, src, pool_id = L.build_custom_pool(
                OWN_QUERY, OWN_RETMAX, OWN_EXCLUDE, progress=status.write)
            status.update(label="Pool built", state="complete", expanded=False)
            meta = {"pool": pool_id, "query": OWN_QUERY.strip(), "retmax": OWN_RETMAX,
                    "exclude": OWN_EXCLUDE, "source": src, "funnel": L.pool_meta(pool_id)}
        except Exception as e:  # noqa: BLE001
            status.update(label="Pool build failed", state="error")
            st.error(str(e))
            return False
    if not docs:
        return False
    meta["n_pool"] = len(docs)   # before the student's own papers are appended
    with st.spinner("Adding your three papers ..."):
        docs, V, u_extra, missing = L.add_papers(docs, V, FIXED)
    if u_extra:
        ss.ledger.add("2 embed chosen papers", u_extra)
    if missing:
        st.error("These PMIDs were not found in PubMed or have no abstract: " + ", ".join(missing) + ". Pick other papers.")
        return False
    meta["fixed"] = list(FIXED)
    if usage:
        ss.ledger.add("2 embed corpus", usage)
    ss.docs, ss.V, ss.corpus_meta = docs, V, meta
    return True


if not QUESTION or not GAP:
    st.info("Write the topic question and the gap in move 1 first.")
elif POOL is None and not OWN_QUERY.strip():
    st.info("Type a PubMed query, or switch to a shipped pool.")
elif len(FIXED) != 3:
    st.info("Give the three PMIDs.")
elif run_button("Fetch and score the three", "btn_q2"):
    if build_corpus_and_papers():
        by = {d["pmid"]: d for d in ss.docs}
        with st.spinner("Scoring the three ..."):
            recs, usage, raw = L.grade_relevance(QUESTION, [by[p] for p in FIXED], client(),
                                                 system=GRADE_PROMPT, model=MODEL, gap=GAPCTX)
        ss.ledger.add("2 score three", usage)
        ss.q2 = {"records": recs, "raw": raw, "model": MODEL, "truncated": usage.truncated}

CITE_CAP = 100   # citations above this all draw at the largest dot size
STANCE_COLOURS = {"thesis": "#2e8b57", "antithesis": "#c0392b", "neutral": "#7f8c8d", "unscored": "#d5d8dc"}


@st.cache_data(show_spinner="Drawing the map ...")
def cached_map(pool_id, _vectors):
    return L.tsne_2d(_vectors)


def paper_map(docs, V, pool_id, fixed=(), stances=None):
    """Every paper as a dot, placed by t-SNE on its embedding, coloured by its
    stance (unscored papers light grey), sized by citations. Hover for the
    title; click to open PubMed. The anchor and companions carry a black
    outline and their names."""
    import altair as alt
    import pandas as pd

    stances = stances or {}
    xy = cached_map(pool_id + "|" + ",".join(fixed), V)
    cites = [d.get("citations") for d in docs]
    have_cites = any(c is not None for c in cites)
    df = pd.DataFrame({
        "x": xy[:, 0], "y": xy[:, 1],
        "PMID": [d["pmid"] for d in docs],
        "Paper": [f"{d['first_author']} {d['year']}" for d in docs],
        "Title": [d["title"] for d in docs],
        "Stance": [stances.get(d["pmid"], "unscored") for d in docs],
        "Citations": [c if c is not None else "n/a" for c in cites],
        "url": [f"https://pubmed.ncbi.nlm.nih.gov/{d['pmid']}/" for d in docs],
    })
    df["size"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).clip(upper=CITE_CAP)
    enc = dict(
        x=alt.X("x", axis=None), y=alt.Y("y", axis=None),
        color=alt.Color("Stance:N", scale=alt.Scale(domain=list(STANCE_COLOURS), range=list(STANCE_COLOURS.values())),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["PMID", "Paper", "Title", "Stance", "Citations"],
        href="url:N",
    )
    if have_cites:
        enc["size"] = alt.Size("size:Q", scale=alt.Scale(domain=[0, CITE_CAP], range=[25, 500]), legend=None)
    base = alt.Chart(df)
    chart = base.transform_filter(alt.datum.Stance == "unscored").mark_circle(opacity=0.45).encode(**enc)
    chart += base.transform_filter(alt.datum.Stance != "unscored").mark_circle(opacity=0.9).encode(**enc)
    red = df[df["PMID"].isin(fixed)]
    if len(red):
        dots = alt.Chart(red).mark_circle(stroke="black", strokeWidth=2, opacity=1).encode(
            x="x", y="y", color=enc["color"], tooltip=enc["tooltip"], href="url:N",
            **({"size": enc["size"]} if "size" in enc else {}))
        labels = alt.Chart(red).mark_text(dy=-14, fontWeight="bold").encode(x="x", y="y", text="Paper")
        chart = chart + dots + labels
    # Rendered through Vega-Embed in an HTML component rather than st.altair_chart:
    # Streamlit drops the loader option, and without target=_blank a click would
    # try to load PubMed inside the page, which PubMed refuses.
    spec = chart.properties(width="container", height=420, background="transparent").interactive().to_json()
    doc = f"""
<div id="vis" style="width:100%"></div>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@6"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<script>
vegaEmbed("#vis", {spec}, {{actions: false, loader: {{target: "_blank"}}}});
</script>"""
    embed_html(doc, 460)
    if not have_cites:
        st.warning("Citation counts are unavailable because iCite could not be reached."
                   + (f" {L.LAST_ICITE_ERROR}" if L.LAST_ICITE_ERROR else ""))


def stances_of(records):
    return {r["pmid"]: L.stance_of(r) or "unscored" for r in records or []}


if "docs" not in ss or "q2" not in ss:
    st.info("Fetch and score the three papers to unlock the other moves.")
    render_rest = False
else:
    render_rest = True
    docs, V, meta = ss.docs, ss.V, ss.corpus_meta
    by = {d["pmid"]: d for d in docs}
    ANCHOR, COMPANIONS = meta["fixed"][0], meta["fixed"][1:]
    st.markdown(funnel_sentence(meta) or f"The corpus has {len(docs)} documents.")
    if ss.q2["records"] is None:
        grading_failed(ss.q2)
    else:
        stance_panels(ss.q2["records"], anchor=ANCHOR)
    # the map shows the evidence stances once move 3 has run, else just the three
    scored = ss.q3["records"] if ss.get("q3") and ss.q3["records"] else ss.q2["records"]
    paper_map(docs, V, meta["pool"], fixed=meta["fixed"], stances=stances_of(scored))
    if ss.q2["records"]:
        with st.expander("The three records in full"):
            show_records(ss.q2)
    comment_box("q2")

if not render_rest:
    render_totals()
    st.stop()

# ---------------------------------------------------------------- 3. Gather evidence

st.header("3. Gather evidence")
st.markdown(
    f"The app runs two searches. One finds the {EVIDENCE_K} papers closest to your topic question. "
    f"The other finds the {EVIDENCE_K} papers closest to the anchor. Every new paper is scored in a "
    "single call and joins the three you started with. A green panel is a paper that argues "
    "for the gap. A red panel is a paper that argues against it. Hover over a panel to read "
    "the title and the reason, and click it to open the paper on PubMed."
)
inputs_changed("q3", (QUESTION, GAP, DISPROOF, ANCHOR, GRADE_PROMPT))
if ss.q2["records"] is None:
    st.info("Move 2 returned no usable records. Fix the scoring prompt and score the three again.")
elif run_button("Find evidence", "btn_q3"):
    with st.spinner("Searching ..."):
        hits_q, u1 = L.search(QUESTION, docs, V, k=EVIDENCE_K)
        ss.ledger.add("3 embed question", u1)
        hits_a = L.nearest(ANCHOR, docs, V, k=EVIDENCE_K)
    seen = set(meta["fixed"])
    new_docs = []
    for d, _ in hits_q + hits_a:
        if d["pmid"] not in seen:
            seen.add(d["pmid"])
            new_docs.append(d)
    with st.spinner(f"Scoring {len(new_docs)} papers ..."):
        recs, u2, raw = L.grade_relevance(QUESTION, new_docs, client(), system=GRADE_PROMPT, model=MODEL, gap=GAPCTX)
    ss.ledger.add("3 score evidence", u2)
    ss.q3 = {"records": (ss.q2["records"] + recs) if recs is not None else None, "new": recs, "raw": raw,
             "model": MODEL, "truncated": u2.truncated, "hits_question": hits_q, "hits_anchor": hits_a,
             "n_new": len(new_docs)}
    st.rerun()   # so the map above redraws with the new stances
if "q3" in ss:
    r3 = ss.q3
    if r3["records"] is None:
        grading_failed(r3)
    else:
        stance_panels(r3["records"], anchor=ANCHOR)
        st.caption(f"{len(r3['records'])} papers have been scored, the three you started with and {r3['n_new']} "
                   f"found by the two searches. " + L.funnel_line(meta.get("funnel"), meta.get("n_pool"), len(r3["records"])))
        with st.expander("The two searches and every record in full"):
            c1, c2 = st.columns(2)
            c1.markdown("**Closest to the topic question**")
            for d, s in r3["hits_question"]:
                c1.markdown(paper_line(d, s, tag="anchor" if d["pmid"] == ANCHOR else ""))
            c2.markdown("**Closest to the anchor**")
            for d, s in r3["hits_anchor"]:
                c2.markdown(paper_line(d, s))
            show_records(r3)
    comment_box("q3")

# ---------------------------------------------------------------- 4. Argue

st.header("4. The argument")
st.markdown(
    "The model now writes the argument from the thesis and antithesis papers. Neutral papers "
    "are left out, and antithesis papers go in with their full abstract. The argument has three "
    "parts. The thesis states the gap and the papers whose own limits point at it. The "
    "antithesis makes the strongest possible case against the gap. The synthesis says what "
    "survives, either a narrower gap or a concession, and ends with a verdict. "
    "After that, every cited sentence goes back to the model together with the one abstract it "
    "cites, and comes back marked supported, not supported, or can't tell."
)
with st.expander("Argument prompt. EDIT THIS", expanded=not TEACHER):
    ARGUE_PROMPT = sectioned_prompt_editor("argue", SECTIONS["argue"], "The argument prompt, one box per part")
with st.expander("Verifier prompt. EDIT THIS", expanded=False):
    VER_PROMPT = sectioned_prompt_editor("verify", SECTIONS["verify"], "The verifier prompt, one box per part")
inputs_changed("q4", (ARGUE_PROMPT, VER_PROMPT))
if "q3" not in ss or ss.q3["records"] is None:
    st.info("Gather the evidence first. The argument is built from its records.")
elif run_button("Write the argument and verify", "btn_q4"):
    kept, had_stances = L.select_records(ss.q3["records"])
    if not had_stances:
        st.warning("No record carries a score or a `bears` field, so every paper goes into the argument.")
    if not kept:
        st.warning("Every paper is neutral, so there is nothing to argue from.")
    else:
        funnel_text = L.funnel_line(meta.get("funnel"), meta.get("n_pool"), len(ss.q3["records"]))
        with st.spinner("Writing the argument ..."):
            out, u1, raw = L.argue(QUESTION, GAPCTX, kept, funnel_text, client(), system=ARGUE_PROMPT, model=MODEL)
        ss.ledger.add("4 argue", u1)
        rows = []
        if out is not None:
            bar = st.progress(0.0, text="Verifying claims ...")
            rows, u2 = L.verify_claims(L.argument_text(out), docs, client(), system=VER_PROMPT, model=MODEL,
                                       progress=lambda i, n: bar.progress(i / n, text=f"Verifying claim {i} of {n}"))
            bar.empty()
            ss.ledger.add("4 verify", u2)
        ss.q4 = {"argument": out, "raw": raw, "rows": rows, "model": MODEL, "kept": kept,
                 "truncated": u1.truncated, "funnel": funnel_text}
if "q4" in ss:
    r4 = ss.q4
    if r4["argument"] is None:
        st.warning("The reply was cut off at the model's output limit." if r4.get("truncated") else
                   "The reply was not a JSON object with thesis, antithesis, synthesis and verdict. "
                   "Read it as text, or fix the prompt and run again.")
        st.code(r4["raw"])
    else:
        out = r4["argument"]
        st.caption(f"Written and verified by {r4['model']} from {len(r4['kept'])} papers. {r4['funnel']}")
        for part in ("thesis", "antithesis", "synthesis"):
            st.markdown(f"**{part.capitalize()}.** {out[part]}")
        st.subheader(f"Verdict: {out['verdict']}")
        rows = r4["rows"]
        counts = {v: sum(r["verdict"] == v for r in rows) for v in L.CLAIM_VERDICTS}
        st.markdown(f"**{verification_sentence(rows)}**")
        st.dataframe(
            [{"Verdict": r["verdict"], "PMID": r["pmid"], "Claim": r["claim"], "Issue": r.get("issue")} for r in rows],
            width="stretch", hide_index=True,
        )
        with st.expander("Raw model reply"):
            st.code(r4["raw"])
    comment_box("q4")

st.divider()
st.markdown("When you are done, download the report from the sidebar. It contains the argument in the "
            "order you would write it, together with the evidence, the verification and your comments. "
            "That file is your homework. Rename it `nameA_idA_nameB_idB.md`, with the names and student "
            "IDs of both of you, and upload it here: "
            "[Dropbox file request](https://www.dropbox.com/request/hyjoadqw1wlpcr0shtjw).")



render_totals()
