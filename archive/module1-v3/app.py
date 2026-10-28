"""
Module 1 v3: build your own literature review.

Everything a student can change is a widget: the paper pool, the research
question, the anchor paper, every system prompt, the search strategy, the
model, and what goes into the review. A cost cap in the sidebar keeps the
bill small. A comment box under each step and a Download button turn the
session into the homework report.

Run:  streamlit run app.py
"""

import datetime as dt
import json
import math
import os
import re
from collections import Counter

import streamlit as st
from dotenv import load_dotenv

import lit_helper as L

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Community Cloud keeps secrets in st.secrets; mirror them into the environment
# so the SDKs pick them up the same way as from .env.
for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NCBI_API_KEY", "NCBI_EMAIL"):
    try:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]
    except Exception:
        pass

st.set_page_config(page_title="Module 1: literature review", layout="wide")
st.title("Module 1 v3: build your own literature review")

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
    </style>""",
    unsafe_allow_html=True,
)

ss = st.session_state
ss.setdefault("ledger", L.Ledger())
ss.setdefault("client", None)

STEPS = ["q1", "q2", "q3", "q4", "q5"]
RESULT_KEYS = {
    "q1": ["docs", "V", "corpus_meta"],
    "q2": ["q2"],
    "q3": ["q3"],
    "q4": ["q4"],
    "q5": ["q5"],
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
        st.error(f"Cost cap reached: spent {100 * spent:.1f} cents of "
                 f"{CAP_CENTS}. Raise the slider in the sidebar to continue.")
        return False
    return True


def prompt_editor(name, default, label):
    """Full system prompt in a text area with a Reset button. Returns the text.
    Same mechanism as sectioned_prompt_editor: the box takes its text from a
    seed through value=, under a versioned key. Writing to a keyed widget's
    session-state entry from elsewhere (a question set, Reset) is not shown by
    the browser, so those write the seed and bump the version instead."""
    seed = ss.setdefault(f"{name}_seed", default)
    ver = ss.setdefault(f"{name}_ver", 0)
    with st.expander(label, expanded=False):
        if st.button("Reset to default", key=f"reset_{name}"):
            ss[f"{name}_seed"] = default
            ss[f"{name}_ver"] = ver + 1
            st.rerun()
        ss[f"prompt_{name}"] = st.text_area("System prompt", value=seed, key=f"w_{name}_{ver}",
                                            height=280, label_visibility="collapsed")
        if ss[f"prompt_{name}"].strip() != default.strip():
            st.caption("Edited. The report will include your version.")
    return ss[f"prompt_{name}"]


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
    c1.caption("The boxes below are joined, in order, into the system prompt. "
               + (f"Edited: {', '.join(edited)}." if edited else "All default."))
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
    with st.expander("Combined prompt as sent", expanded=False):
        st.code(combined, language="text")
    return combined


def comment_box(step):
    st.text_area("Comment", key=f"comment_{step}", height=80)


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


def stance_panels(records, anchor=None):
    html_doc, height = L.stance_panels_html(records, anchor)
    st.components.v1.html(html_doc, height=height)


def show_grades(result):
    """Render a grading result: table of whatever came back, or the raw text."""
    records, raw = result["records"], result["raw"]
    st.caption(f"Graded by {result['model']}")
    if records is None:
        st.warning("The reply was cut off at the model's output limit, so the JSON is incomplete and there "
                   "is no table. Shorten what the grading prompt asks for per paper and grade again."
                   if result.get("truncated") else
                   "The reply was not a JSON array of objects, so there is no table. "
                   "Read the answer as text, or fix the prompt and grade again.")
        st.code(raw)
        return
    st.dataframe(record_rows(records), width="stretch", hide_index=True)
    for r in records:
        d = r["doc"]
        with st.expander(f"{d['pmid']} {L.stance_of(r) or ''}: {d['first_author']} {d['year']}, full record and abstract"):
            for k, v in r.items():
                if k not in ("doc", "pmid", "quote_found"):
                    st.write(f"**{k}:** {v}")
            if "quote_found" in r and not r["quote_found"]:
                st.warning("The evidence quote was not found in the abstract. Read it and judge for yourself.")
            st.caption(d["abstract"])
    with st.expander("Raw model reply"):
        st.code(raw)


def md_table(rows):
    if not rows:
        return "_empty_\n"
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- sidebar: controls first, totals at the end

ALLOW_SONNET = False   # flip to True to let students pick Sonnet 5.5

# A question set prefills the own-query mode so the demo runs with no typing:
# question, gap, disproof, PubMed query, the three PMIDs, and optionally edited
# prompts. One JSON file per set in question_sets/; pick one at the top of Q0.
QSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "question_sets")
DEFAULT_QSET = "tkr_prediction.json"
PROMPT_DEFAULTS = {"synth": L.SYNTHESIS_INSTRUCTIONS, "verify": L.VERIFY_INSTRUCTIONS, "gap": L.GAP_INSTRUCTIONS}
GRADE_DEFAULTS = {key: text for key, _, text in L.GRADE_SECTIONS}   # the grade prompt is saved per section


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
    for name, default in PROMPT_DEFAULTS.items():
        ss[f"{name}_seed"] = prompts.get(name, default)
        ss[f"{name}_ver"] = ss.get(f"{name}_ver", 0) + 1
        ss[f"prompt_{name}"] = ss[f"{name}_seed"]
    grade = prompts.get("grade") or {}
    if isinstance(grade, str):   # a whole prompt as one string: first box gets it, the rest go blank
        grade = {key: (grade if key == L.GRADE_SECTIONS[0][0] else "") for key in GRADE_DEFAULTS}
    ss.grade_seed = {key: grade.get(key, default) for key, default in GRADE_DEFAULTS.items()}
    ss.grade_ver = ss.get("grade_ver", 0) + 1
    for key, text in ss.grade_seed.items():
        ss[f"prompt_grade_{key}"] = text
    ss.prompt_grade = L.join_sections((title, ss.grade_seed[key]) for key, title, _ in L.GRADE_SECTIONS)


def on_pick_qset():
    apply_qset(load_qset(ss.qset))


def on_upload_qset():
    f = ss.get("qset_upload")
    if f is None:
        return
    try:
        apply_qset(json.loads(f.getvalue()))
        ss.qset_error = None
    except (ValueError, AttributeError) as e:
        ss.qset_error = f"Could not read {f.name}: {e}"


def current_qset():
    """The current text fields as a question set. Prompts are saved only when edited."""
    qs = {"name": ss.get("question", "")[:80],
          "question": ss.get("question", ""), "gap": ss.get("gap", ""), "disproof": ss.get("disproof", ""),
          "query": ss.get("own_query", ""),
          "pmids": re.findall(r"\d{7,8}", ss.get("fixed_custom", ""))[:3]}
    edited = {n: ss[f"prompt_{n}"] for n, d in PROMPT_DEFAULTS.items()
              if ss.get(f"prompt_{n}", d).strip() != d.strip()}
    grade = {k: ss[f"prompt_grade_{k}"] for k, d in GRADE_DEFAULTS.items()
             if ss.get(f"prompt_grade_{k}", d).strip() != d.strip()}
    if grade:
        edited["grade"] = grade
    if edited:
        qs["prompts"] = edited
    return qs


if "qset" not in ss:
    sets = list_qsets()
    ss.qset = DEFAULT_QSET if DEFAULT_QSET in sets else (sets[0] if sets else None)
    if ss.qset:
        apply_qset(load_qset(ss.qset))

with st.sidebar:
    st.subheader("Model")
    if ALLOW_SONNET:
        model_label = st.radio(
            "Chat model", list(L.MODELS), key="model_label",
            captions=["$0.10 in / $0.50 out per M tokens", "$3 in / $15 out per M tokens, 30x Haiku"],
        )
    else:
        model_label = "Haiku 5.5"
        st.radio("Chat model", ["Haiku 5.5"], key="model_label", captions=["$0.10 in / $0.50 out per M tokens"])
        st.toggle("Sonnet 5.5, $3 in / $15 out, 30x Haiku", value=False, disabled=True)
    MODEL = L.MODELS[model_label]
    st.subheader("Cost cap")
    CAP_CENTS = st.slider("Stop making calls after (cents)", 1, 50, 10, key="cap_cents")
    CAP_USD = CAP_CENTS / 100
    totals_box = st.container()

# ---------------------------------------------------------------- Q0

st.header("Q0. Define the gap")
st.markdown(
    "A review starts with a question and a claim about what nobody has done yet. "
    "Write both, and say what kind of paper would prove the claim wrong. The grader, "
    "the synthesis and the final verdict in Q5 all read these three lines."
)
c1, c2, c3 = st.columns([2, 2, 1])
c1.selectbox("Question set", list_qsets(), key="qset", on_change=on_pick_qset,
             format_func=lambda f: f"{load_qset(f).get('name', f)} ({f})")
c2.file_uploader("Or load a question set (.json)", type="json", key="qset_upload", on_change=on_upload_qset)
c3.download_button("Save current set (.json)", json.dumps(current_qset(), indent=1, ensure_ascii=False),
                   file_name="question_set.json", mime="application/json")
if ss.get("qset_error"):
    st.error(ss.qset_error)
for k in ("question", "gap", "disproof"):
    ss.setdefault(k, "")
QUESTION = st.text_area("Research question", key="question", height=80).strip()
c1, c2 = st.columns(2)
GAP = c1.text_area("Proposed gap (what nobody has done)", key="gap", height=180).strip()
DISPROOF = c2.text_area("What evidence would disprove it", key="disproof", height=180).strip()
ss.active_question = QUESTION
ss.active_gap = {"gap": GAP, "disproof": DISPROOF}
GAPCTX = ss.active_gap


def use_pool_question(papers):
    """Button callback: copy a shipped pool's question and gap into Q0."""
    ss.question = papers.get("question", "")
    ss.gap = papers.get("gap", "")
    ss.disproof = papers.get("disproof", "")


# ---------------------------------------------------------------- Q1

st.header("Q1. Build corpus")
st.markdown(
    "A corpus is a set of PubMed abstracts, each turned into a 1,536-number vector "
    "with `text-embedding-3-small`."
)
source = st.radio("Source", ["My own PubMed query", "Shipped pool"], horizontal=True, key="source")
if source == "Shipped pool":
    pools = L.list_pools()
    if not pools:
        st.error("No pools found. Run build_pool.py first.")
        st.stop()
    POOL = st.selectbox("Pool", pools, index=pools.index(L.DEFAULT_POOL) if L.DEFAULT_POOL in pools else 0, key="pool")
    PAPERS = L.load_papers(POOL)
    st.caption(f"`{L.pool_query(POOL)}`")
    st.caption(f"Built for: _{PAPERS['question']}_")
    st.button("Use this pool's question and gap in Q0", key="use_pool_q", on_click=use_pool_question, args=(PAPERS,))
    q1_sig = ("pool", POOL)
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
    q1_sig = ("custom", OWN_QUERY.strip(), OWN_RETMAX, OWN_EXCLUDE)
q1_sig = q1_sig + (tuple(FIXED),)
inputs_changed("q1", q1_sig)


@st.cache_resource(show_spinner=False)
def cached_corpus(pool):
    return L.build_corpus(pool)


if run_button("Build corpus", "btn_q1"):
    if POOL:
        with st.spinner("Loading corpus ..."):
            docs, V, usage, src = cached_corpus(POOL)
            if docs and not all("citations" in d for d in docs):
                L.ensure_citations(docs, L.pool_path(POOL, "abstracts.json"))
        meta = {"pool": POOL, "query": L.pool_query(POOL), "source": src, "funnel": L.pool_meta(POOL)}
    elif not OWN_QUERY.strip():
        st.warning("Type a PubMed query first.")
        docs = None
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
            docs = None
    if docs:
        meta["n_pool"] = len(docs)   # before the student's own papers are appended
    if docs and len(FIXED) == 3:
        with st.spinner("Adding your three papers ..."):
            docs, V, u_extra, missing = L.add_papers(docs, V, FIXED)
        if u_extra:
            ss.ledger.add("Q1 embed chosen papers", u_extra)
        if missing:
            st.warning("Not found in PubMed, or no abstract: " + ", ".join(missing) + ". They are left out.")
        meta["fixed"] = [p for p in FIXED if p not in missing]
    if docs:
        if usage:
            ss.ledger.add("Q1 embed", usage)
        ss.docs, ss.V, ss.corpus_meta = docs, V, meta

CITE_CAP = 100   # citations above this all draw at the largest dot size


@st.cache_data(show_spinner="Drawing the map ...")
def cached_map(pool_id, _vectors):
    return L.tsne_2d(_vectors)


def paper_map(docs, V, pool_id, fixed=()):
    """Every paper as a dot, placed by t-SNE on its embedding. Hover for the
    title. The anchor and companions are drawn in red with their names."""
    import altair as alt
    import pandas as pd

    xy = cached_map(pool_id + "|" + ",".join(fixed), V)
    cites = [d.get("citations") for d in docs]
    have_cites = any(c is not None for c in cites)
    df = pd.DataFrame({
        "x": xy[:, 0], "y": xy[:, 1],
        "PMID": [d["pmid"] for d in docs],
        "Paper": [f"{d['first_author']} {d['year']}" for d in docs],
        "Title": [d["title"] for d in docs],
        "Year": [d.get("year") for d in docs],
        "Citations": [c if c is not None else "n/a" for c in cites],
        "url": [f"https://pubmed.ncbi.nlm.nih.gov/{d['pmid']}/" for d in docs],
    })
    df["size"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).clip(upper=CITE_CAP)
    enc = dict(
        x=alt.X("x", axis=None), y=alt.Y("y", axis=None),
        color=alt.Color("Year:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["PMID", "Paper", "Title", "Citations"],
        href="url:N",
    )
    if have_cites:
        enc["size"] = alt.Size("size:Q", scale=alt.Scale(domain=[0, CITE_CAP], range=[25, 500]), legend=None)
    chart = alt.Chart(df).mark_circle(opacity=0.75).encode(**enc)
    red = df[df["PMID"].isin(fixed)]
    if len(red):
        dots = alt.Chart(red).mark_circle(color="red", stroke="black", strokeWidth=1.5, opacity=1).encode(
            x="x", y="y", tooltip=["PMID", "Paper", "Title", "Citations"], href="url:N",
            **({"size": enc["size"]} if "size" in enc else {}))
        labels = alt.Chart(red).mark_text(dy=-14, color="red", fontWeight="bold").encode(
            x="x", y="y", text="Paper")
        chart = chart + dots + labels
    # Rendered through Vega-Embed in an HTML component rather than st.altair_chart:
    # Streamlit drops the loader option, and without target=_blank a click would
    # try to load PubMed inside the page, which PubMed refuses.
    spec = chart.properties(width="container", height=420, background="transparent").interactive().to_json()
    html = f"""
<div id="vis" style="width:100%"></div>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@6"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<script>
vegaEmbed("#vis", {spec}, {{actions: false, loader: {{target: "_blank"}}}});
</script>"""
    st.components.v1.html(html, height=440)
    if not have_cites:
        st.warning("Citation counts unavailable (iCite unreachable)."
                   + (f" {L.LAST_ICITE_ERROR}" if L.LAST_ICITE_ERROR else ""))


if "docs" not in ss:
    st.info("Build the corpus to unlock the other steps.")
else:
    docs, V = ss.docs, ss.V
    meta = ss.corpus_meta
    years = sorted({d["year"] for d in docs if d.get("year")})
    st.success(f"{len(docs)} documents, vectors {V.shape[0]} x {V.shape[1]}, loaded from {meta['source']}. "
               f"Years {years[0]} to {years[-1]}.")
    if funnel_sentence(meta):
        st.markdown(funnel_sentence(meta))
    paper_map(docs, V, meta["pool"], fixed=meta.get("fixed", ()))


def render_totals():
    with totals_box:
        st.subheader("Spend this session")
        st.metric("Model calls", ss.ledger.calls)
        st.metric("Cost", f"{100 * ss.ledger.cost_usd:.2f} cents")
        st.progress(min(1.0, ss.ledger.cost_usd / CAP_USD), text=f"of the {CAP_CENTS} cent cap")
        with st.expander("Per step"):
            for step, u in ss.ledger.entries:
                st.caption(f"{step} ({u.model}): {u.input_tokens} in / {u.output_tokens} out, {100 * u.cost_usd:.3f}c")
        ss.report_md = build_report()   # also readable by the headless tests
        st.download_button("Download report (.md)", ss.report_md,
                           file_name=f"module1_report_{dt.date.today()}.md", mime="text/markdown")


def build_report():
    lines = [f"# Module 1 literature review report", f"Date: {dt.date.today()}", ""]
    gap = ss.get("active_gap", {})
    lines += ["## Q0. Question and gap", "",
              f"- Research question: {ss.get('active_question', '')}",
              f"- Proposed gap: {gap.get('gap', '')}",
              f"- What would disprove it: {gap.get('disproof', '')}", ""]
    meta = ss.get("corpus_meta")
    lines += ["## Q1. Corpus", ""]
    if meta:
        lines += [f"- Pool: `{meta['pool']}`", f"- PubMed query: `{meta['query']}`",
                  f"- Documents: {len(ss.docs)}",
                  f"- {L.funnel_line(meta.get('funnel'), meta.get('n_pool'))}",
                  f"- Anchor and companions: {', '.join(meta.get('fixed', [])) or 'not given'}", ""]
    for step, title in (("q2", "Q2. Grade three fixed papers"), ("q3", "Q3. Search the corpus")):
        lines += [f"## {title}", ""]
        res = ss.get(step)
        if res:
            for k, v in res.get("settings", {}).items():
                lines.append(f"- {k}: {v}")
            lines.append(f"- model: {res['model']}")
            lines.append("")
            if res.get("hits"):
                lines += ["Ranking:", "", md_table([{"rank": i, "score": f"{s:.3f}", "pmid": d["pmid"],
                                                    "paper": f"{d['first_author']} {d['year']}"}
                                                   for i, (d, s) in enumerate(res["hits"], 1)])]
            if res["records"] is None:
                lines += ["Raw reply (not JSON):", "", "```", res["raw"], "```", ""]
            else:
                lines += [md_table(record_rows(res["records"]))]
        lines += [f"Comment: {ss.get(f'comment_{step}', '')}", ""]
    lines += ["## Q4. Synthesis and verification", ""]
    res = ss.get("q4")
    if res:
        for k, v in res["settings"].items():
            lines.append(f"- {k}: {v}")
        lines += [f"- model: {res['model']}", "", res["review"], "",
                  md_table([{"verdict": r["verdict"], "pmid": r["pmid"], "claim": r["claim"],
                             "issue": r.get("issue")} for r in res["rows"]])]
    lines += [f"Comment: {ss.get('comment_q4', '')}", ""]
    lines += ["## Q5. Gap assessment", ""]
    res = ss.get("q5")
    if res:
        for k, v in res["settings"].items():
            lines.append(f"- {k}: {v}")
        lines += [f"- model: {res['model']}", ""]
        out = res["result"]
        if out is None:
            lines += ["Raw reply (not JSON):", "", "```", res["raw"], "```", ""]
        else:
            cx = out.get("counterexamples") or []
            lines += [f"**Verdict: {out.get('verdict', '?')}**", "",
                      md_table([{"pmid": c.get("pmid"), "why": c.get("why")} for c in cx if isinstance(c, dict)]),
                      f"Remaining uncertainty: {out.get('uncertainty', '')}", "",
                      f"Search limits: {out.get('search_limits', '')}", ""]
    lines += [f"Comment: {ss.get('comment_q5', '')}", ""]
    lines += ["## Prompts used", ""]
    for name, default, label in (("grade", L.GRADE_INSTRUCTIONS, "Grading"),
                                 ("synth", L.SYNTHESIS_INSTRUCTIONS, "Synthesis"),
                                 ("verify", L.VERIFY_INSTRUCTIONS, "Verifier"),
                                 ("gap", L.GAP_INSTRUCTIONS, "Gap")):
        text = ss.get(f"prompt_{name}", default)
        tag = "default" if text.strip() == default.strip() else "EDITED"
        lines += [f"### {label} prompt ({tag})", "", "```", text, "```", ""]
    lines += ["## Cost", "", f"- Model calls: {ss.ledger.calls}",
              f"- Cost: {100 * ss.ledger.cost_usd:.2f} cents", ""]
    return "\n".join(lines)


if "docs" not in ss:
    render_totals()
    st.stop()

by = {d["pmid"]: d for d in docs}
pmids = [d["pmid"] for d in docs]

# ---------------------------------------------------------------- Q2

st.header("Q2. Grade three fixed papers")
st.markdown(
    "The model scores three papers from 1 to 10 for what they say about the gap you wrote in Q0: "
    "the **anchor** and two **companions** you named in Q1. 1 means the paper's own study does "
    "the work or shows the premise wrong; 10 means it names the gap as undone. The stance is "
    "computed from the number, not asked of the model: **antithesis** at 5 or below, **thesis** "
    "above, **neutral** when the paper does not bear on the question. The grading prompt below "
    "defines the scale and the return format. Edit it freely: the table shows whatever keys the "
    "model returns, plus the stance, the gap relation computed from the condition checks, and a "
    "flag where the score and the checks disagree."
)
fixed_in_pool = [p for p in meta.get("fixed", []) if p in by]
if len(fixed_in_pool) == 3:
    ANCHOR, COMPANIONS = fixed_in_pool[0], fixed_in_pool[1:]
    st.caption("Anchor and companions come from the PMIDs you typed in Q1.")
else:
    st.caption("No PMIDs were given in Q1, so pick an anchor here; the companions are two random papers (seed 42).")
    default_anchor = PAPERS["anchor"] if PAPERS and PAPERS["anchor"] in by else pmids[0]
    ANCHOR = st.selectbox("Anchor paper", pmids, index=pmids.index(default_anchor),
                          format_func=lambda p: paper_label(by[p]), key=f"anchor_{POOL_KEY}")
    COMPANIONS = L.companions(docs, ANCHOR)
GRADE_PROMPT = sectioned_prompt_editor("grade", L.GRADE_SECTIONS, "Grading prompt, shared by Q2 and Q3 (edit me)")
inputs_changed("q2", (QUESTION, GAP, DISPROOF, ANCHOR, tuple(COMPANIONS), GRADE_PROMPT))

fixed = [by[ANCHOR]] + [by[p] for p in COMPANIONS]
for d, tag in zip(fixed, ["anchor", "companion", "companion"]):
    st.markdown(paper_line(d, tag=tag))
if not QUESTION:
    st.info("Type a research question in Q0 first.")
elif run_button("Grade", "btn_q2"):
    with st.spinner("Grading ..."):
        recs, usage, raw = L.grade_relevance(QUESTION, fixed, client(), system=GRADE_PROMPT, model=MODEL, gap=GAPCTX)
    ss.ledger.add("Q2 grade", usage)
    ss.q2 = {"records": recs, "raw": raw, "model": MODEL, "truncated": usage.truncated,
             "settings": {"question": QUESTION, "anchor": ANCHOR, "companions": ", ".join(COMPANIONS)}}
if "q2" in ss:
    show_grades(ss.q2)
    comment_box("q2")

# ---------------------------------------------------------------- Q3

st.header("Q3. Search the corpus")
st.markdown(
    "Two ways to pull papers out of the corpus. **Embedding**: the question itself does "
    "the retrieving, by cosine similarity between its vector and every paper's. "
    "**Anchor's neighbours**: start from the paper you trust and take the papers nearest "
    "to it, the snowballing a reviewer does by hand. Either way the top 10 are graded."
)
K3 = 10
STRATEGY = st.radio("Strategy", ["Embedding", "Anchor's neighbours"], horizontal=True, key="strategy")
NEIGHBOURS = STRATEGY == "Anchor's neighbours"
inputs_changed("q3", (QUESTION, ANCHOR, STRATEGY, GRADE_PROMPT))
if QUESTION and run_button("Search", "btn_q3"):
    with st.spinner("Searching and grading ..."):
        if NEIGHBOURS:
            hits = L.nearest(ANCHOR, docs, V, k=K3)
        else:
            hits, u1 = L.search(QUESTION, docs, V, k=K3)
            ss.ledger.add("Q3 embed", u1)
        recs, u2, raw = L.grade_relevance(QUESTION, [d for d, _ in hits], client(), system=GRADE_PROMPT,
                                          model=MODEL, gap=GAPCTX)
        ss.ledger.add("Q3 grade", u2)
    ss.q3 = {"records": recs, "raw": raw, "model": MODEL, "hits": hits, "truncated": u2.truncated,
             "settings": {"strategy": STRATEGY, "k": K3, **({"anchor": ANCHOR} if NEIGHBOURS else {})}}
if "q3" in ss:
    r3 = ss.q3
    if r3["settings"]["strategy"] == "Embedding":
        rank = L.rank_of(ANCHOR, r3["hits"])
        st.markdown(f"Anchor rank among the {len(r3['hits'])}: **{rank if rank else 'not in the list'}**")
    else:
        st.markdown(f"**{len(r3['hits'])} nearest to the anchor**")
    if r3["records"]:
        stance_panels(r3["records"], anchor=ANCHOR)
    with st.expander("Ranked hits and the full grading table"):
        for d, s in r3["hits"]:
            st.markdown(paper_line(d, s, tag="anchor" if d["pmid"] == ANCHOR else ""))
        show_grades(r3)
    comment_box("q3")

# ---------------------------------------------------------------- Q4

st.header("Q4. Synthesis")
st.markdown(
    "The model argues the gap as a dialectic from the thesis and antithesis papers, citing PMIDs: "
    "**thesis**, the gap and the papers whose own limits point at it; **antithesis**, the "
    "strongest counterexamples at their strongest; **synthesis**, the narrowest gap that "
    "survives, or a concession. Antithesis papers go in with their full abstract. "
    "Then every cited sentence goes back to the model with only the abstract it cites, "
    "and is marked supported, partial, insufficient (the abstract says nothing about it), "
    "or unsupported (the abstract contradicts it)."
)
SYN_PROMPT = prompt_editor("synth", L.SYNTHESIS_INSTRUCTIONS, "Synthesis prompt (edit me)")
VER_PROMPT = prompt_editor("verify", L.VERIFY_INSTRUCTIONS, "Verifier prompt (edit me)")
inputs_changed("q4", (SYN_PROMPT, VER_PROMPT))
if "q3" not in ss or ss.q3["records"] is None:
    st.info("Run Q3 with a prompt that returns JSON first. The synthesis is built from its records.")
elif run_button("Write the synthesis and verify", "btn_q4"):
    kept, had_stances = L.select_records(ss.q3["records"])
    if not had_stances:
        st.warning("No record carries a score or a `bears` field, so every Q3 paper goes into the synthesis.")
    if not kept:
        st.warning("Every paper is neutral, so there is nothing to argue from.")
    else:
        with st.spinner("Writing the synthesis ..."):
            review, u1 = L.synthesize(QUESTION, kept, client=client(), system=SYN_PROMPT, model=MODEL, gap=GAPCTX)
        ss.ledger.add("Q4 synthesize", u1)
        bar = st.progress(0.0, text="Verifying claims ...")
        rows, u2 = L.verify_claims(review, docs, client(), system=VER_PROMPT, model=MODEL,
                                   progress=lambda i, n: bar.progress(i / n, text=f"Verifying claim {i} of {n}"))
        bar.empty()
        ss.ledger.add("Q4 verify", u2)
        clear_from("q5")   # a new review invalidates the verdict built on the old one
        ss.q4 = {"review": review, "rows": rows, "model": MODEL, "kept": kept,
                 "settings": {"papers": len(kept)}}
if "q4" in ss:
    st.caption(f"Written and verified by {ss.q4['model']} from {ss.q4['settings']['papers']} papers")
    st.markdown(ss.q4["review"])
    rows = ss.q4["rows"]
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("supported", "partial", "insufficient", "unsupported")}
    st.markdown(f"**{len(rows)} claim checks: {counts['supported']} supported, {counts['partial']} partial, "
                f"{counts['insufficient']} insufficient, {counts['unsupported']} unsupported**")
    st.dataframe(
        [{"Verdict": r["verdict"], "PMID": r["pmid"], "Claim": r["claim"], "Issue": r.get("issue")} for r in rows],
        width="stretch", hide_index=True,
    )
    comment_box("q4")

# ---------------------------------------------------------------- Q5

st.header("Q5. Is the gap still open?")
st.markdown(
    "One call weighs the gap you wrote in Q0 against the strongest papers, the verified "
    "review, and the search funnel. Four verdicts: **contradicted**, **partly addressed**, "
    "**provisionally supported**, **insufficient evidence**. Read the counterexamples "
    "yourself before you believe the verdict."
)
GAP_PROMPT = prompt_editor("gap", L.GAP_INSTRUCTIONS, "Gap prompt (edit me)")
inputs_changed("q5", (GAP_PROMPT,))
if "q4" not in ss:
    st.info("Run Q4 first. The verdict is built from its review and verification table.")
elif not GAP:
    st.info("Write a proposed gap in Q0 first.")
elif run_button("Assess the gap", "btn_q5"):
    funnel_text = L.funnel_line(meta.get("funnel"), meta.get("n_pool"), ss.q3["settings"]["k"])
    with st.spinner("Weighing the gap ..."):
        out, usage, raw = L.assess_gap(QUESTION, GAPCTX, ss.q4["kept"], ss.q4["review"], ss.q4["rows"],
                                       funnel_text, client(), system=GAP_PROMPT, model=MODEL)
    ss.ledger.add("Q5 assess gap", usage)
    ss.q5 = {"result": out, "raw": raw, "model": MODEL,
             "settings": {"gap": GAP, "disproof": DISPROOF, "papers": len(ss.q4["kept"]), "funnel": funnel_text}}
if "q5" in ss:
    r5 = ss.q5
    st.caption(f"Judged by {r5['model']} from {r5['settings']['papers']} papers. {r5['settings']['funnel']}")
    if r5["result"] is None:
        st.warning("The reply was not a JSON object. Read it as text, or fix the prompt and run again.")
        st.code(r5["raw"])
    else:
        out = r5["result"]
        st.subheader(f"Verdict: {out.get('verdict', '?')}")
        cx = out.get("counterexamples") or []
        st.markdown("**Closest counterexamples**" if cx else "**No counterexamples named.**")
        for c in cx if isinstance(cx, list) else []:
            pmid = str(c.get("pmid", "")) if isinstance(c, dict) else str(c)
            d = by.get(pmid)
            why = c.get("why", "") if isinstance(c, dict) else ""
            st.markdown((paper_line(d) if d else f"**{pmid}** (not in the corpus)") + f": {why}")
        st.markdown(f"**Remaining uncertainty:** {out.get('uncertainty', '')}")
        st.markdown(f"**Search limits:** {out.get('search_limits', '')}")
        with st.expander("Raw model reply"):
            st.code(r5["raw"])
    comment_box("q5")

st.divider()
st.markdown("When you are done, download the report from the sidebar. It holds every setting, "
            "prompt, table, and your comments. That file is the homework.")
render_totals()
