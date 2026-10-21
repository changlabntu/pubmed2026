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
import os

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

ss = st.session_state
ss.setdefault("ledger", L.Ledger())
ss.setdefault("client", None)

STEPS = ["q1", "q2", "q3", "q4", "q5"]
RESULT_KEYS = {
    "q1": ["docs", "V", "V_title", "corpus_meta"],
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
    if clicked and ss.ledger.cost_usd >= CAP_USD:
        st.error(f"Cost cap reached: spent {100 * ss.ledger.cost_usd:.1f} cents of "
                 f"{CAP_CENTS}. Raise the slider in the sidebar to continue.")
        return False
    return clicked


def prompt_editor(name, default, label):
    """Full system prompt in a text area with a Reset button. Returns the text."""
    key = f"prompt_{name}"
    ss.setdefault(key, default)
    with st.expander(label, expanded=False):
        if st.button("Reset to default", key=f"reset_{name}"):
            ss[key] = default
            st.rerun()
        st.text_area("System prompt", key=key, height=280, label_visibility="collapsed")
        if ss[key].strip() != default.strip():
            st.caption("Edited. The report will include your version.")
    return ss[key]


def comment_box(step):
    st.text_area("My comment on this step", key=f"comment_{step}", height=80,
                 placeholder="What surprised you? Do you agree with the model? Why or why not?")


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
            row[k] = v if isinstance(v, (str, int, float, type(None))) else json.dumps(v, ensure_ascii=False)
        if "quote_found" in r:
            row["quote found"] = "yes" if r["quote_found"] else "NO"
        rows.append(row)
    return rows


def show_grades(result):
    """Render a grading result: table of whatever came back, or the raw text."""
    records, raw = result["records"], result["raw"]
    st.caption(f"Graded by {result['model']}")
    if records is None:
        st.warning("The reply was not a JSON array of objects, so there is no table. "
                   "Read the answer as text, or fix the prompt and grade again.")
        st.code(raw)
        return
    st.dataframe(record_rows(records), width="stretch", hide_index=True)
    for r in records:
        d = r["doc"]
        with st.expander(f"{d['pmid']} {L.grade_of(r) or ''}: {d['first_author']} {d['year']}, full record and abstract"):
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
        st.toggle("Sonnet 5.5, $3 in / $15 out, 30x Haiku", value=False, disabled=True,
                  help="Switched off for the class. The instructor can enable it in app.py.")
    MODEL = L.MODELS[model_label]
    st.subheader("Cost cap")
    CAP_CENTS = st.slider("Stop making calls after (cents)", 1, 50, 5, key="cap_cents")
    CAP_USD = CAP_CENTS / 100
    totals_box = st.container()

# ---------------------------------------------------------------- Q1

st.header("Q1. Build corpus")
st.markdown(
    "A corpus is a set of PubMed abstracts, each turned into a 1,536-number vector "
    "with `text-embedding-3-small`. Write your own PubMed query, or use a pool the "
    "instructor prepared."
)
source = st.radio("Source", ["My own PubMed query", "Shipped pool"], horizontal=True, key="source")
if source == "Shipped pool":
    pools = L.list_pools()
    if not pools:
        st.error("No pools found. Run build_pool.py first.")
        st.stop()
    POOL = st.selectbox("Pool", pools, index=pools.index(L.DEFAULT_POOL) if L.DEFAULT_POOL in pools else 0, key="pool")
    PAPERS = L.load_papers(POOL)
    st.caption(f"PubMed query behind this pool: `{L.pool_query(POOL)}` plus the review exclusion.")
    q1_sig = ("pool", POOL)
    POOL_KEY = POOL
else:
    POOL, PAPERS = None, None
    OWN_QUERY = st.text_input("PubMed query", key="own_query",
                              placeholder='(knee) AND (MRI) AND ("deep learning" OR "machine learning") AND (pain)')
    OWN_RETMAX = st.slider("Max records", 50, 500, 200, 50, key="own_retmax")
    OWN_EXCLUDE = True   # reviews, editorials, letters, meta-analyses are always excluded
    st.caption("Reviews, editorials, letters and meta-analyses are excluded automatically.")
    q1_sig = ("custom", OWN_QUERY.strip(), OWN_RETMAX, OWN_EXCLUDE)
    POOL_KEY = "custom"
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
        meta = {"pool": POOL, "query": L.pool_query(POOL), "source": src}
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
                    "exclude": OWN_EXCLUDE, "source": src}
        except Exception as e:  # noqa: BLE001
            status.update(label="Pool build failed", state="error")
            st.error(str(e))
            docs = None
    if docs:
        if usage:
            ss.ledger.add("Q1 embed", usage)
        ss.docs, ss.V, ss.corpus_meta = docs, V, meta

CITE_CAP = 100   # citations above this all draw at the largest dot size


@st.cache_data(show_spinner="Drawing the map ...")
def cached_map(pool_id, _vectors):
    return L.tsne_2d(_vectors)


def paper_map(docs, V, pool_id):
    """Every paper as a dot, placed by t-SNE on its embedding. Hover for the title."""
    import altair as alt
    import pandas as pd

    xy = cached_map(pool_id, V)
    cites = [d.get("citations") for d in docs]
    have_cites = any(c is not None for c in cites)
    df = pd.DataFrame({
        "x": xy[:, 0], "y": xy[:, 1],
        "PMID": [d["pmid"] for d in docs],
        "Paper": [f"{d['first_author']} {d['year']}" for d in docs],
        "Title": [d["title"] for d in docs],
        "Year": [d.get("year") for d in docs],
        "Citations": [c if c is not None else "n/a" for c in cites],
    })
    df["size"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).clip(upper=CITE_CAP)
    enc = dict(
        x=alt.X("x", axis=None), y=alt.Y("y", axis=None),
        color=alt.Color("Year:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["PMID", "Paper", "Title", "Citations"],
    )
    if have_cites:
        enc["size"] = alt.Size("size:Q", scale=alt.Scale(domain=[0, CITE_CAP], range=[25, 500]), legend=None)
    chart = alt.Chart(df).mark_circle(opacity=0.75).encode(**enc).properties(height=420).interactive()
    st.altair_chart(chart, use_container_width=True)
    if have_cites:
        top = df.loc[df["size"].idxmax()]
        st.caption(f"Dot size: citation count from NIH iCite, capped at {CITE_CAP}. "
                   f"Most cited: {top['Paper']} ({top['Citations']}).")
    else:
        st.warning("Citation counts are unavailable: the NIH iCite service could not be reached from this "
                   "server, so all dots are the same size. Press Build corpus again to retry."
                   + (f" Last error: {L.LAST_ICITE_ERROR}" if L.LAST_ICITE_ERROR else ""))


if "docs" not in ss:
    st.info("Build the corpus to unlock the other steps.")
else:
    docs, V = ss.docs, ss.V
    meta = ss.corpus_meta
    years = sorted({d["year"] for d in docs if d.get("year")})
    st.success(f"{len(docs)} documents, vectors {V.shape[0]} x {V.shape[1]}, loaded from {meta['source']}. "
               f"Years {years[0]} to {years[-1]}.")
    paper_map(docs, V, meta["pool"])
    st.caption("**Report: the document count.**")
    comment_box("q1")


def render_totals():
    with totals_box:
        st.subheader("Spend this session")
        st.metric("Model calls", ss.ledger.calls)
        st.metric("Cost", f"{100 * ss.ledger.cost_usd:.2f} cents")
        st.progress(min(1.0, ss.ledger.cost_usd / CAP_USD), text=f"of the {CAP_CENTS} cent cap")
        with st.expander("Per step"):
            for step, u in ss.ledger.entries:
                st.caption(f"{step} ({u.model}): {u.input_tokens} in / {u.output_tokens} out, {100 * u.cost_usd:.3f}c")
        st.download_button("Download report (.md)", build_report(),
                           file_name=f"module1_report_{dt.date.today()}.md", mime="text/markdown")


def build_report():
    lines = [f"# Module 1 literature review report", f"Date: {dt.date.today()}", ""]
    meta = ss.get("corpus_meta")
    lines += ["## Q1. Corpus", ""]
    if meta:
        lines += [f"- Pool: `{meta['pool']}`", f"- PubMed query: `{meta['query']}`",
                  f"- Documents: {len(ss.docs)}", ""]
    lines += [f"Comment: {ss.get('comment_q1', '')}", ""]
    if "docs" in ss:
        lines += ["## Research question", "", ss.get("active_question", ""), ""]
    for step, title in (("q2", "Q2. Grade three fixed papers"), ("q3", "Q3. Anchor's neighbours"),
                        ("q4", "Q4. Search with the question")):
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
    lines += ["## Q5. Review and verification", ""]
    res = ss.get("q5")
    if res:
        for k, v in res["settings"].items():
            lines.append(f"- {k}: {v}")
        lines += [f"- model: {res['model']}", "", res["review"], "",
                  md_table([{"verdict": r["verdict"], "pmid": r["pmid"], "claim": r["claim"],
                             "issue": r.get("issue")} for r in res["rows"]])]
    lines += [f"Comment: {ss.get('comment_q5', '')}", ""]
    lines += ["## Prompts used", ""]
    for name, default, label in (("grade", L.GRADE_INSTRUCTIONS, "Grading"),
                                 ("synth", L.SYNTHESIS_INSTRUCTIONS, "Synthesis"),
                                 ("verify", L.VERIFY_INSTRUCTIONS, "Verifier")):
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
    "The model grades three papers for how well they answer the research question: "
    "an **anchor** you choose and two **companions** drawn at random (seed 42). "
    "The grading prompt below decides the scale and the return format. Edit it freely: "
    "the table shows whatever keys the model returns."
)
default_q = PAPERS["question"] if PAPERS else ""
QUESTION = st.text_input("Research question", value=default_q, key=f"question_{POOL_KEY}").strip()
ss.active_question = QUESTION
default_anchor = PAPERS["anchor"] if PAPERS and PAPERS["anchor"] in by else pmids[0]
ANCHOR = st.selectbox("Anchor paper", pmids, index=pmids.index(default_anchor),
                      format_func=lambda p: paper_label(by[p]), key=f"anchor_{POOL_KEY}")
if PAPERS and ANCHOR == PAPERS["anchor"]:
    COMPANIONS = PAPERS["companions"]
else:
    COMPANIONS = L.companions(docs, ANCHOR)
GRADE_PROMPT = prompt_editor("grade", L.GRADE_INSTRUCTIONS, "Grading prompt, shared by Q2, Q3 and Q4 (edit me)")
inputs_changed("q2", (QUESTION, ANCHOR, tuple(COMPANIONS), GRADE_PROMPT))

fixed = [by[ANCHOR]] + [by[p] for p in COMPANIONS]
for d, tag in zip(fixed, ["anchor", "companion", "companion"]):
    st.markdown(paper_line(d, tag=tag))
if not QUESTION:
    st.info("Type a research question first.")
elif run_button("Grade", "btn_q2"):
    with st.spinner("Grading ..."):
        recs, usage, raw = L.grade_relevance(QUESTION, fixed, client(), system=GRADE_PROMPT, model=MODEL)
    ss.ledger.add("Q2 grade", usage)
    ss.q2 = {"records": recs, "raw": raw, "model": MODEL,
             "settings": {"question": QUESTION, "anchor": ANCHOR, "companions": ", ".join(COMPANIONS)}}
if "q2" in ss:
    show_grades(ss.q2)
    st.caption("**Report: which PMID graded highest, and the model's reason.**")
    comment_box("q2")

# ---------------------------------------------------------------- Q3

st.header("Q3. Find the anchor's neighbours and grade them")
st.markdown(
    "Cosine similarity between the anchor's vector and every other paper. The nearest "
    "two replace the random companions and the trio is graded with the same prompt. "
    "Same grader, better company?"
)
c1, c2 = st.columns(2)
K3 = c1.slider("Neighbours to list", 3, 20, 10, key="k3")
EMBED_MODE = c2.radio("What was embedded", ["Title + abstract", "Title only"], horizontal=True, key="embed_mode")
inputs_changed("q3", (QUESTION, ANCHOR, K3, EMBED_MODE, GRADE_PROMPT))
if QUESTION and run_button("Find similar", "btn_q3"):
    if EMBED_MODE == "Title only":
        if "V_title" not in ss:
            with st.spinner("Embedding titles ..."):
                ss.V_title, u0 = L.embed_texts([d["title"] for d in docs])
            ss.ledger.add("Q3 embed titles", u0)
        V_used = ss.V_title
    else:
        V_used = V
    nn = L.nearest(ANCHOR, docs, V_used, k=K3)
    trio = [by[ANCHOR], nn[0][0], nn[1][0]]
    with st.spinner("Grading ..."):
        recs, usage, raw = L.grade_relevance(QUESTION, trio, client(), system=GRADE_PROMPT, model=MODEL)
    ss.ledger.add("Q3 grade", usage)
    ss.q3 = {"records": recs, "raw": raw, "model": MODEL, "hits": nn,
             "settings": {"anchor": ANCHOR, "k": K3, "embedded": EMBED_MODE}}
if "q3" in ss:
    st.markdown(f"**{len(ss.q3['hits'])} nearest to the anchor**")
    for d, s in ss.q3["hits"]:
        st.markdown(paper_line(d, s))
    st.markdown("**Grades for anchor + two nearest**")
    show_grades(ss.q3)
    st.caption("**Report: did the neighbours grade higher than Q2's companions?**")
    comment_box("q3")

# ---------------------------------------------------------------- Q4

st.header("Q4. Search with the question")
st.markdown(
    "Now the question itself does the retrieving. Three strategies: **embedding** "
    "(meaning), **keyword** (BM25 word overlap, no model), or **hybrid** (both rankings "
    "fused). Optionally let the model rewrite the question into a search string first. "
    "Every hit is graded."
)
c1, c2, c3 = st.columns([2, 1, 1])
STRATEGY = c1.radio("Strategy", ["Embedding", "Keyword", "Hybrid"], horizontal=True, key="strategy")
REWRITE = c2.toggle("Rewrite the question first", value=False, key="rewrite")
K4 = c3.slider("Hits to grade", 3, 20, 10, key="k4")
inputs_changed("q4", (QUESTION, STRATEGY, REWRITE, K4, GRADE_PROMPT))
if QUESTION and run_button("Search", "btn_q4"):
    with st.spinner("Searching and grading ..."):
        q_text = QUESTION
        if REWRITE:
            q_text, u0 = L.rewrite_query(QUESTION, client(), model=MODEL)
            ss.ledger.add("Q4 rewrite", u0)
        if STRATEGY == "Embedding":
            hits, u1 = L.search(q_text, docs, V, k=K4)
        elif STRATEGY == "Keyword":
            hits, u1 = L.keyword_search(q_text, docs, k=K4), None
        else:
            hits, u1 = L.hybrid_search(q_text, docs, V, k=K4)
        if u1:
            ss.ledger.add("Q4 embed", u1)
        recs, u2, raw = L.grade_relevance(QUESTION, [d for d, _ in hits], client(), system=GRADE_PROMPT, model=MODEL)
        ss.ledger.add("Q4 grade", u2)
    ss.q4 = {"records": recs, "raw": raw, "model": MODEL, "hits": hits,
             "settings": {"strategy": STRATEGY, "search text": q_text, "k": K4}}
if "q4" in ss:
    r4 = ss.q4
    if r4["settings"]["search text"] != QUESTION:
        st.markdown(f"Search string after rewrite: _{r4['settings']['search text']}_")
    rank = L.rank_of(ANCHOR, r4["hits"])
    st.markdown(f"Anchor rank among the {len(r4['hits'])}: **{rank if rank else 'not in the list'}**")
    for d, s in r4["hits"]:
        st.markdown(paper_line(d, s, tag="anchor" if d["pmid"] == ANCHOR else ""))
    show_grades(r4)
    if r4["records"] is not None:
        n_high = sum(L.grade_of(r) == "HIGH" for r in r4["records"])
        st.caption(f"**Report: {n_high} of {len(r4['hits'])} graded HIGH, and the anchor's rank.**")
    comment_box("q4")

# ---------------------------------------------------------------- Q5

st.header("Q5. Synthesize and verify")
st.markdown(
    "The model writes a short review from the papers you include, citing PMIDs. "
    "Then every cited sentence goes back to the model with only the abstract it cites, "
    "and is marked supported, partial, or unsupported."
)
c1, c2 = st.columns(2)
INCLUDE = c1.multiselect("Include papers graded", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"], key="include")
ATTACH = c2.toggle("Attach full abstracts for HIGH papers (others as the grader's record)", value=True, key="attach")
SYN_PROMPT = prompt_editor("synth", L.SYNTHESIS_INSTRUCTIONS, "Synthesis prompt (edit me)")
VER_PROMPT = prompt_editor("verify", L.VERIFY_INSTRUCTIONS, "Verifier prompt (edit me)")
inputs_changed("q5", (tuple(INCLUDE), ATTACH, SYN_PROMPT, VER_PROMPT))
if "q4" not in ss or ss.q4["records"] is None:
    st.info("Run Q4 with a prompt that returns JSON first. The review is built from its records.")
elif run_button("Summarize and verify", "btn_q5"):
    kept, had_grades = L.select_records(ss.q4["records"], include=tuple(INCLUDE))
    if not had_grades:
        st.warning("No record carries a `grade` key, so every Q4 paper goes into the review.")
    if not kept:
        st.warning("No paper matches the grades you included.")
    else:
        with st.spinner("Writing the review ..."):
            review, u1 = L.synthesize(QUESTION, kept, attach_full_high=ATTACH, client=client(),
                                      system=SYN_PROMPT, model=MODEL)
        ss.ledger.add("Q5 synthesize", u1)
        bar = st.progress(0.0, text="Verifying claims ...")
        rows, u2 = L.verify_claims(review, docs, client(), system=VER_PROMPT, model=MODEL,
                                   progress=lambda i, n: bar.progress(i / n, text=f"Verifying claim {i} of {n}"))
        bar.empty()
        ss.ledger.add("Q5 verify", u2)
        ss.q5 = {"review": review, "rows": rows, "model": MODEL,
                 "settings": {"included": ", ".join(INCLUDE), "papers": len(kept), "full HIGH abstracts": ATTACH}}
if "q5" in ss:
    st.caption(f"Written and verified by {ss.q5['model']} from {ss.q5['settings']['papers']} papers")
    st.markdown(ss.q5["review"])
    rows = ss.q5["rows"]
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("supported", "partial", "unsupported")}
    st.markdown(f"**{len(rows)} claim checks: {counts['supported']} supported, "
                f"{counts['partial']} partial, {counts['unsupported']} unsupported**")
    st.dataframe(
        [{"Verdict": r["verdict"], "PMID": r["pmid"], "Claim": r["claim"], "Issue": r.get("issue")} for r in rows],
        width="stretch", hide_index=True,
    )
    st.caption("**Report: the number of claims and how many were supported.**")
    comment_box("q5")

st.divider()
st.markdown("When you are done, download the report from the sidebar. It holds every setting, "
            "prompt, table, and your comments. That file is the homework.")
render_totals()
