"""
Module 1 v2: literature review with embeddings, grading, and verification.

Run:  streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

import lit_helper as L

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Streamlit Community Cloud keeps secrets in st.secrets; mirror them into the
# environment so the SDKs pick them up the same way as from .env.
for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NCBI_API_KEY", "NCBI_EMAIL"):
    try:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]
    except Exception:
        pass

st.set_page_config(page_title="Module 1 v2: literature review", layout="wide")
st.title("Module 1 v2: a literature review in five steps")

ss = st.session_state
ss.setdefault("ledger", L.Ledger())
ss.setdefault("client", None)

# ---- pool selector. Changing the pool clears every result below Q1.
pools = L.list_pools()
if not pools:
    st.error("No pools found. Run build_pool.py first.")
    st.stop()
POOL = st.selectbox("Paper pool", pools, index=pools.index(L.DEFAULT_POOL) if L.DEFAULT_POOL in pools else 0, key="pool")
if ss.get("active_pool") != POOL:
    for k in ("docs", "V", "q2", "q3", "q3_nn", "q4", "q4_hits", "q5_review", "q5_rows"):
        ss.pop(k, None)
    ss.active_pool = POOL

PAPERS = L.load_papers(POOL)
QUESTION = PAPERS["question"]
ANCHOR = PAPERS["anchor"]
st.markdown(f"**Question:** {QUESTION}")


def client():
    if ss.client is None:
        ss.client = L.anthropic_client()
    return ss.client


@st.cache_resource(show_spinner=False)
def cached_corpus(pool):
    return L.build_corpus(pool)


def paper_line(d, score=None, tag=""):
    s = f"**{d['pmid']}** · {d['first_author']} {d['year']} · {d['title']}"
    if score is not None:
        s = f"`{score:.3f}` " + s
    return s + (f" **{tag}**" if tag else "")


def grade_table(records):
    rows = []
    for r in records:
        d = r["doc"]
        rows.append({
            "PMID": d["pmid"],
            "Paper": f"{d['first_author']} {d['year']}",
            "Grade": r["grade"],
            "Cohort": r.get("cohort"),
            "n": r.get("n"),
            "Headline result": r.get("headline_result"),
            "Quote found": "yes" if r.get("quote_found") else "NO",
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    for r in records:
        with st.expander(f"{r['doc']['pmid']} {r['grade']}: why and evidence"):
            st.write("**Why:** " + str(r.get("why")))
            st.write("**Evidence quote:** " + str(r.get("evidence_quote")))
            if not r.get("quote_found"):
                st.warning("This quote was not found in the abstract. Read the abstract and judge for yourself.")
            st.caption(r["doc"]["abstract"])


# ------------------------------------------------------------------ Q1

st.header("Q1. Download and embed")
st.markdown(
    "Fetch the pool's frozen PubMed list, keep records with an abstract, and turn each "
    "title plus abstract into a 1,536-number vector with `text-embedding-3-small`. "
    "The files are cached under `pools/`, so this is a network call only once."
)
if st.button("Build corpus", key="btn_q1"):
    with st.spinner("Loading corpus ..."):
        docs, V, usage, source = cached_corpus(POOL)
    if usage:
        ss.ledger.add("Q1 embed", usage)
    ss.docs, ss.V = docs, V
    st.success(f"{len(docs)} documents, vectors {V.shape[0]} x {V.shape[1]}, loaded from {source}")
    years = sorted({d["year"] for d in docs if d.get("year")})
    st.caption(f"Years {years[0]} to {years[-1]}. **Report: the document count.**")

def render_sidebar():
    # ------------------------------------------------------------------ sidebar

    with st.sidebar:
        st.subheader("Pool")
        st.write(POOL)
        st.subheader("Spend this session")
        st.metric("Model calls", ss.ledger.calls)
        st.metric("Cost", f"${ss.ledger.cost_usd:.4f}")
        with st.expander("Per step"):
            for step, u in ss.ledger.entries:
                st.caption(f"{step}: {u.input_tokens} in / {u.output_tokens} out, ${u.cost_usd:.5f}")


if "docs" not in ss:
    render_sidebar()
    st.stop()
docs, V = ss.docs, ss.V
by = {d["pmid"]: d for d in docs}

# ------------------------------------------------------------------ Q2

st.header("Q2. Grade three fixed papers")
st.markdown(
    "One **anchor** paper chosen by the instructor and two **companions** picked "
    "at random (seed 42). Haiku grades each one HIGH, MEDIUM, or LOW for how well "
    "it answers the question, and must quote the sentence that justifies the grade. "
    "No search yet."
)
fixed = [by[ANCHOR]] + [by[p] for p in PAPERS["companions"]]
for d, tag in zip(fixed, ["anchor", "companion", "companion"]):
    st.markdown(paper_line(d, tag=tag))
if st.button("Grade", key="btn_q2"):
    with st.spinner("Haiku is grading ..."):
        recs, usage = L.grade_relevance(QUESTION, fixed, client())
    ss.ledger.add("Q2 grade", usage)
    ss.q2 = recs
if "q2" in ss:
    grade_table(ss.q2)
    st.caption("**Report: which PMID is HIGH.**")

# ------------------------------------------------------------------ Q3

st.header("Q3. Find the anchor's neighbours and grade them")
st.markdown(
    "Cosine similarity between the anchor's vector and all the others. The two "
    "nearest replace the random companions, and Haiku grades the new trio. "
    "Same grader, better company?"
)
if st.button("Find similar", key="btn_q3"):
    nn = L.nearest(ANCHOR, docs, V, k=10)
    ss.q3_nn = nn
    trio = [by[ANCHOR], nn[0][0], nn[1][0]]
    with st.spinner("Haiku is grading ..."):
        recs, usage = L.grade_relevance(QUESTION, trio, client())
    ss.ledger.add("Q3 grade", usage)
    ss.q3 = recs
if "q3" in ss:
    st.markdown("**Ten nearest to the anchor**")
    for d, s in ss.q3_nn:
        st.markdown(paper_line(d, s))
    st.markdown("**Grades for anchor + two nearest**")
    grade_table(ss.q3)
    st.caption("**Report: did the neighbours grade higher than Q2's companions?**")

# ------------------------------------------------------------------ Q4

st.header("Q4. Search with the question")
st.markdown(
    "Now the question itself is embedded and compared with every paper. "
    "This is retrieval: the question chooses the context. Haiku grades all ten hits."
)
if st.button("Search", key="btn_q4"):
    with st.spinner("Embedding the question and grading ten papers ..."):
        hits, u1 = L.search(QUESTION, docs, V, k=10)
        ss.ledger.add("Q4 embed", u1)
        recs, u2 = L.grade_relevance(QUESTION, [d for d, _ in hits], client())
        ss.ledger.add("Q4 grade", u2)
    ss.q4_hits, ss.q4 = hits, recs
if "q4" in ss:
    rank = L.rank_of(ANCHOR, ss.q4_hits)
    st.markdown(f"Anchor rank among the ten: **{rank if rank else 'not in top 10'}**")
    for d, s in ss.q4_hits:
        st.markdown(paper_line(d, s, tag="anchor" if d["pmid"] == ANCHOR else ""))
    grade_table(ss.q4)
    n_high = sum(r["grade"] == "HIGH" for r in ss.q4)
    st.caption(f"**Report: {n_high} of 10 graded HIGH, and the anchor's rank.**")

# ------------------------------------------------------------------ Q5

st.header("Q5. Synthesize and verify")
st.markdown(
    "Haiku writes a short review from the HIGH and MEDIUM papers of Q4, citing "
    "PMIDs. Then every cited sentence goes back to Haiku with only its abstract, "
    "and is marked supported, partial, or unsupported."
)
attach = st.toggle("Attach full abstracts for HIGH papers (MEDIUM always as structured record)", value=True)
if "q4" not in ss:
    st.info("Run Q4 first.")
elif st.button("Summarize and verify", key="btn_q5"):
    with st.spinner("Writing the review ..."):
        review, u1 = L.synthesize(QUESTION, ss.q4, attach_full_high=attach, client=client())
    ss.ledger.add("Q5 synthesize", u1)
    ss.q5_review = review
    bar = st.progress(0.0, text="Verifying claims ...")
    rows, u2 = L.verify_claims(review, docs, client(),
                               progress=lambda i, n: bar.progress(i / n, text=f"Verifying claim {i} of {n}"))
    bar.empty()
    ss.ledger.add("Q5 verify", u2)
    ss.q5_rows = rows
if "q5_review" in ss:
    st.markdown(ss.q5_review)
    rows = ss.q5_rows
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("supported", "partial", "unsupported")}
    st.markdown(f"**{len(rows)} claim checks: {counts['supported']} supported, "
                f"{counts['partial']} partial, {counts['unsupported']} unsupported**")
    st.dataframe(
        [{"Verdict": r["verdict"], "PMID": r["pmid"], "Claim": r["claim"], "Issue": r.get("issue")} for r in rows],
        width="stretch", hide_index=True,
    )
    st.caption("**Report: the number of claims and how many were supported. Paste this table.**")

render_sidebar()
