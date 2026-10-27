"""Headless tests for the gap loop (Q0, Q1 funnel, Q4 screen-all, Q6).

Run from the repo root:  app/.venv/bin/python -m pytest tests
Every model call is faked, so no API key and no network are needed."""

import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
sys.path.insert(0, APP)

import lit_helper as L  # noqa: E402

PMID_IN_CONTEXT = re.compile(r"^PMID (\d+) \|", re.M)


def fake_chat_factory(prose_for=(), prose_first_gap=False, log=None):
    """A stand-in for lit_helper._chat that answers by the system prompt."""
    state = {"gap_calls": 0}

    def fake(client, system, user, max_tokens=4096, model=L.CHAT_MODEL):
        usage = L.Usage(model, 1000, 200, calls=1)
        if log is not None:
            log.append(user)
        if "JSON array" in system:                      # grader
            pmids = PMID_IN_CONTEXT.findall(user)
            if any(p in prose_for for p in pmids):
                return "I cannot grade these.", usage
            items = [{"pmid": p, "grade": "HIGH" if int(p) % 7 == 0 else "LOW",
                      "gap_relation": "partly", "evidence_quote": "x", "why": "y"} for p in pmids]
            return json.dumps(items), usage
        if "fact-checker" in system:                     # verifier
            return json.dumps({"verdict": "insufficient", "quote": "", "issue": "abstract silent"}), usage
        if "research gap" in system:                     # Q6
            state["gap_calls"] += 1
            if prose_first_gap and state["gap_calls"] == 1:
                return "Let me think about that.", usage
            return json.dumps({"verdict": "partly addressed",
                               "counterexamples": [{"pmid": "41308576", "why": "closest"}],
                               "uncertainty": "u", "search_limits": "s"}), usage
        if "literature review" in system:                # synthesis
            pmids = re.findall(r"^PMID (\d+) \|", user, re.M)[:3]
            return "\n".join(f"- Paper {p} reported a result on its cohort [PMID {p}]." for p in pmids), usage
        return "search string", usage                    # rewrite

    return fake


@pytest.fixture
def docs():
    with open(os.path.join(APP, "pools", "lung_ct", "abstracts.json")) as f:
        return json.load(f)


def test_pool_meta_shipped():
    m = L.pool_meta("lung_ct")
    assert m["total"] == 183 and m["fetched"] == 183 and m["fetch_date"] == "2026-10-18"


def test_question_block():
    assert L.question_block("q") == "QUESTION: q"
    assert "DISPROVE" in L.question_block("q", {"gap": "g", "disproof": "d"})
    assert "DISPROVE" not in L.question_block("q", {"gap": "g", "disproof": ""})


def test_grade_all_full(monkeypatch, docs):
    monkeypatch.setattr(L, "_chat", fake_chat_factory())
    recs, usage, failed = L.grade_all("q", docs, client=object(), gap={"gap": "g", "disproof": "d"})
    assert [r["pmid"] for r in recs] == [d["pmid"] for d in docs]
    assert failed == [] and usage.calls == 19
    assert all(r["doc"]["pmid"] == r["pmid"] for r in recs)


def test_grade_all_prose_batch(monkeypatch, docs):
    bad = docs[25]["pmid"]                              # inside the third batch
    monkeypatch.setattr(L, "_chat", fake_chat_factory(prose_for={bad}))
    recs, usage, failed = L.grade_all("q", docs, client=object())
    assert len(recs) == 171 and len(failed) == 10 and bad in failed
    assert usage.calls == 20                            # one retry for the bad batch


def test_grade_all_budget(monkeypatch, docs):
    monkeypatch.setattr(L, "_chat", fake_chat_factory())
    recs, usage, failed = L.grade_all("q", docs, client=object(), budget_usd=0.0001, workers=1)
    assert len(recs) == 10 and len(failed) == 171 and usage.calls == 1


def test_assess_gap_retry(monkeypatch, docs):
    monkeypatch.setattr(L, "_chat", fake_chat_factory(prose_first_gap=True))
    recs = [{"pmid": docs[0]["pmid"], "grade": "HIGH", "doc": docs[0]}]
    out, usage, raw = L.assess_gap("q", {"gap": "g", "disproof": "d"}, recs, "review", [], "funnel", client=object())
    assert out["verdict"] == "partly addressed" and usage.calls == 2


def test_estimate_is_cents(docs):
    cents = 100 * L.estimate_grade_all(docs)
    assert 1 < cents < 6


def test_app_end_to_end(monkeypatch):
    from streamlit.testing.v1 import AppTest

    log = []
    monkeypatch.setattr(L, "_chat", fake_chat_factory(log=log))
    monkeypatch.setattr(L, "anthropic_client", lambda: object())
    monkeypatch.setattr(L, "embed_texts", lambda texts, *a, **k: (_ for _ in ()).throw(AssertionError("no embedding calls")))

    at = AppTest.from_file(os.path.join(APP, "app.py"), default_timeout=300)
    at.run()
    assert at.session_state["gap"].startswith("No AI system")

    at.radio(key="source").set_value("Shipped pool").run()
    at.button(key="btn_q1").click().run()
    assert "docs" in at.session_state
    assert at.session_state["corpus_meta"]["funnel"]["total"] == 183
    assert any("PubMed matched" in m.value for m in at.markdown), "funnel caption missing"

    at.button(key="btn_q2").click().run()
    assert at.session_state["q2"]["records"] is not None
    assert "PROPOSED GAP" in log[-1]

    at.radio(key="strategy").set_value("Keyword")
    at.toggle(key="screen_all").set_value(True)
    at.slider(key="cap_cents").set_value(1).run()
    at.button(key="btn_q4").click().run()
    assert at.error and "estimated" in at.error[0].value
    assert "q4" not in at.session_state

    at.slider(key="cap_cents").set_value(10).run()
    at.button(key="btn_q4").click().run()
    q4 = at.session_state["q4"]
    assert q4["all_records"] is not None and len(q4["all_records"]) == 181
    assert q4["settings"]["screened"] == 181
    assert any("Full screen of 181 papers" in m.value for m in at.markdown)

    at.button(key="btn_q5").click().run()
    q5 = at.session_state["q5"]
    assert q5["settings"]["source"] == "Full screen" and q5["kept"]
    assert all(r["verdict"] == "insufficient" for r in q5["rows"])
    assert any("insufficient" in m.value for m in at.markdown)

    at.button(key="btn_q6").click().run()
    assert at.session_state["q6"]["result"]["verdict"] == "partly addressed"
    assert "Search funnel: PubMed matched 183" in log[-1]

    report = at.session_state["report_md"]
    for needle in ("## Q0. Question and gap", "Search funnel", "Full screen of 181 papers",
                   "## Q6. Gap assessment", "Verdict: partly addressed", "### Gap prompt (default)"):
        assert needle in report, needle
    assert not at.exception
