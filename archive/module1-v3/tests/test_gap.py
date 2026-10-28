"""Headless tests for the gap loop (Q0, Q1 funnel, Q3 top-10 search, Q4).

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
            log.append(system + "\n" + user)
        if "JSON array" in system:                      # grader
            pmids = PMID_IN_CONTEXT.findall(user)
            if any(p in prose_for for p in pmids):
                return "I cannot grade these.", usage
            items = [{"pmid": p, "bears": "yes", "score": 3 if int(p) % 7 == 0 else 8,
                      "evidence_quote": "x", "why": "y",
                      "gap_checks": [{"condition": "routine data", "met": "yes", "quote": "q"},
                                     {"condition": "external test", "met": "not reported", "quote": None}]}
                     for p in pmids]
            return json.dumps(items), usage
        if "fact-checker" in system:                     # verifier
            return json.dumps({"verdict": "insufficient", "quote": "", "issue": "abstract silent"}), usage
        if "still open" in system:                       # Q5 verdict
            state["gap_calls"] += 1
            if prose_first_gap and state["gap_calls"] == 1:
                return "Let me think about that.", usage
            return json.dumps({"verdict": "partly addressed",
                               "counterexamples": [{"pmid": "41308576", "why": "closest"}],
                               "uncertainty": "u", "search_limits": "s"}), usage
        if "dialectic" in system:                        # synthesis
            pmids = re.findall(r"^PMID (\d+) \|", user, re.M)[:3]
            return "\n".join(f"- Paper {p} reported a result on its cohort [PMID {p}]." for p in pmids), usage
        raise AssertionError(f"unexpected model call: {system[:60]}")

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


def test_grade_truncated_not_retried(monkeypatch, docs):
    calls = []
    def cut(client, system, user, max_tokens=4096, model=L.CHAT_MODEL):
        calls.append(max_tokens)
        return '[{"pmid": "1", "score": 2', L.Usage(model, 10, 10, calls=1, truncated=True)
    monkeypatch.setattr(L, "_chat", cut)
    recs, usage, raw = L.grade_relevance("q", docs[:10], client=object())
    assert recs is None and usage.truncated and usage.calls == 1
    assert calls == [L.GRADE_MAX_TOKENS]


def test_assess_gap_retry(monkeypatch, docs):
    monkeypatch.setattr(L, "_chat", fake_chat_factory(prose_first_gap=True))
    recs = [{"pmid": docs[0]["pmid"], "stance": "antithesis", "score": 2, "doc": docs[0]}]
    out, usage, raw = L.assess_gap("q", {"gap": "g", "disproof": "d"}, recs, "review", [], "funnel", client=object())
    assert out["verdict"] == "partly addressed" and usage.calls == 2


def test_app_end_to_end(monkeypatch):
    from streamlit.testing.v1 import AppTest

    log = []
    monkeypatch.setattr(L, "_chat", fake_chat_factory(log=log))
    monkeypatch.setattr(L, "anthropic_client", lambda: object())
    monkeypatch.setattr(L, "embed_texts", lambda texts, *a, **k: (_ for _ in ()).throw(AssertionError("no embedding calls")))

    at = AppTest.from_file(os.path.join(APP, "app.py"), default_timeout=300)
    at.run()
    assert at.session_state["gap"].startswith("No AI model")          # default question set
    at.selectbox(key="qset").set_value("lung_ct_screening.json").run()
    assert at.session_state["gap"].startswith("No AI system")

    at.radio(key="source").set_value("Shipped pool").run()
    at.button(key="btn_q1").click().run()
    assert "docs" in at.session_state
    assert at.session_state["corpus_meta"]["funnel"]["total"] == 183
    assert any("PubMed matched" in m.value for m in at.markdown), "funnel caption missing"

    # every single-box prompt editor shows its default text, after load and after a question set
    def box(name):
        return next(ta for ta in at.text_area if ta.key.startswith(f"w_{name}_"))
    for name, default in (("synth", L.SYNTHESIS_INSTRUCTIONS), ("verify", L.VERIFY_INSTRUCTIONS), ("gap", L.GAP_INSTRUCTIONS)):
        assert box(name).value == default, f"{name} prompt box empty or stale"
    box("synth").set_value("my synthesis prompt").run()
    assert at.session_state["prompt_synth"] == "my synthesis prompt"
    at.button(key="reset_synth").click().run()
    assert box("synth").value == L.SYNTHESIS_INSTRUCTIONS and at.session_state["prompt_synth"] == L.SYNTHESIS_INSTRUCTIONS


    # the grading prompt is several boxes joined into one system prompt
    rules_box = next(ta for ta in at.text_area if ta.key.startswith("w_grade_todo_"))
    rules_box.set_value("Be brief.").run()
    assert at.session_state["prompt_grade"].endswith("Be brief.\n\n" + L.GRADE_SECTIONS[-1][1] + "\n\n" + L.GRADE_SECTIONS[-1][2])
    at.button(key="btn_q2").click().run()
    assert at.session_state["q2"]["records"] is not None
    assert "Proposed research gap:\nNo AI system" in log[-1]   # filled into the system prompt
    assert "Evidence that would disprove it: A paper whose own study" in log[-1]
    assert "{research_question}" not in log[-1] and "QUESTION:" not in log[-1]
    tables = [d.value for d in at.dataframe if "gap_checks" in d.value.columns]
    assert tables and tables[0]["gap_checks"].str.contains("yes: routine data").all(), "gap_checks not flattened"
    assert (tables[0]["gap_relation"] == "partial").all()   # derived in code from the checks
    flags = dict(zip(tables[0]["pmid"], tables[0]["flag"]))
    assert all(f is None for f in flags.values())   # partial checks contradict neither stance
    stances = dict(zip(tables[0]["pmid"], tables[0]["stance"]))
    assert all((st == "antithesis") == (int(p) % 7 == 0) for p, st in stances.items())
    assert list(tables[0].columns[:2]) == ["pmid", "stance"]

    at.radio(key="strategy").set_value("Anchor's neighbours")
    at.button(key="btn_q3").click().run()
    q3 = at.session_state["q3"]
    assert q3["settings"]["strategy"] == "Anchor's neighbours" and q3["settings"]["anchor"] == "41308576"
    assert len(q3["hits"]) == 10 and "41308576" not in {d["pmid"] for d, _ in q3["hits"]}
    assert len(q3["records"]) == 10
    assert any("nearest to the anchor" in m.value for m in at.markdown)
    html_doc, height = L.stance_panels_html(q3["records"])
    assert "Thesis: for the gap (" in html_doc and "Antithesis: against the gap (" in html_doc
    assert html_doc.count('class="p t"') + html_doc.count('class="p a r"') == 10
    assert "score 8:" in html_doc and "score 3:" in html_doc and "(anchor)" not in html_doc
    assert height > 100
    calls_before = at.session_state["ledger"].calls
    at.radio(key="strategy").set_value("Embedding").run()
    assert "q3" not in at.session_state                      # strategy change clears the step
    monkeypatch.setattr(L, "search", lambda q, docs, V, k=10, **kw: (L.nearest("41308576", docs, V, k=k), L.Usage(L.EMBED_MODEL)))
    at.button(key="btn_q3").click().run()
    assert at.session_state["q3"]["settings"]["strategy"] == "Embedding"
    assert at.session_state["ledger"].calls == calls_before + 1
    assert any("Anchor rank among the 10" in m.value for m in at.markdown)

    at.button(key="btn_q4").click().run()
    q4 = at.session_state["q4"]
    assert q4["kept"]
    assert all(r["verdict"] == "insufficient" for r in q4["rows"])
    assert any("insufficient" in m.value for m in at.markdown)

    at.button(key="btn_q5").click().run()
    assert at.session_state["q5"]["result"]["verdict"] == "partly addressed"
    assert "Search funnel: PubMed matched 183" in log[-1]

    report = at.session_state["report_md"]
    for needle in ("## Q0. Question and gap", "Search funnel", "10 graded",
                   "## Q5. Gap assessment", "Verdict: partly addressed", "### Gap prompt (default)"):
        assert needle in report, needle
    assert not at.exception


def test_surname():
    assert L.surname("Paul R") == "Paul" and L.surname("van der Berg JM") == "van der Berg"
    assert L.surname("Ledda") == "Ledda" and L.surname("") == "?"


def test_stance_from_score():
    assert L.derive_stance({"bears": "no", "score": 2}) == "neutral"
    assert L.derive_stance({"bears": "yes", "score": 5}) == "antithesis"
    assert L.derive_stance({"bears": "yes", "score": "6"}) == "thesis"
    assert L.derive_stance({"bears": "yes"}) is None
    assert L.derive_stance({}) is None
    assert L.select_records([{"stance": "thesis"}, {"stance": "neutral"}]) == ([{"stance": "thesis"}], True)
    assert L.select_records([{"why": "no fields"}]) == ([{"why": "no fields"}], False)


def test_stance_flag():
    rel = lambda c: dict(zip(("gap_relation", "gap_why"), L.gap_relation(c)))
    met = [{"condition": "a", "met": "yes", "quote": "q"}]
    unmet = [{"condition": "a", "met": "no", "quote": "q"}]
    assert L.stance_flag({"score": 2, **rel(met)}) is None
    assert L.stance_flag({"score": 8, **rel(met)}) == "score 8, but every condition is met"
    assert L.stance_flag({"score": 2, **rel(unmet)}).startswith("score 2 with a condition shown unmet")
    assert L.stance_flag({"score": 4, **rel(unmet)}) is None
    assert L.stance_flag({"score": 8, **rel(unmet)}) is None
    assert L.stance_flag({"score": None, **rel(met)}) is None
    assert L.stance_flag({"score": 8}) is None


def test_gap_relation_rule():
    c = lambda met: {"condition": met, "met": met, "quote": None}
    assert L.gap_relation([c("yes"), c("yes")])[0] == "counterexample"
    assert L.gap_relation([c("yes"), c("no")])[0] == "no"
    assert L.gap_relation([c("yes"), c("not reported")])[0] == "partial"
    assert L.gap_relation([c("not reported")])[0] == "unclear"
    assert L.gap_relation([])[0] == "unclear"
    assert L.gap_relation([c("Yes"), c("NO")]) == ("no", "not met: NO")
