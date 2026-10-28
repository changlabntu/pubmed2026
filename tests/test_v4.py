"""Headless tests for v4: the four moves with a faked model, plus the pure
helpers (stance from score, flags, panels, argue parsing, report order)."""

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


def fake_chat_factory(prose_for=(), prose_first_argue=False, log=None):
    """A stand-in for lit_helper._chat that answers by the system prompt."""
    state = {"argue_calls": 0}

    def fake(client, system, user, max_tokens=4096, model=L.CHAT_MODEL):
        usage = L.Usage(model, 1000, 200, calls=1)
        if log is not None:
            log.append(system + "\n" + user)
        if "JSON array" in system:                      # scorer
            pmids = PMID_IN_CONTEXT.findall(user)
            if any(p in prose_for for p in pmids):
                return "I cannot score these.", usage
            items = [{"pmid": p, "bears": "no" if int(p) % 11 == 0 else "yes",
                      "score": None if int(p) % 11 == 0 else (3 if int(p) % 7 == 0 else 8),
                      "evidence_quote": "x", "why": "y",
                      "gap_checks": None if int(p) % 11 == 0 else
                      [{"condition": "routine data", "met": "yes", "quote": "q"},
                       {"condition": "external test", "met": "not reported", "quote": None}]}
                     for p in pmids]
            return json.dumps(items), usage
        if "fact-checker" in system:                     # verifier
            return json.dumps({"verdict": "can't tell", "quote": "", "issue": "abstract silent"}), usage
        if "Return ONLY a JSON object with keys" in system:   # argue
            state["argue_calls"] += 1
            if prose_first_argue and state["argue_calls"] == 1:
                return "Let me think about that.", usage
            pmids = PMID_IN_CONTEXT.findall(user)[:3]
            return json.dumps({
                "thesis": " ".join(f"Paper {p} stopped short of external testing [PMID {p}]." for p in pmids),
                "antithesis": f"The strongest counterexample is [PMID {pmids[0]}]." if pmids else "None.",
                "synthesis": "The gap survives for external validation.",
                "verdict": "Gap Narrowed"}), usage
        raise AssertionError(f"unexpected model call: {system[:60]}")

    return fake


@pytest.fixture
def docs():
    with open(os.path.join(APP, "pools", "lung_ct", "abstracts.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- pure helpers

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
    assert L.stance_flag({"score": 4, **rel(unmet)}).startswith("score 4 with")
    assert L.stance_flag({"score": 5, **rel(unmet)}) is None
    assert L.stance_flag({"score": 8}) is None


def test_gap_relation_rule():
    c = lambda met: {"condition": met, "met": met, "quote": None}
    assert L.gap_relation([c("yes"), c("yes")])[0] == "counterexample"
    assert L.gap_relation([c("yes"), c("no")])[0] == "no"
    assert L.gap_relation([c("yes"), c("not reported")])[0] == "partial"
    assert L.gap_relation([c("not reported")])[0] == "unclear"
    assert L.gap_relation([])[0] == "unclear"


def test_surname():
    assert L.surname("Paul R") == "Paul" and L.surname("van der Berg JM") == "van der Berg"
    assert L.surname("Ledda") == "Ledda" and L.surname("") == "?"


def test_panels_html(docs):
    recs = [{"pmid": d["pmid"], "stance": s, "score": sc, "why": "w", "doc": d}
            for d, (s, sc) in zip(docs[:4], [("thesis", 8), ("antithesis", 2), ("neutral", None), ("thesis", 9)])]
    html_doc, height = L.stance_panels_html(recs, anchor=docs[1]["pmid"])
    assert "Papers for the gap, thesis (2)" in html_doc and "Papers against the gap, antithesis (1)" in html_doc
    assert "(anchor)" in html_doc and "Neutral" in html_doc and height > 60
    assert f'href="https://pubmed.ncbi.nlm.nih.gov/{docs[1]["pmid"]}/" target="_blank"' in html_doc
    assert 'id="detail"' in html_doc and "data-detail=" in html_doc and "mouseenter" in html_doc
    # thesis column sorted by score descending
    assert html_doc.index("Score 9 out of 10") < html_doc.index("Score 8 out of 10")


def test_score_truncated_not_retried(monkeypatch, docs):
    calls = []
    def cut(client, system, user, max_tokens=4096, model=L.CHAT_MODEL):
        calls.append(max_tokens)
        return '[{"pmid": "1", "score": 2', L.Usage(model, 10, 10, calls=1, truncated=True)
    monkeypatch.setattr(L, "_chat", cut)
    recs, usage, raw = L.grade_relevance("q", docs[:10], client=object())
    assert recs is None and usage.truncated and usage.calls == 1
    assert calls == [L.GRADE_MAX_TOKENS]


def test_argue_parse_retry_and_verdict(monkeypatch, docs):
    monkeypatch.setattr(L, "_chat", fake_chat_factory(prose_first_argue=True))
    recs = [{"pmid": docs[0]["pmid"], "stance": "antithesis", "score": 2, "doc": docs[0]}]
    out, usage, raw = L.argue("q", {"gap": "g", "disproof": "d"}, recs, "funnel", client=object())
    assert usage.calls == 2                                  # one retry on prose
    assert out["verdict"] == "gap narrowed"                  # normalised to the VERDICTS wording
    assert out["thesis"].startswith("Paper") and "[PMID" in out["antithesis"]
    text = L.argument_text(out)
    assert text.startswith("THESIS\n") and "\n\nANTITHESIS\n" in text and "\n\nSYNTHESIS\n" in text


def test_argue_unknown_verdict_becomes_cant_tell(monkeypatch, docs):
    monkeypatch.setattr(L, "_chat", lambda *a, **k: (json.dumps(
        {"thesis": "t", "antithesis": "a", "synthesis": "s", "verdict": "probably"}), L.Usage(L.CHAT_MODEL, 1, 1, calls=1)))
    out, _, _ = L.argue("q", {}, [], "f", client=object())
    assert out["verdict"] == "can't tell"


def test_verifier_passes_label_through(monkeypatch, docs):
    monkeypatch.setattr(L, "_chat", fake_chat_factory())
    rows, usage = L.verify_claims(f"A claim about this paper [PMID {docs[0]['pmid']}]. "
                                  f"A second claim about a paper that is not here [PMID 1234567].",
                                  docs[:1], client=object(), workers=1)
    assert [r["verdict"] for r in rows] == ["can't tell", "not supported"]
    assert set(r["verdict"] for r in rows) <= set(L.CLAIM_VERDICTS)


def test_split_claims_repairs():
    # a citation list: every PMID attaches to the sentence before the list
    text = "Several others come close but fail the blinding condition: Weber et al [PMID 1111111], Bai et al [PMID 2222222], Bondok et al [PMID 3333333]."
    assert L.split_claims(text) == [("Several others come close but fail the blinding condition: Weber et al",
                                     ["1111111", "2222222", "3333333"])]
    # an uncited sentence between two citations belongs to the earlier one
    text = "Ji et al. ran a cohort of 259,121 participants [PMID 4444444]. It is retrospective and single-centre. Hwang et al. ran a prospective trial of 911 participants [PMID 5555555]."
    out = L.split_claims(text)
    assert out[0] == ("Ji et al. ran a cohort of 259,121 participants It is retrospective and single-centre", ["4444444"])
    assert out[1] == ("Hwang et al. ran a prospective trial of 911 participants", ["5555555"])
    # a label before its detail is checked as label + detail
    assert L.split_claims("- Chang et al., 2020 [PMID 6666666]: trained a Siamese network on knee MRI with pain as the target.") == \
        [("Chang et al., 2020 trained a Siamese network on knee MRI with pain as the target", ["6666666"])]
    # several PMIDs in one bracket, and a trailing uncited clause
    assert L.split_claims("Both trials report higher adenoma detection with CADe [PMID 7777777; PMID 8888888], but neither reports interval cancer.") == \
        [("Both trials report higher adenoma detection with CADe but neither reports interval cancer", ["7777777", "8888888"])]


# ---------------------------------------------------------------- the whole app, headless

def test_app_student_mode(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(L, "_chat", fake_chat_factory())
    monkeypatch.setattr(L, "anthropic_client", lambda: object())
    monkeypatch.setattr(L, "embed_texts", lambda texts, *a, **k: (_ for _ in ()).throw(AssertionError("no embedding calls")))
    monkeypatch.setattr(L, "search", lambda q, docs, V, k=10, **kw: (L.nearest("41308576", docs, V, k=k + 3)[3:], L.Usage(L.EMBED_MODEL)))
    monkeypatch.delenv("TEACHER", raising=False)
    at = AppTest.from_file(os.path.join(APP, "app.py"), default_timeout=300)
    at.run()
    assert not any(s.key == "cap_cents" for s in at.slider)
    assert not any(r.key == "model_label" for r in at.radio)
    assert not any(m.label == "Model calls" for m in at.metric)
    at.selectbox(key="qset").set_value("lung_ct_screening.json").run()
    at.radio(key="source").set_value("Shipped pool").run()
    at.button(key="btn_q2").click().run()
    at.button(key="btn_q3").click().run()
    boxes = {ta.key.rsplit("_", 1)[0]: ta.value for ta in at.text_area if ta.key.startswith("w_argue_") or ta.key.startswith("w_verify_")}
    assert boxes["w_argue_todo"] == L.ARGUE_SECTIONS_STUDENT[1][2] and boxes["w_verify_todo"] == L.VERIFY_SECTIONS_STUDENT[1][2]
    assert all(len(v.splitlines()) == 1 for v in boxes.values()), "student boxes are one line each"
    assert at.session_state["prompt_argue"] == L.join_sections((t, x) for _, t, x in L.ARGUE_SECTIONS_STUDENT)
    at.button(key="btn_q4").click().run()
    assert at.session_state["q4"]["argument"]["verdict"] == "gap narrowed"
    assert "## Appendix" in at.session_state["report_md"]
    assert not at.exception


def test_app_end_to_end(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("TEACHER", "1")
    log = []
    monkeypatch.setattr(L, "_chat", fake_chat_factory(log=log))
    monkeypatch.setattr(L, "anthropic_client", lambda: object())
    monkeypatch.setattr(L, "embed_texts", lambda texts, *a, **k: (_ for _ in ()).throw(AssertionError("no embedding calls")))
    # the question search would embed; stand in with the anchor's neighbours shifted by one
    monkeypatch.setattr(L, "search", lambda q, docs, V, k=10, **kw: (L.nearest("41308576", docs, V, k=k + 3)[3:], L.Usage(L.EMBED_MODEL)))

    at = AppTest.from_file(os.path.join(APP, "app.py"), default_timeout=300)
    at.run()
    assert at.session_state["gap"].startswith("No AI model")          # default question set
    at.selectbox(key="qset").set_value("lung_ct_screening.json").run()
    assert at.session_state["gap"].startswith("No AI system")
    assert not at.exception

    # move 2: shipped pool, the three PMIDs from the set, score them
    at.radio(key="source").set_value("Shipped pool").run()
    assert at.session_state["fixed_lung_ct"].startswith("41308576")
    rules_box = next(ta for ta in at.text_area if ta.key.startswith("w_grade_todo_"))
    rules_box.set_value("Be brief.").run()
    assert at.session_state["prompt_grade"].endswith("Be brief.\n\n" + L.GRADE_SECTIONS[-1][1] + "\n\n" + L.GRADE_SECTIONS[-1][2])
    at.button(key="btn_q2").click().run()
    assert "docs" in at.session_state and at.session_state["corpus_meta"]["fixed"] == ["41308576", "42229468", "33708918"]
    q2 = at.session_state["q2"]
    assert len(q2["records"]) == 3 and q2["records"][0]["pmid"] == "41308576"
    assert "Proposed research gap:\nNo AI system" in log[-1]
    assert any("PubMed matched" in m.value for m in at.markdown), "funnel caption missing"

    # prompt boxes show their text after load and after a question set; edit and reset work
    def box(name, key):
        return next(ta for ta in at.text_area if ta.key.startswith(f"w_{name}_{key}_"))
    for name, secs in (("argue", L.ARGUE_SECTIONS), ("verify", L.VERIFY_SECTIONS)):
        for key, _, default in secs:
            assert box(name, key).value == default, f"{name}/{key} prompt box empty or stale"
    assert at.session_state["prompt_argue"] == L.ARGUE_INSTRUCTIONS
    box("argue", "todo").set_value("Always say gap stands.").run()
    assert "WHAT TO DO\n\nAlways say gap stands." in at.session_state["prompt_argue"]
    assert "BACKGROUND\n\n" in at.session_state["prompt_argue"] and "{research_question}" in at.session_state["prompt_argue"]
    at.button(key="reset_argue").click().run()
    assert at.session_state["prompt_argue"] == L.ARGUE_INSTRUCTIONS

    # move 3: both searches, the union scored, panels for all
    at.button(key="btn_q3").click().run()
    q3 = at.session_state["q3"]
    assert 3 < len(q3["records"]) <= 15 and q3["n_new"] == len(q3["records"]) - 3
    assert "41308576" not in {d["pmid"] for d, _ in q3["hits_anchor"]}
    stances = {L.stance_of(r) for r in q3["records"]}
    assert "thesis" in stances and "antithesis" in stances
    html_doc, _ = L.stance_panels_html(q3["records"], anchor="41308576")
    assert "(anchor)" in html_doc and "Papers against the gap, antithesis (" in html_doc
    tables = [d.value for d in at.dataframe if "gap_checks" in d.value.columns]
    assert tables and list(tables[0].columns[:2]) == ["pmid", "stance"]

    # move 4: the argument with its verdict, then verification
    at.button(key="btn_q4").click().run()
    q4 = at.session_state["q4"]
    assert q4["argument"]["verdict"] in L.VERDICTS and q4["argument"]["verdict"] == "gap narrowed"
    assert q4["rows"] and all(r["verdict"] == "can't tell" for r in q4["rows"])
    argue_call = log[-len(q4["rows"]) - 1]
    assert "PubMed matched 183 records on 2026-10-18" in argue_call   # the argue call carried the funnel
    assert "Proposed research gap:\nNo AI system" in argue_call and "{proposed_gap}" not in argue_call
    assert any("Verdict: gap narrowed" in h.value for h in at.subheader)

    # the report reads as the argument, in order
    report = at.session_state["report_md"]
    order = ["## 1. The claim", "## 2. Starting papers", "(anchor)", "## 3. Evidence", "### Papers for the gap (thesis)",
             "### Papers against the gap (antithesis)", "### How it was searched", "## 4. The argument", "### Thesis",
             "### Antithesis", "### Synthesis", "**Verdict: gap narrowed**", "## 5. Verification",
             "## Appendix", "#### Scoring prompt (EDITED)", "#### Argument prompt (default)", "### Every scored record",
             "### Papers closest to the topic question", "### Papers closest to the anchor"]
    pos = [report.index(s) for s in order]
    assert pos == sorted(pos), "report sections out of order"
    assert "settings" not in report.lower().split("## appendix")[0]
    assert not at.exception
