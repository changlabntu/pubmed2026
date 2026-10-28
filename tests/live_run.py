"""
Live, headless run of the four moves against the real models, to judge whether
the output makes sense. Not collected by pytest. About 3-5 cents per run.

    cd app && /opt/miniconda3/envs/CAMI/bin/python ../tests/live_run.py [question_set.json]
"""

import json
import os
import sys
from collections import Counter

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(HERE), "app")
sys.path.insert(0, APP)
load_dotenv(os.path.join(APP, ".env"))

import lit_helper as L  # noqa: E402


def main(qset_file="lung_ct_screening.json", pool="lung_ct"):
    with open(os.path.join(APP, "question_sets", qset_file)) as f:
        qs = json.load(f)
    question, gap = qs["question"], {"gap": qs["gap"], "disproof": qs["disproof"]}
    fixed = qs["pmids"]
    total = L.Usage(L.CHAT_MODEL)

    docs, V, usage, src = L.build_corpus(pool)
    docs, V, u_extra, missing = L.add_papers(docs, V, fixed)
    assert not missing, missing
    by = {d["pmid"]: d for d in docs}
    print(f"corpus {pool}: {len(docs)} docs ({src}); anchor {fixed[0]}, companions {fixed[1:]}")

    # move 2
    recs2, u2, raw = L.grade_relevance(question, [by[p] for p in fixed], gap=gap)
    L._add(total, u2)
    assert recs2 is not None, raw[-300:]
    print("\n== 2. starting papers")
    for r in recs2:
        print(f"  {L.stance_of(r):10s} {r.get('score')}  {L.surname(r['doc']['first_author'])} {r['doc']['year']}: {r.get('why')}"
              + (f"  FLAG {r['flag']}" if r.get("flag") else ""))

    # move 3
    hits_q, u1 = L.search(question, docs, V, k=6)
    hits_a = L.nearest(fixed[0], docs, V, k=6)
    seen, new_docs = set(fixed), []
    for d, _ in hits_q + hits_a:
        if d["pmid"] not in seen:
            seen.add(d["pmid"]); new_docs.append(d)
    recs3, u3, raw = L.grade_relevance(question, new_docs, gap=gap)
    L._add(total, u3)
    assert recs3 is not None, raw[-300:]
    records = recs2 + recs3
    print(f"\n== 3. evidence: {len(new_docs)} new papers scored, {u3.output_tokens} output tokens, truncated={u3.truncated}")
    print("  stances:", dict(Counter(L.stance_of(r) for r in records)))
    print("  scores: ", sorted(L.score_of(r) for r in records if L.score_of(r) is not None))
    for r in sorted(records, key=lambda r: L.score_of(r) or 11):
        print(f"  {L.stance_of(r):10s} {str(r.get('score')):4s} {L.surname(r['doc']['first_author'])} {r['doc']['year']}: "
              f"{(r.get('why') or '')[:140]}" + (f"  FLAG {r['flag']}" if r.get("flag") else ""))

    # move 4
    kept, _ = L.select_records(records)
    funnel = L.funnel_line(L.pool_meta(pool), len(docs) - len(fixed), len(records))
    out, u4, raw = L.argue(question, gap, kept, funnel)
    L._add(total, u4)
    assert out is not None, raw[-500:]
    print(f"\n== 4. argument ({len(kept)} papers)")
    for part in ("thesis", "antithesis", "synthesis"):
        print(f"\n{part.upper()}\n{out[part]}")
    print(f"\nVERDICT: {out['verdict']}")
    rows, u5 = L.verify_claims(L.argument_text(out), docs)
    L._add(total, u5)
    print(f"\n== verification: {len(rows)} checks", dict(Counter(r["verdict"] for r in rows)))
    for r in rows:
        if r["verdict"] != "supported":
            print(f"  {r['verdict']:14s} {r['pmid']}: {r['claim'][:110]} | {r.get('issue', '')}")
    print(f"\ncalls {total.calls}, cost {100 * total.cost_usd:.2f} cents")


if __name__ == "__main__":
    main(*sys.argv[1:])
