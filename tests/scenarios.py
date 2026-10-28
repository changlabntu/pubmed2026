"""
Run the four moves on every scenario in scenarios.json against the real models
and score the output. Not collected by pytest.

    cd app && /opt/miniconda3/envs/CAMI/bin/python ../tests/scenarios.py [out_dir] [id ...]

For each scenario: fetch the PubMed corpus (cached under pools/custom), add the
given PMIDs (a null PMID is filled in: the anchor from the top hit for the
question, companions from the seed-42 picker), score the three, gather and
score the evidence, write the argument, verify it. Writes <out_dir>/<id>.json
with everything, and prints a summary table with these quality checks:

  parsed       every model reply parsed (no truncation, no prose)
  mix          not every scored paper has the same stance
  anchor_ok    the given anchor is not neutral
  verdict      the verdict, and whether it matched the scenario's expectation
  cites_ok     every PMID cited in the argument is in the scored set
  supported    share of verified claims marked supported
  flags        papers whose score and checks disagree
"""

import json
import os
import re
import sys
import time
from collections import Counter

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(HERE), "app")
sys.path.insert(0, APP)
load_dotenv(os.path.join(APP, ".env"))

import lit_helper as L  # noqa: E402

PMID_RE = re.compile(r"PMID\s*(\d{7,8})")


def run_one(sc, retmax=150):
    t0 = time.time()
    total = L.Usage(L.CHAT_MODEL)
    question, gap = sc["question"], {"gap": sc["gap"], "disproof": sc["disproof"]}
    docs, V, usage, src, pool_id = L.build_custom_pool(sc["query"], retmax, True)
    funnel_meta = L.pool_meta(pool_id)
    n_pool = len(docs)

    # fill in missing PMIDs: anchor = top hit for the question, companions = seed 42
    given = [p for p in sc["pmids"] if p]
    if given:
        docs, V, u_extra, missing = L.add_papers(docs, V, given)
        given = [p for p in given if p not in missing]
    anchor = sc["pmids"][0] if sc["pmids"][0] and sc["pmids"][0] in {d["pmid"] for d in docs} else None
    if anchor is None:
        hits, _ = L.search(question, docs, V, k=1)
        anchor = hits[0][0]["pmid"]
    comps = [p for p in given if p != anchor][:2]
    if len(comps) < 2:
        comps += [p for p in L.companions(docs, anchor, k=4) if p not in comps and p != anchor][: 2 - len(comps)]
    fixed = [anchor] + comps
    by = {d["pmid"]: d for d in docs}

    recs2, u2, raw2 = L.grade_relevance(question, [by[p] for p in fixed], gap=gap)
    L._add(total, u2)
    hits_q, _ = L.search(question, docs, V, k=6)
    hits_a = L.nearest(anchor, docs, V, k=6)
    seen, new_docs = set(fixed), []
    for d, _ in hits_q + hits_a:
        if d["pmid"] not in seen:
            seen.add(d["pmid"]); new_docs.append(d)
    recs3, u3, raw3 = L.grade_relevance(question, new_docs, gap=gap)
    L._add(total, u3)
    records = (recs2 or []) + (recs3 or [])
    parsed = recs2 is not None and recs3 is not None
    out, rows = None, []
    if records:
        kept, _ = L.select_records(records)
        funnel = L.funnel_line(funnel_meta, n_pool, len(records))
        out, u4, raw4 = L.argue(question, gap, kept, funnel)
        L._add(total, u4)
        parsed = parsed and out is not None
        if out is not None:
            rows, u5 = L.verify_claims(L.argument_text(out), docs)
            L._add(total, u5)

    stances = Counter(L.stance_of(r) for r in records)
    cited = set(PMID_RE.findall(L.argument_text(out))) if out else set()
    scored = {r["pmid"] for r in records}
    vcounts = Counter(r["verdict"] for r in rows)
    return {
        "id": sc["id"], "name": sc["name"], "expected": sc.get("expected"),
        "pool_id": pool_id, "pubmed_total": (funnel_meta or {}).get("total"), "n_pool": n_pool,
        "fixed": fixed, "fixed_titles": {p: by[p]["title"] for p in fixed},
        "n_scored": len(records), "stances": dict(stances),
        "scores": sorted(L.score_of(r) for r in records if L.score_of(r) is not None),
        "records": [{"pmid": r["pmid"], "author": f"{L.surname(r['doc']['first_author'])} {r['doc']['year']}",
                     "title": r["doc"]["title"], "stance": L.stance_of(r), "score": r.get("score"),
                     "bears": r.get("bears"), "why": r.get("why"), "gap_relation": r.get("gap_relation"),
                     "flag": r.get("flag")} for r in records],
        "argument": out, "verification": rows,
        "checks": {
            "parsed": parsed,
            "mix": len([k for k, v in stances.items() if v]) > 1,
            "anchor_ok": bool(records) and L.stance_of(records[0]) != "neutral",
            "verdict": out["verdict"] if out else None,
            "verdict_ok": bool(out) and out["verdict"] == sc.get("expected"),
            "cites_ok": bool(out) and cited <= scored, "uncited_or_unknown": sorted(cited - scored),
            "supported": round(vcounts.get("supported", 0) / len(rows), 2) if rows else None,
            "not_supported": vcounts.get("not supported", 0), "cant_tell": vcounts.get("can't tell", 0),
            "flags": sum(1 for r in records if r.get("flag")),
        },
        "calls": total.calls, "cost_cents": round(100 * total.cost_usd, 2), "seconds": round(time.time() - t0),
    }


def main(out_dir, ids=()):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(HERE, "scenarios.json")) as f:
        scenarios = json.load(f)
    results = []
    for sc in scenarios:
        if ids and sc["id"] not in ids:
            continue
        print(f"\n### {sc['id']}: {sc['name']}", flush=True)
        try:
            res = run_one(sc)
        except Exception as e:  # noqa: BLE001
            res = {"id": sc["id"], "name": sc["name"], "error": repr(e)}
            print("  ERROR", repr(e), flush=True)
        results.append(res)
        with open(os.path.join(out_dir, f"{sc['id']}.json"), "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        if "error" not in res:
            c = res["checks"]
            print(f"  anchor: {res['fixed'][0]} {res['fixed_titles'][res['fixed'][0]][:80]}")
            print(f"  pool {res['n_pool']} of {res['pubmed_total']} matched; scored {res['n_scored']}; stances {res['stances']}")
            print(f"  verdict: {c['verdict']} (expected {res['expected']}) | supported {c['supported']} | "
                  f"not supported {c['not_supported']} | flags {c['flags']} | {res['calls']} calls, {res['cost_cents']}c, {res['seconds']}s", flush=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k in ("id", "name", "expected", "n_scored", "stances", "checks",
                                                       "calls", "cost_cents", "error")} for r in results],
                  f, indent=1, ensure_ascii=False)
    print("\n| id | scored | thesis/anti/neutral | verdict | expected | ok | supported | not sup | flags | cents |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            print(f"| {r['id']} | ERROR {r['error'][:60]} |")
            continue
        c, s = r["checks"], r["stances"]
        print(f"| {r['id']} | {r['n_scored']} | {s.get('thesis', 0)}/{s.get('antithesis', 0)}/{s.get('neutral', 0)} | "
              f"{c['verdict']} | {r['expected']} | {'yes' if c['verdict_ok'] else 'no'} | {c['supported']} | "
              f"{c['not_supported']} | {c['flags']} | {r['cost_cents']} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "scenario_results"), sys.argv[2:])
