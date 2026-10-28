"""
Instructor tool: help choose the anchor paper for a pool.

    python pick_anchor.py <pool> "<question>" ["<proposed gap>" "<what would disprove it>"]

Embeds the question, has Haiku score its 10 nearest papers 1-10 against
the gap, and prints them with their rank. Pick one (an antithesis paper, the
closest to closing the gap, else a thesis paper), then write pools/<pool>/papers.json:
    {"question": ..., "anchor": <pmid>, "companions": [...], "companion_seed": 42}
The companions come from build_pool.companions(documents, anchor).
"""

import json
import sys

from dotenv import load_dotenv

load_dotenv(".env")

import lit_helper as L  # noqa: E402
from build_pool import companions  # noqa: E402


def main(pool, question, gap=None, disproof=None, k=10):
    docs, V, _, _ = L.build_corpus(pool)
    hits, _ = L.search(question, docs, V, k=k)
    gapctx = {"gap": gap, "disproof": disproof} if gap else None
    recs, usage, _ = L.grade_relevance(question, [d for d, _ in hits], gap=gapctx)
    by = {r["pmid"]: r for r in recs}
    print(f"{len(docs)} documents in pool '{pool}'. Top {k} for the question:\n")
    for rank, (d, sim) in enumerate(hits, 1):
        r = by.get(d["pmid"], {})
        print(f"{rank:2d} {sim:.3f} {d['pmid']} {L.stance_of(r) or '?':10s} {d['first_author']} {d['year']}: {d['title'][:75]}")
        if r.get("why"):
            print(f"          {r.get('why')}" + (f" | FLAG: {r['flag']}" if r.get("flag") else ""))
    for stance in ("antithesis", "thesis"):
        picks = [d["pmid"] for d, _ in hits if L.stance_of(by.get(d["pmid"], {})) == stance]
        if picks:
            print(f"\nSuggested anchor: {picks[0]} (highest-ranked {stance}). Companions with seed 42:",
                  companions(docs, picks[0]))
            break
    print(f"\n{usage.calls} Haiku call(s), ${usage.cost_usd:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:5])
