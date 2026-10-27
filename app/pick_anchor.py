"""
Instructor tool: help choose the anchor paper for a pool.

    python pick_anchor.py <pool> "<question>"

Embeds the question, grades its 10 nearest papers with Haiku, and prints the
HIGH ones with their rank. Pick one, then write pools/<pool>/papers.json:
    {"question": ..., "anchor": <pmid>, "companions": [...], "companion_seed": 42}
The companions come from build_pool.companions(documents, anchor).
"""

import json
import sys

from dotenv import load_dotenv

load_dotenv(".env")

import lit_helper as L  # noqa: E402
from build_pool import companions  # noqa: E402


def main(pool, question, k=10):
    docs, V, _, _ = L.build_corpus(pool)
    hits, _ = L.search(question, docs, V, k=k)
    recs, usage, _ = L.grade_relevance(question, [d for d, _ in hits])
    by = {r["pmid"]: r for r in recs}
    print(f"{len(docs)} documents in pool '{pool}'. Top {k} for the question:\n")
    for rank, (d, sim) in enumerate(hits, 1):
        r = by.get(d["pmid"], {})
        print(f"{rank:2d} {sim:.3f} {d['pmid']} {r.get('grade', '?'):6s} {d['first_author']} {d['year']}: {d['title'][:75]}")
        if r.get("grade") == "HIGH":
            print(f"          n={r.get('n')} | {r.get('headline_result')}")
    highs = [d["pmid"] for d, _ in hits if by.get(d["pmid"], {}).get("grade") == "HIGH"]
    if highs:
        print(f"\nSuggested anchor: {highs[0]} (highest-ranked HIGH). Companions with seed 42:",
              companions(docs, highs[0]))
    print(f"\n{usage.calls} Haiku call(s), ${usage.cost_usd:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
