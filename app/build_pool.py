"""
Instructor tool: build a paper pool from a PubMed query.

    python build_pool.py <name> "<pubmed query>"

Writes pools/<name>/query.txt, pmids.txt, meta.json (match count and
fetch date), abstracts.json, embeddings.npy.
Reviews, editorials, letters and similar are excluded automatically.
You still need to write pools/<name>/papers.json (question, anchor,
companions); see pick_anchor.py for help choosing.
"""

import json
import os
import random
import sys

import numpy as np
from dotenv import load_dotenv

load_dotenv(".env")

from pubmed_ingest import load_corpus, search_pmids  # noqa: E402
from lit_helper import embed_texts, pool_path, write_pool_meta  # noqa: E402

EXCLUDE = (' NOT (Review[pt] OR "Systematic Review"[pt] OR Meta-Analysis[pt] '
           'OR Editorial[pt] OR Comment[pt] OR Letter[pt] OR News[pt])')


def build_pool(name, query, retmax=500):
    os.makedirs(os.path.join("pools", name), exist_ok=True)
    pmids, total = search_pmids(query + EXCLUDE, retmax=retmax)
    pmids = sorted(pmids, key=int)
    print(f"PubMed reports {total} hits, retrieved {len(pmids)} PMIDs")
    with open(pool_path(name, "query.txt"), "w") as f:
        f.write(query + "\n")
    write_pool_meta(os.path.join("pools", name), query + EXCLUDE, total, pmids)   # pmids.txt, meta.json
    documents = load_corpus(pmids)
    with open(pool_path(name, "abstracts.json"), "w") as f:
        json.dump(documents, f, indent=1, ensure_ascii=False)
    vectors, usage = embed_texts([d["title"] + "\n" + d["abstract"] for d in documents])
    np.save(pool_path(name, "embeddings.npy"), vectors)
    print(f"{len(documents)} documents with abstracts, embedded with {usage.input_tokens} tokens")
    return documents


def companions(documents, anchor, seed=42, k=2):
    pool = [d["pmid"] for d in documents if d["pmid"] != anchor]
    return random.Random(seed).sample(pool, k)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    build_pool(sys.argv[1], sys.argv[2])
