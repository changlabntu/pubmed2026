"""
PubMed ingestion for the Module 1 RAG demo.

Two steps, mirroring the course's ingest.py:
  1. get the documents  (search_pmids / fetch_abstracts / parse_pubmed_xml)
  2. build the index    (build_index)

NCBI E-utilities docs: https://www.ncbi.nlm.nih.gov/books/NBK25499/
Rate limits: 3 requests/s without an API key, 10/s with one.
"""

import json
import os
import time
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv

load_dotenv()

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "llm-course-module1"
EMAIL = os.getenv("NCBI_EMAIL", "student@example.com")
API_KEY = os.getenv("NCBI_API_KEY")  # optional, raises the rate limit

QUERY = (
    '"lung cancer" AND ("deep learning" OR "artificial intelligence") '
    'AND ("low-dose CT" OR "low-dose computed tomography") AND screening '
    'NOT (Review[pt] OR "Systematic Review"[pt] OR Meta-Analysis[pt] '
    'OR Editorial[pt] OR Comment[pt] OR Letter[pt] OR News[pt])'
)


def _params(**extra):
    p = {"db": "pubmed", "tool": TOOL, "email": EMAIL}
    if API_KEY:
        p["api_key"] = API_KEY
    p.update(extra)
    return p


def _sleep():
    # stay under NCBI's rate limit
    time.sleep(0.11 if API_KEY else 0.34)


def search_pmids(term=QUERY, retmax=500, sort=None):
    """esearch: turn a PubMed query into a list of PMIDs. PubMed's default
    order is newest first, so a truncated list is all recent papers; pass
    sort="relevance" to get the best matches across years instead."""
    extra = {"sort": sort} if sort else {}
    r = requests.get(
        f"{EUTILS}/esearch.fcgi",
        params=_params(term=term, retmax=retmax, retmode="json", **extra),
        timeout=60,
    )
    r.raise_for_status()
    _sleep()
    result = r.json()["esearchresult"]
    return result["idlist"], int(result["count"])


def _post_with_retry(url, data, attempts=5, timeout=120):
    """NCBI occasionally answers 5xx under load. Retry with a growing pause
    instead of failing the whole ingest on one bad response."""
    for attempt in range(attempts):
        r = requests.post(url, data=data, timeout=timeout)
        if r.status_code < 500:
            r.raise_for_status()
            return r
        wait = 2 ** attempt
        print(f"NCBI returned {r.status_code}, retrying in {wait}s ...")
        time.sleep(wait)
    r.raise_for_status()
    return r


def fetch_abstracts(pmids, batch_size=200):
    """efetch: download PubMed records as XML, 200 PMIDs per request."""
    xml_chunks = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        r = _post_with_retry(
            f"{EUTILS}/efetch.fcgi",
            data=_params(id=",".join(batch), retmode="xml"),
        )
        _sleep()
        xml_chunks.append(r.text)
    return xml_chunks


def _text(el):
    """All text inside an element, including nested tags like <i>."""
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_pubmed_xml(xml_text):
    """Turn one efetch XML response into a list of document dicts."""
    root = ET.fromstring(xml_text)
    documents = []

    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        art = citation.find("Article")

        # Abstracts can be split into labeled sections (BACKGROUND, METHODS, ...)
        parts = []
        for sec in art.findall("Abstract/AbstractText"):
            label = sec.get("Label")
            body = _text(sec)
            if not body:
                continue
            parts.append(f"{label}: {body}" if label else body)
        abstract = "\n".join(parts)

        authors = []
        for a in art.findall("AuthorList/Author"):
            last = _text(a.find("LastName"))
            initials = _text(a.find("Initials"))
            collective = _text(a.find("CollectiveName"))
            if last:
                authors.append(f"{last} {initials}".strip())
            elif collective:
                authors.append(collective)

        journal = art.find("Journal")
        year = _text(journal.find("JournalIssue/PubDate/Year"))
        if not year:
            # e.g. "2023 Jan-Feb" or "Winter 2022"
            medline_date = _text(journal.find("JournalIssue/PubDate/MedlineDate"))
            digits = [t for t in medline_date.replace("-", " ").split() if t.isdigit()]
            year = digits[0] if digits else ""

        doi = ""
        for aid in article.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = _text(aid)

        documents.append(
            {
                "pmid": _text(citation.find("PMID")),
                "title": _text(art.find("ArticleTitle")),
                "abstract": abstract,
                "authors": authors,
                "first_author": authors[0] if authors else "",
                "journal": _text(journal.find("ISOAbbreviation")) or _text(journal.find("Title")),
                "year": int(year) if year else None,
                "mesh": [_text(m.find("DescriptorName")) for m in citation.findall("MeshHeadingList/MeshHeading")],
                "keywords": [_text(k) for k in citation.findall("KeywordList/Keyword")],
                "doi": doi,
            }
        )

    return documents


def load_corpus(pmids):
    """Fetch + parse + clean. Drops records without an abstract, de-duplicates by PMID."""
    documents = []
    for xml_text in fetch_abstracts(pmids):
        documents.extend(parse_pubmed_xml(xml_text))

    seen = set()
    cleaned = []
    for doc in documents:
        if not doc["abstract"] or doc["pmid"] in seen:
            continue
        seen.add(doc["pmid"])
        cleaned.append(doc)
    return cleaned


def build_index(documents):
    """minsearch index: title and abstract are searched, the rest is for filtering.

    Only review_pipeline.py uses this; the app does not. Imported lazily so the
    Streamlit deploy does not need minsearch (and its scikit-learn dependency)."""
    from minsearch import Index

    index = Index(
        text_fields=["title", "abstract"],
        keyword_fields=["pmid", "journal"],
        numeric_fields=["year"],
    )
    index.fit(documents)
    return index


def read_pmids(path="data/pmids.txt"):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def read_corpus(path="data/abstracts.json"):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # Instructor use: build the frozen corpus once.
    pmids, total = search_pmids()
    print(f"PubMed reports {total} hits, retrieved {len(pmids)} PMIDs")
    pmids = sorted(pmids, key=int)
    with open("data/pmids.txt", "w") as f:
        f.write("\n".join(pmids) + "\n")

    documents = load_corpus(pmids)
    with open("data/abstracts.json", "w") as f:
        json.dump(documents, f, indent=1, ensure_ascii=False)
    print(f"Saved {len(documents)} documents with abstracts to data/abstracts.json")
