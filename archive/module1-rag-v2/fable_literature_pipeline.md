# Fable literature review pipeline

How an independent Fable 5.1 agent answered a literature-review question over
181 PubMed abstracts on 2026-10-18. Full log with exact prompts, token counts,
and the per-PMID verification table: `reference/independent_review.md`.

Question: Who has validated an AI system for lung cancer screening in a real
screening cohort?

## Steps

1. **Inspect corpus.** Python. 181 abstracts loaded, about 93K tokens.
2. **Rank by similarity.** 1 OpenAI embedding call plus minsearch keyword
   search, fused with reciprocal rank fusion. Used as a cross-check only.
3. **Screen every abstract.** 181 Haiku calls, one per abstract, 8 in
   parallel, cached system prompt. Each returns JSON: include, partial, or
   exclude, with cohort, n, setting, AI task, headline result, and a verbatim
   evidence quote. Kept 39 includes.
4. **Check the screener.** Fable judges. Read the 7 top-ranked records the
   screener excluded, confirmed all 7, rescued 1 trial protocol for the gaps
   paragraph.
5. **Read and select.** Fable judges. Read 26 candidate abstracts in full,
   dropped 15, kept 25.
6. **Write the review.** Fable writes. Under 400 words, every claim cited
   as [PMID n].
7. **Verify each claim.** 25 Haiku calls plus 7 reruns. One claim against
   its one cited abstract, instructed to be literal about numbers. Found 3
   over-claims and 2 wording errors. Fable rewrote those sentences.
8. **Number check.** Python. Every number in each citing clause must appear
   in the cited abstract. 4 flags, all rounding or formatting.

## Division of labour

| Who | Steps | Calls | Role |
|---|---|---|---|
| Python | 1, 2, 8 | 1 embedding | deterministic checks |
| Haiku 5.5 | 3, 7 | 213 | volume work with structured JSON output |
| Fable 5.1 | 4, 5, 6, 7 | 0 API calls | judgment and writing, as the agent itself |

Total cost about 6 cents. Wall time about 10 minutes.

## Compared with the top-k pipeline in `review_pipeline.py`

| | Top-k pipeline | This pipeline |
|---|---|---|
| Retrieval | three top-15 searches, pool of 36 | screened all 181 |
| Papers cited | 12 | 25, including all 12 |
| Verification | cited PMID is in context | each claim checked against its abstract |
| Cost | 0.6 cents | 6 cents |
| Scales to | tens of thousands | a few thousand |

Eleven of the thirteen extra papers never entered the top-k pool. Retrieval,
not grading, was the bottleneck. The claim-level verification is the one step
worth porting to the cheaper pipeline.

## Gotchas

- Haiku 5.5 emits a thinking block before its text even at low effort. Set
  `max_tokens` to 4096 or more, or the answer comes back empty.
- Haiku can emit an unescaped quote inside JSON. Retry once on parse failure.
- Prompt caching needs a minimum prefix length. A short system prompt will not
  cache, and the cost line will show cache_read = 0.
