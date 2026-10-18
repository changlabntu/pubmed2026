# Independent mini literature review
**Question:** Who has validated an AI system for lung cancer screening in a real screening cohort?
Corpus: data/abstracts.json (181 PubMed records). Date: 2026-10-18. Author: Claude (Fable 5.1) as orchestrator; Claude Haiku 5.5 for screening and claim-checking; OpenAI text-embedding-3-small for one query embedding.
---

## 1. Review (for clinicians)
**Malignancy-risk models.** An NLST-trained nodule-risk deep-learning algorithm tested on pooled Danish, MILD and Dutch-Belgian baseline scans (4146 participants) gave AUC 0.94-0.98 and 39% fewer false positives than PanCan among indeterminate nodules [PMID 40956165]. Related algorithms reached AUC 0.92 in 599 DLCST participants [PMID 40833260], 0.91-0.94 in DLCST/MILD with a prior CT [PMID 37526548], and outperformed 65 radiologists' mean on 300 indeterminate nodules (AUC 0.78 vs 0.70) [PMID 42340186]. Sybil was validated on NLST hold-out, MGH and Chang Gung LDCTs (1-year AUC 0.86-0.94) [PMID 36634294], then in Asian health-check/screening cohorts of 18,057 [PMID 40762850] and 21,087 [PMID 42140481] and four screening programmes (22,469 participants; three external) [PMID 41833807]. An end-to-end model reached 94.4% AUC on 6716 NLST cases [PMID 31110349]; a follow-up-risk model was blind-validated in 2294 PanCan participants (1-year AUC 0.97) [PMID 32864596].

**First reader / detection.** Volumetric AI was tested as first reader on 4104 BioMILD baseline scans (sensitivity 91.2%, specificity 75.7% vs 90.0% for humans, NPV 99.8%, 74.7% workload reduction) [PMID 41308576], 3678 4-IN-THE-LUNG-RUN scans (negative misclassification 0.8% vs 11.1% for radiologists) [PMID 39754864], 283 Moscow ultra-LDCT screenees [PMID 35123156] and 1252 UKLS participants [PMID 41872344]. A deep-learning CAD beat double reading in a Chinese programme (360 nodule-enriched scans; sensitivity 90.1% vs 76.0%, 1.0 vs 0.04 false positives/scan) [PMID 34871936]; a commercial detector found 40/43 Japanese screen-detected cancers at 7 false positives/scan [PMID 39613978].

**Prospective / real-world.** A Korean randomised trial (911 health-checkup LDCTs) found AI assistance raised Lung-RADS-positive detection (16.9% vs 10.3%) without changing reading time [PMID 41848489]. A prospective 12,360-participant mobile-CT programme [PMID 35866406] and a 259,121-person West China cohort comparing AI-assisted thin-section with conventional LDCT [PMID 40213646] report AI-assisted screening.

**Follow-up rounds.** On UKLS follow-up scans, AI matched 83.5% of persisting nodules [PMID 42668303] and beat human readers on volume-doubling-time referral (3-month negative misclassification 1/11 vs 18-64%) [PMID 41319449].

**Themes and gaps.** Validations are almost all retrospective, re-use the same trials (NLST to train; DLCST, MILD, UKLS, BioMILD to test) and benchmark against PanCan or radiologists, not outcomes. Performance falls in never/light smokers (6-year AUC 0.56) [PMID 40762850], without a baseline nodule (AUC 0.64) [PMID 41833807] and outside screening [PMID 39907586]; nodule-type classification depends on reconstruction kernel [PMID 40883523], and commercial systems disagree widely (sensitivity 43-77%) [PMID 41815002]. The sole randomised trial recorded no cancers [PMID 41848489]; a 2722-patient RCT is registered but unreported [PMID 38355174]. Outcome-based prospective trials remain the gap.

*(397 words, 25 PMIDs cited.)*

---

## 2. Pipeline
Design principle: the corpus is small (181 records, ~93k tokens), so instead of retrieving a top-k slice and risking misses I screened **every** abstract with a cheap model, used embedding/BM25 ranking only as an independent cross-check, read the candidate abstracts myself, wrote the review by hand, and then had a second, separate model pass fact-check every claim-citation pair against its abstract.

### Step 0 - Inspect corpus (Python, no model)
- Loaded `data/abstracts.json`: 181 records, all with non-empty abstracts; years 2005-2026 (86 from 2025-26). Approx. 93k tokens of title+abstract.

### Step 1 - Hybrid retrieval ranking (OpenAI embedding + minsearch BM25; cross-check only)
- Script: `scratchpad/rank.py`. Embedded one query with `text-embedding-3-small` (42 tokens, cost ~$0.000001) and ranked all 181 rows of `data/embeddings.npy` by cosine similarity. Separately ran a minsearch keyword search (title+abstract fields) over all 181. Fused the two rank lists with reciprocal-rank fusion (k=60).
- Embedding query text: `Validation of an artificial intelligence deep learning system in a real lung cancer screening cohort (NLST, NELSON, national LDCT screening programme participants): external test, prospective evaluation, sensitivity, specificity, AUC`
- BM25 query text: `validation screening cohort NLST NELSON prospective external test sensitivity specificity deep learning`
- Considered 181, kept 181 (ranked). Used later to (a) order the screening output and (b) check whether anything highly ranked had been excluded by the screener.

### Step 2 - Full-corpus screening (claude-haiku-5-5, effort=low, max_tokens=4096, one call per abstract, 8 threads)
- Script: `scratchpad/screen.py`. Every one of the 181 abstracts was sent individually with the system prompt below (cached with `cache_control`) and a user message of the form `PMID: {pmid}\nTitle: {title}\nYear: {year}\nJournal: {journal}\nAbstract: {abstract}`. Output was parsed as JSON.
- Result: include 39, partial 71, exclude 71, errors 0.
- Tokens (summed over 181 calls): input (uncached) 140,357; cache-read 115,020; cache-write 639; output 68,782. Cost ~$0.0497 (Haiku 5.5: $0.10/M input, $0.50/M output, cache read $0.01/M, cache write $0.125/M).
- Exact system prompt:

```
You are screening PubMed abstracts for a clinical literature review. The review question is:
"Who has validated an AI system for lung cancer screening in a real screening cohort?"

Definition of INCLUDE: the abstract reports an empirical evaluation (validation, external test, prospective or retrospective deployment, reader study with real cases) of an AI / deep-learning / machine-learning / CAD system, where the evaluation data come from an actual lung cancer screening population (e.g. NLST, NELSON, I-ELCAP, SUMMIT, UKLS, LDCT screening programme participants, screening trial, national screening program, opportunistic/health-check LDCT screening cohort). Simulation, phantom, non-screening diagnostic cohorts (e.g. incidental nodules, biopsy-referred patients), reviews, editorials, protocols without results, and purely methodological papers are EXCLUDE. PARTIAL: AI evaluated on a mixed or unclear cohort, or screening cohort used but the AI task is only tangentially about screening (e.g. only emphysema quantification).

Respond with ONLY a JSON object, no prose, with these keys:
{
 "decision": "include" | "partial" | "exclude",
 "cohort": "<name/description of the screening cohort(s) used for evaluation, or null>",
 "n": "<number of participants or scans in the evaluation set as stated, or null>",
 "setting": "retrospective" | "prospective" | "reader_study" | "simulation" | "other" | null,
 "ai_task": "<nodule detection / malignancy risk / Lung-RADS categorisation / triage / eligibility / other - brief>",
 "headline_result": "<the key quantitative result with numbers exactly as written, or null>",
 "evidence_quote": "<a verbatim sentence from the abstract that justifies the decision>",
 "reason": "<one sentence>"
}
```

### Step 3 - Cross-check screener against retrieval ranking (Python, no model)
- Listed every record in the fused top-40 that the screener had excluded (7 records). I read the screener's one-line reasons: LUNA16/LIDC benchmark papers, a never-smoker screening study without AI, two blood-test ML papers, a conventional-risk-model eligibility study and an RCT protocol. All exclusions were correct for the question, but I decided to cite the RCT protocol [PMID 38355174] in the gaps paragraph as 'registered but unreported' after reading its abstract myself.

### Step 4 - Manual reading and selection (me, no model)
- From the 39 'include' records (plus the protocol) I printed the full title/abstract of 26 candidates ordered by fused rank and read them all. I dropped 15 lower-value 'includes' (small NLST sub-samples, challenge summaries, methodology papers, body-composition/CAC papers where the AI task is not screening) and kept 24 + the protocol = 25 papers to cite. Selection criterion: named real screening cohort (trial or programme), stated n, and a quantitative headline result on a screening-relevant AI task.
- I wrote the review text myself from the abstracts (no generation model). Word count trimmed iteratively to under 400.

### Step 5 - Claim-level verification (claude-haiku-5-5, effort=low, max_tokens=4096, one call per claim)
- Script: `scratchpad/verify.py`. I decomposed the review into 25 atomic claims (one per cited PMID, `scratchpad/claims.json`) and sent each with only its cited abstract. The verifier was told to be literal about numbers and cohort names.
- Round 1 (25 claims): input 28,177, output 6,721 tokens, cost ~$0.0062. Verdicts: supported 14, partially_supported 11, unsupported 0.
- I acted on the substantive 'partial' findings (see Section 3), rewrote 7 claims/sentences, and re-ran those 7. Round 2: input 7,682, output 2,357 tokens, cost ~$0.0019. Verdicts: supported 5, partially 2 (both were about omitted context, not wrong numbers - see Section 3).
- An aborted first attempt at round 1 (5 calls before a JSON parse crash) cost roughly $0.001 extra (approx. 5,500 input / 1,300 output tokens).
- Exact system prompt:

```
You are a strict fact-checker for a literature review. You will be given one claim and the single PubMed abstract it cites. Judge ONLY from the abstract text. Respond with ONLY a JSON object:
{"verdict": "supported" | "partially_supported" | "unsupported", "quote": "<verbatim abstract text that supports or contradicts the claim>", "issue": "<any number, cohort, or wording in the claim that the abstract does not state, or null>"}
Be literal: every number and named cohort in the claim must appear in or be directly derivable from the abstract.
```

- User message format: `CLAIM: {claim}\n\nABSTRACT (PMID {pmid}, {first_author} {year}):\nTitle: {title}\n{abstract}`

### Step 6 - Deterministic number check (Python, no model)
- For every `[PMID n]` in the review, extracted all numbers in the preceding clause and checked they appear literally in that abstract. 4 flags, all benign: '18,057' and '259,121' are written with thin spaces in the abstracts; '0.97' is 0·968 rounded; '64' is 63.6 rounded.

### Totals
- Model calls: 181 (screen) + 25 + 7 + 5 (verify, incl. aborted) = 218 Haiku calls, 1 OpenAI embedding call, 0 Sonnet calls.
- Total cost ~$0.059 (dominated by screening).

### What went wrong
1. The verifier crashed on the 5th claim because Haiku emitted an unescaped double quote inside the JSON `quote` field. Fixed by adding a one-shot retry and a PARSE_FAIL fallback; re-ran from scratch.
2. Prompt caching silently did nothing for the verifier (cache_read = 0) because its system prompt is below Haiku's minimum cacheable prefix; cost impact negligible. Caching did work for the screener (115k cache-read tokens).
3. First draft of the review was 442 words and over-claimed in three places the verifier caught: the 39% false-positive reduction in [40956165] applies to the indeterminate-nodule subset; 22,469 in [41833807] is the total across all four programmes, not the validation set; [40213646] does not say all 259,121 were AI-read. All three were reworded. I also wrongly called '180 malignant nodules' '180 cancers' and called the BioMILD software 'commercial' when the abstract does not say so; both fixed.
4. The screener labelled the Korean RCT [41848489] a 'health checkup' rather than a formal screening programme - correct, and I kept that qualifier in the review.

---

## 3. Cited PMIDs and verification notes
Columns: fused retrieval rank (0 = best), screener decision, Haiku verifier verdict on the final claim, and my own manual check of the abstract text.

| PMID | First author, year | Rank | Screener | Verifier (final) | Manual check against abstract |
|---|---|---|---|---|---|
| 40956165 | Antonissen N 2025 | 0 | include | supported | Abstract: pooled 4146 participants, 180 malignant nodules; AUCs 0.98/0.96/0.94; '39.4% relative reduction in false-positive findings' reported in subset A (indeterminate nodules) - review wording now says 'among indeterminate nodules'. |
| 39613978 | Fukumoto W 2025 | 1 | include | supported | Abstract: 43 lung cancers, DL-LND detected 40 (93.0%), mean 7.0 FP nodules per scan; Japanese LCS LDCT. |
| 32864596 | Huang P 2019 | 5 | include | supported | Abstract: 'Double-blinded validation ... 2,294 participants from PanCan'; 1-year AUC 0·968 (review rounds to 0.97). |
| 40833260 | Obreja B 2025 | 6 | include | partially_supported | Abstract: external test 599 DLCST participants, AUC 0.92 (full NLST training). |
| 42340186 | Peeters D 2026 | 10 | include | supported | Abstract: external test nodules from three European screening trials; observer study 300 nodules; AI AUC 0.78 vs 65 radiologists' mean 0.70. |
| 40762850 | Lee JH 2025 | 12 | include | partially_supported | Abstract: 18 057 individuals from a single medical checkup facility (health-check cohort, hence 'health-check/screening'); never/light-smoking subgroup AUC 0.56 for future cancers in the 6-year analysis. |
| 41848489 | Hwang EJ 2026 | 15 | include | supported | Abstract: prospective randomised trial, 911 individuals, self-initiated health checkup LDCT, Korean registry KCT0010206; interpretation time 187 vs 172 s (p=.23); Lung-RADS-positive 16.9% vs 10.3%; no lung cancer diagnosed. |
| 38355174 | Benzaquen J 2024 | 17 | exclude | partially_supported | Abstract: RCT protocol, 2722 patients to be included, NCT05704920; no results. |
| 42668303 | Jiang B 2026 | 19 | include | partially_supported | Abstract: UKLS 3-month follow-up; 83.5% (283/339) matching success for persisting nodules. |
| 35866406 | Shao J 2022 | 22 | include | supported | Abstract: prospectively recruited, 12,360 participants scanned by mobile CT; DL system for nodule detection and risk stratification. |
| 35123156 | Lancaster HL 2022 | 23 | include | supported | Abstract: 283 participants, Moscow Lung Cancer Screening, ultra-LDCT baseline, AI (AVIEW LCS) as standalone reader. |
| 41308576 | Ledda RE 2026 | 26 | include | supported | Abstract: 4104 BioMILD baseline LDCTs; sensitivity 91.2%/specificity 75.7% (AI) vs 89.7%/90.0% (human); NPV 99.8%; workload reduction 74.7%; title 'AI as first reader'. |
| 31110349 | Ardila D 2019 | 27 | include | supported | Abstract: '94.4% area under the curve on 6,716 National Lung Cancer Screening Trial cases'. |
| 34871936 | Cui X 2022 | 28 | include | supported | Abstract: 180 with nodules + 180 without = 360 (nodule-enriched); DL-CAD 90.1% at 1.0 FP/scan vs double reading 76.0% at 0.04 FP/scan. |
| 41319449 | Jiang B 2026 | 29 | include | supported | Abstract: UKLS follow-up; AI 3-month NM 1/11 (9.1%) vs human readers 18.2-63.6% (review rounds to 18-64%). |
| 37526548 | Venkadesh KV 2023 | 31 | include | supported | Abstract: external test sets from DLCST and MILD, AUCs 0.91 and 0.94 for current+prior CT model. |
| 36634294 | Mikhael PG 2023 | 38 | include | supported | Abstract: validated on NLST heldout (6,282), MGH (8,821), CGMH (12,280); 1-year AUC 0.92/0.86/0.94 -> range 0.86-0.94. |
| 40213646 | Ji G 2025 | 44 | include | partially_supported | Abstract: real-world cohort at West China Hospital, 259 121 non-high-risk participants screened with LDCT-TRAI (thin-section + AI) or traditional LDCT. |
| 41833807 | Phellan Aro R 2026 | 48 | include | supported | Abstract: 4 screening programs, 1 training + 3 external validation; 22,469 participants total (not validation-only - review says 'four screening programmes (22,469 participants; three external)'); AUC 0.64 year 6 when nodules absent. |
| 41815002 | Khurelsukh K 2026 | 50 | include | supported | Abstract: 3 commercial AI systems, 100 LDCT screening subjects; sensitivities 77.0%, 72.2%, 42.9% -> '43-77%'. |
| 39754864 | Walstra ANH 2025 | 77 | include | partially_supported | Abstract: 3678 baseline 4ITLR scans; NMs 31 (0.8%) AI vs 407 (11.1%) radiologists. |
| 39907586 | Li TZ 2025 | 90 | include | supported | Abstract: single-time-point CT AI performed well for screening-detected nodules but did not generalize to other clinical settings. |
| 40883523 | Mao Y 2026 | 100 | include | supported | Abstract: NELCIN-B3 baseline LDCT, 300 participants; kernel changed nodule-type classification agreement (87.7% vs 17.7%) but not detection rate. |
| 42140481 | Kim YW 2026 | 101 | include | supported | Abstract: 21,087 individuals who underwent LDCT screening ('pragmatic Asian LCS cohort'). |
| 41872344 | Jiang B 2026 | 144 | include | supported | Abstract: 1252 UKLS participants analysed (Methods says 1250 scans), commercially available AI in AI-first reader workflow. |

Notes on the remaining 'partially_supported' verdicts (none involve a wrong number):
- [40762850]: verifier noted 18,057 is the whole cohort while the AUC 0.56 is a never/light-smoker subgroup; the review cites the two facts in separate sentences and labels the 0.56 as the never/light-smoker result.
- [40213646]: verifier noted the abstract does not give the split between AI-assisted and conventional arms; the review only says the cohort 'compared' the two, which is what the abstract states.
- [40833260], [41308576], [39754864], [42668303], [41319449], [38355174] round-1 partials were about omitted context (e.g. FP trade-off, definition of negative misclassification, small denominators); I added the context where it changes interpretation (FP/scan for [34871936], specificity for [41308576], '1/11' for [41319449]) and otherwise left the numbers, which the verifier confirmed match.

How verification was done overall: (1) I read all 25 cited abstracts in full before writing; (2) each claim was independently checked by Haiku 5.5 against only its cited abstract with a literal-numbers instruction; (3) a script confirmed every number in each citing clause occurs in the cited abstract (modulo thin-space formatting and two explicit roundings); (4) every substantive discrepancy found in (2) was corrected and re-checked.
