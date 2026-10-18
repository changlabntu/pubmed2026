# Mini literature review: trace and template

Question: **Who has validated an AI system for lung cancer screening in a real screening cohort?**

Corpus: 181 PubMed abstracts in `pools/lung_ct/abstracts.json`, embedded with
text-embedding-3-small in `pools/lung_ct/embeddings.npy`. LLM: Claude Haiku 5.5.
Script: `review_pipeline.py`. Raw trace: `reference/review_trace.json`.

Total spend for the whole review: 49,193 input tokens,
6,719 output tokens, about 0.8 cents.

## Steps

1. **Embedding search with the raw question, k=15.** Embed the question, cosine
   against all 181 vectors. Anchor rank: 2.

| PMID | Paper | Title |
|---|---|---|
| 38355174 | Benzaquen J 2024 | Integrating artificial intelligence into lung cancer screening: a rand |
| 41308576 | Ledda RE 2026 | Potential for AI as first reader in lung cancer screening. |
| 35406501 | Ziegelmayer S 2022 | Cost-Effectiveness of Artificial Intelligence Support in Computed Tomo |
| 33240975 | Guo J 2020 | DeepLN: an artificial intelligence-based automated system for lung can |
| 42340186 | Peeters D 2026 | Benchmarking of AI and Radiologists for Indeterminate Lung Nodule Mali |
| 41319449 | Jiang B 2026 | AI performance for nodule volume doubling time in the follow-up of the |
| 40213646 | Ji G 2025 | Early-stage lung cancer detection via thin-section low-dose CT reconst |
| 33658025 | Chamberlin J 2021 | Automated detection of lung nodules and coronary artery calcium using  |
| 42668303 | Jiang B 2026 | Automated artificial intelligence performance for longitudinal pulmona |
| 39165498 | Jin Y 2024 | Development and validation of an integrated system for lung cancer scr |

2. **Query rewrite, then embedding search, k=15.** Haiku writes a 3-sentence
   hypothetical abstract; embed that instead. Anchor rank: 42.
   Lesson: the rewrite pulled toward nodule-level malignancy studies and away
   from the first-reader study, so a rewrite changes *what kind* of paper you
   find. Keep both lists.

   Hypothetical abstract:
   > We retrospectively applied a deep learning nodule-detection and malignancy-risk model to low-dose CT scans from a population-based lung cancer screening program (n = 12,400 participants, 2015–2021), with histopathology and two-year follow-up as the reference standard. Performance was measured as per-nodule sensitivity and specificity, AUROC for cancer diagnosis within one year, and the proportion of benign nodules correctly downgraded from Lung-RADS category 4 or higher, compared with radiologist readings. The AI system matched radiologist sens

| PMID | Paper | Title |
|---|---|---|
| 42340186 | Peeters D 2026 | Benchmarking of AI and Radiologists for Indeterminate Lung Nodule Mali |
| 36672655 | Chao HS 2023 | Artificial Intelligence Assisted Computational Tomographic Detection o |
| 33482120 | Adams SJ 2021 | Development and Cost Analysis of a Lung Nodule Management Strategy Com |
| 34264825 | Balagurunathan Y 2021 | Lung Nodule Malignancy Prediction in Sequential CT Scans: Summary of I |
| 29713615 | Ali I 2018 | Lung Nodule Detection via Deep Reinforcement Learning. |
| 42668303 | Jiang B 2026 | Automated artificial intelligence performance for longitudinal pulmona |
| 32788705 | Cui S 2020 | Development and clinical application of deep learning model for lung n |
| 37526548 | Venkadesh KV 2023 | Prior CT Improves Deep Learning for Malignancy Risk Estimation of Scre |
| 40956165 | Antonissen N 2025 | External Test of a Deep Learning Algorithm for Pulmonary Nodule Malign |
| 33895622 | Trajanovski S 2021 | Towards radiologist-level cancer risk assessment in CT lung screening  |

3. **Keyword search, minsearch TF-IDF, k=15.** Title boosted 2x.
   Anchor rank: 7.

| PMID | Paper | Title |
|---|---|---|
| 33240975 | Guo J 2020 | DeepLN: an artificial intelligence-based automated system for lung can |
| 39165498 | Jin Y 2024 | Development and validation of an integrated system for lung cancer scr |
| 40213646 | Ji G 2025 | Early-stage lung cancer detection via thin-section low-dose CT reconst |
| 34871936 | Cui X 2022 | Performance of a deep learning-based lung nodule detection system as a |
| 31558626 | Ruparel M 2019 | Evaluation of cardiovascular risk in a lung cancer screening cohort. |
| 40001094 | Huang F 2025 | The feasibility and cost-effectiveness of implementing mobile low-dose |
| 41308576 | Ledda RE 2026 | Potential for AI as first reader in lung cancer screening. |
| 36634294 | Mikhael PG 2023 | Sybil: A Validated Deep Learning Model to Predict Future Lung Cancer R |
| 37489991 | Xu K 2023 | AI Body Composition in Lung Cancer Screening: Added Value Beyond Lung  |
| 31675322 | Ozdemir O 2020 | A 3D Probabilistic Deep Learning System for Detection and Diagnosis of |

4. **Union of candidates.** 36 unique PMIDs from the three lists.
   No search alone found everything the graders later rated HIGH.

5. **Grade every candidate with Haiku, 10 per call.** Prompt asks for one line
   per paper: `PMID | HIGH/MEDIUM/LOW | why`. HIGH means an AI system evaluated
   on scans from an actual screening program or trial, with results.
   Result: 12 HIGH, 10 MEDIUM, 14 LOW.

| PMID | Paper | Grade | Why |
|---|---|---|---|
| 38355174 | Benzaquen J 2024 | LOW | Protocol for a planned RCT of AI in a real-life screening setting with no validation results yet reported. |
| 41308576 | Ledda RE 2026 | HIGH | Retrospective validation of an AI first-reader on 4104 baseline LDCTs from the BioMILD screening trial with sensitivity, specificity, and PPV/NPV results. |
| 35406501 | Ziegelmayer S 2022 | LOW | Markov cost-effectiveness model of AI-supported screening CT, not a validation of AI performance in a screening cohort. |
| 33240975 | Guo J 2020 | MEDIUM | DeepLN is evaluated on a large LDCT dataset with retrospective clinical comparisons, but the data are not clearly from an actual screening program. |
| 42340186 | Peeters D 2026 | HIGH | External test on nodules from three European lung screening trials, with a reader study comparing AI to 65 radiologists. |
| 41319449 | Jiang B 2026 | HIGH | Validates AI nodule volume doubling time on protocol-mandated follow-up scans from the UK Lung Cancer Screening trial against expert panel and histology. |
| 40213646 | Ji G 2025 | HIGH | Real-world screening cohort of 259,121 participants comparing AI-assisted thin-section LDCT with traditional LDCT, with cancer detection and survival outcomes. |
| 33658025 | Chamberlin J 2021 | MEDIUM | AI nodule detection validated against experts on 117 retrospective LDCT screening scans, but the cohort is small and it is a prototype with focus on cardiac outcomes. |
| 42668303 | Jiang B 2026 | HIGH | Evaluates AI longitudinal nodule matching in all 361 UK Lung Cancer Screening trial participants with 3-month follow-up scans. |
| 39165498 | Jin Y 2024 | MEDIUM | Proof-of-concept case-control study combining liquid biopsy with an AI-aided CT nodule model, with validation largely on blood-based and selected cohorts rather than a screening program. |
| 35123156 | Lancaster HL 2022 | HIGH | AI (AVIEW LCS) was evaluated as a standalone reader on 283 participants from the Moscow Lung Cancer Screening program's baseline ultra-LDCT scans, with direct comparison to radiologists' misclassification rates. |
| 35311112 | Ye M 2022 | LOW | Develops a multi-factor classifier including an AI risk score and liquid biopsy for nodule diagnosis, but the AI is only one component and the cohort is a prospective diagnostic study rather than an AI validation in a screening program. |
| 41749798 | Lei F 2026 | LOW | Uses AI to build educational modules for screening uptake; it does not evaluate an AI system for lung cancer image interpretation. |
| 36672655 | Chao HS 2023 | MEDIUM | AI-assisted CT detection was tested in a 200-person clinical trial at two hospitals, showing increased nodule sensitivity, but it is a hospital-embedded trial rather than a defined screening program cohort. |
| 39754864 | Walstra ANH 2025 | HIGH | AI (AVIEW-LCS) was validated as a first reader on 3678 baseline LDCTs from the 4-IN-THE-LUNG-RUN screening trial, with negative-misclassification and referral outcomes against radiologists. |
| 33482120 | Adams SJ 2021 | MEDIUM | Combines AI risk scores with Lung-RADS using retrospective baseline screening CTs from 192 studies weighted to a screening population, but it is a secondary cost/management modeling analysis rather than direct validation in a screening cohort. |
| 34264825 | Balagurunathan Y 2021 | LOW | ISBI 2018 challenge on malignancy prediction using NLST-derived data, with algorithms evaluated on a small curated test set rather than an AI system validated in a screening program. |
| 29713615 | Ali I 2018 | LOW | Nodule detection on the LIDC/IDRI diagnostic dataset with poor test performance; it is not a screening cohort validation. |
| 32788705 | Cui S 2020 | MEDIUM | Deep learning nodule screening on LDCT with a clinical application component, but external testing was on the public LUNA dataset and the abstract does not clearly report validation in a real screening cohort. |
| 37526548 | Venkadesh KV 2023 | MEDIUM | Deep learning malignancy risk estimation validated on external screening trial data (DLCST and MILD), though it uses size-matched enriched subsets rather than a full screening cohort and is a risk model rather than a screening reader. |
| 40956165 | Antonissen N 2025 | HIGH | Externally tests a DL nodule malignancy algorithm on baseline scans from three European screening trials (DLCST, MILD, NELSON-type Dutch-Belgian NELSON), with AUC results versus PanCan. |
| 33895622 | Trajanovski S 2021 | MEDIUM | Develops a DL lung cancer risk model validated on the LHMC and Kaggle datasets and compared to radiologists, but the test sets are not clearly from an actual screening program or trial. |
| 36909593 | Mukherjee P 2023 | LOW | Builds a nodule-based risk model on an NLST subset (n=1138) with only internal evaluation described, without external screening-cohort validation. |
| 32784681 | Wan YL 2020 | LOW | Evaluates commercial AI on 50 surgically resected patients with nodules, a diagnostic rather than screening-cohort validation. |
| 32864596 | Huang P 2019 | HIGH | Trains a DL model on NLST and validates it double-blind on the PanCan lung cancer screening cohort with time-dependent AUC performance. |
| 34871936 | Cui X 2022 | HIGH | Evaluates a DL-CAD system as an alternative reader within a Chinese LDCT lung cancer screening program with detection sensitivity and false-positive results. |
| 31558626 | Ruparel M 2019 | LOW | Reports cardiovascular risk and coronary calcification in a lung screening cohort, with no AI system evaluated for lung cancer screening. |
| 40001094 | Huang F 2025 | MEDIUM | Reports a mobile LDCT screening program in China using an AI diagnostic system, but without reported AI performance metrics such as sensitivity or AUC. |
| 36634294 | Mikhael PG 2023 | HIGH | Sybil is validated on NLST held-out data and external screening datasets (MGH, CGMH) with AUC results for future lung cancer risk. |
| 37489991 | Xu K 2023 | LOW | Assesses AI-derived body composition for risk prediction in NLST rather than validating an AI system for lung cancer detection or screening performance. |
| 31675322 | Ozdemir O 2020 | LOW | Develops and validates a 3D deep learning system on public LUNA16 and Kaggle datasets rather than scans from an actual screening program or trial. |
| 42608820 | Mozafary M 2026 | LOW | A review of integrating cardiovascular (CAC) assessment into lung screening, with no AI system evaluated on a screening cohort with performance results. |
| 31747155 | Morozov SP 2019 | MEDIUM | Describes Moscow's real lung screening program and reports neural networks used for quality control of screening results, though performance metrics for the AI are not given. |
| 41872344 | Jiang B 2026 | HIGH | Evaluates a commercial AI software as the first reader on 1,252 baseline scans from the UK Lung Screening Trial, with workload results. |
| 39278738 | Ledda RE 2025 | LOW | Uses an AI body composition tool on ultra-LDCT from a screening population, but addresses fat and muscle associations rather than lung cancer detection performance. |
| 39709114 | Gabe R 2025 | LOW | Prospectively evaluates eligibility criteria in the Yorkshire screening trial without assessing any AI system. |

6. **Synthesize from the HIGH papers only.** 12 abstracts,
   10,191 input tokens, 2,430 output tokens.
   Prompt structure: who validated what, what recurs, gaps; cite every claim.
   Note: `max_tokens` must be at least 4096. Haiku 5.5 emits a thinking block
   before the text even at low effort, and 1,500 was consumed entirely by it,
   returning an empty answer.

7. **Verify citations.** Regex every `[PMID n]` in the review and check it is in
   the HIGH set. Cited: 12, not in context: 0.

## The review

**Who has validated AI in real screening cohorts**

- **Ledda et al. (2026)**: 4,104 baseline LDCTs from the BioMILD trial, retrospectively read by a radiologist and by AI. Against 2-year lung cancer diagnosis, AI had 91.2% sensitivity and 75.7% specificity, versus 89.7% and 90.0% for humans. AI's PPV was 6.0%, versus 13.1% for humans. AI was estimated to cut reading workload by 74.7% [PMID 41308576].
- **Peeters et al. (2026, LUNA25)**: External test on 156 malignant and 312 benign indeterminate nodules from three European screening trials. In an observer study of 300 nodules, the AI system's AUC was 0.78 versus 0.70 for 65 radiologists. At matched specificity it classified 12% more malignant nodules correctly [PMID 42340186].
- **Jiang et al. (2026, UKLS volume doubling time)**: 710 UK Lung Screening participants with 939 follow-up scans. AI's negative misclassification rate at 3 months was 9.1% (1/11), versus 18.2–63.6% for human readers against the expert panel. AI also flagged rapid growth in all 4 histologically confirmed cancers the panel called positive at 3 months [PMID 41319449].
- **Walstra et al. (2025, 4-IN-THE-LUNG-RUN)**: 3,678 baseline LDCTs. AI missed 0.8% of cases with nodules >100 mm³ (31 cases), versus 11.1% for radiologists. Radiologists as first reader would have missed 11.8% of referrals, versus 2.9% with AI [PMID 39754864].
- **Antonissen et al. (2025)**: Deep learning tested externally on 4,146 participants from three European screening trials. AUC was 0.96 for cancers diagnosed within 2 years, versus 0.94 for PanCan. In indeterminate nodules, it reduced false positives by 39.4% relative to PanCan at 100% sensitivity for 1-year cancers [PMID 40956165].
- **Huang et al. (2019)**: Trained on NLST (25,097 participants) and validated on 2,294 PanCan participants. The 1-year time-dependent AUC was 0.968, and the high-score group (10% of the sample) captured 94% of 1-year cancers [PMID 32864596].
- **Lancaster et al. (2022)**: 283 Moscow screening participants with solid nodules. AI's negative misclassification (8) was lower than that of four of five radiologists (21–35), and workload could fall by up to 86.7% [PMID 35123156].
- **Cui et al. (2022)**: 360 individuals from a Chinese LDCT screening program. Deep learning detection had 90.1% sensitivity at 1.0 false positive per scan, versus 76.0% for double reading [PMID 34871936].
- **Ji et al. (2025)**: Real-world cohort of 259,121 non-high-risk individuals at West China Hospital. LDCT with AI detected more cancers than conventional LDCT (0.3% vs 0.2%), with 5-year overall survival of 95.4% vs 81.3%. This was a retrospective comparison [PMID 40213646].
- **Mikhael et al. (2023, Sybil)**: Trained on NLST and externally validated on MGH and CGMH. 1-year AUCs were 0.86–0.94. The abstract does not specify whether the MGH and CGMH sets were screening cohorts [PMID 36634294].

**What recurs**

Several studies use trial-derived or screening-program data: NLST, UKLS, BioMILD, 4ITLR, and the European trials pooled in LUNA25 and the Antonissen study [PMID 41308576; PMID 42340186; PMID 41319449; PMID 39754864; PMID 40956165]. Most compare AI against radiologist reads or expert consensus, and several use negative misclassification or referral rates as the main endpoint [PMID 39754864; PMID 35123156]. AI tasks cluster into first-reader triage, nodule malignancy risk, growth assessment, and longitudinal matching [PMID 42668303; PMID 41872344].

**Gaps**

- No study in the context reports a prospective, live-workflow deployment. Ledda's design is retrospective [PMID 41308576], and the matching study explicitly calls for prospective validation in diverse populations [PMID 42668303].
- Few studies compare AI against human readers on hard-outcome endpoints such as mortality or stage shift. The Ji survival result is a non-randomized comparison [PMID 40213646].
- Specificity and PPV remain weak points. AI's specificity was lower than human reading in Ledda's cohort [PMID 41308576].
- Lancaster's cohort was small and from a single site [PMID 35123156].

## Comparison with an independent agent

A fresh Fable 5.1 agent was given only the question, the corpus, the keys,
and the instruction not to read this pipeline. Its review and full log are in
`reference/independent_review.md`.

| | This pipeline (Haiku) | Independent agent |
|---|---|---|
| Retrieval | three top-15 searches, union of 36 | screened all 181 abstracts, one Haiku call each |
| Selection | Haiku grades, keep HIGH | Haiku include/partial/exclude, then the agent read 26 abstracts itself |
| Writing | Haiku synthesizes | the agent wrote the text itself |
| Verification | regex: cited PMID is in context | one Haiku call per claim against its own abstract, then a number check |
| Papers cited | 12 | 25 |
| Overlap | all 12 of ours are in its 25 | |
| Haiku calls | 6 | 218 |
| Cost | 0.6 cents | 5.9 cents |

What the top-k pipeline missed: 11 of the 13 extra papers never entered the
36-candidate pool at all. They include the Korean randomised trial of AI-assisted
reading (Hwang 2026), three Sybil external validations in Asian cohorts, a
three-vendor comparison (Khurelsukh 2026), and Ardila 2019. Retrieval, not
grading, was the bottleneck. The other two were in the pool and graded MEDIUM
and LOW.

What the full-screen approach costs: ten times the money and about ten minutes
of wall time, both still trivial at this corpus size. It stops scaling at a few
thousand abstracts, where top-k retrieval becomes necessary again.

Verification quality: the agent's claim-by-claim check caught three over-claims
in its own draft before publication. Our regex check only confirms a PMID
exists in the context, not that the sentence is true. That is the single
biggest gap in the pipeline, and the cheapest to close.
