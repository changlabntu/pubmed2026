# Module 1 v4: argue a research gap

This Streamlit app builds a mini literature review the way a paper's
introduction argues. The student states a claim about a gap, starts from
three papers, gathers evidence for and against, and has the model write the
argument as thesis, antithesis and synthesis with a verdict. Every cited
sentence is then checked against the abstract it cites. Embeddings come from
OpenAI `text-embedding-3-small`. Scoring, writing and fact-checking are done
by Claude Haiku 5.5, with Sonnet 5.5 available as an option. The homework is
the report the app writes.

Three question sets ship with the app, in `question_sets/`. They cover AI
for lung cancer screening on low-dose CT, AI for knee pain from MRI, and AI
prediction of total knee replacement. Each set carries the topic question,
the gap, what would refute it, a PubMed query and three PMIDs. Two pools of
abstracts are cached under `pools/`, named `lung_ct` and `knee_mri`. Any
question set can also fetch its own corpus from PubMed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # paste ANTHROPIC_API_KEY and OPENAI_API_KEY
streamlit run app.py
```

On Streamlit Community Cloud the main file is `app/app.py`. Put the same two
keys in the app's Secrets panel instead of `.env`.

## The four moves

**1. The claim.** The student writes the topic question, the gap, and the
kind of paper that would refute the gap. These three lines go into every
model call as context.

**2. Start from three papers.** The student picks a corpus, either a shipped
pool or their own PubMed query, and names an anchor paper and two companion
papers by PMID. The scoring prompt sits here in three editable boxes. One
button fetches the corpus, scores the three papers, and draws them with a
black outline on the map. The point is to see where the three papers fall
before any search.

**3. Gather evidence.** One button runs two searches, the six papers closest
to the question and the six closest to the anchor. The new papers are scored
in one call and join the three. Green panels argue for the gap and red panels
argue against it. The map recolours by stance. The student should notice
which search found the paper they would have chosen, and whether any red
panel deserves to be red.

**4. The argument.** The argument prompt and the verifier prompt sit here,
each as three editable boxes with a red edge. One button has the model write the thesis, the antithesis, the
synthesis and a verdict from the thesis and antithesis papers. Antithesis
papers go in with their full abstract. Every cited sentence is then checked
against the abstract it cites. The student should say whether the synthesis
narrowed or conceded the gap, and name one verification row they dispute.

A comment box sits under moves 2, 3 and 4. The sidebar holds the question-set
loader and the download button for the report.

## Teacher mode

The app has two modes. Students see Haiku only and a cost cap of 10 cents
that is enforced but not shown. Teachers also see the model switch with
Sonnet 5.5, the cost cap slider, and the spend for the session. Start
teacher mode with

```bash
streamlit run app.py -- --teacher
```

or set `TEACHER=1` in the environment. On Community Cloud put `TEACHER = "1"`
in the Secrets panel.

All three prompts are editable in both modes, each as three boxes headed
BACKGROUND, WHAT TO DO and OUTPUT FORMAT. The boxes are joined in order into
the prompt the model receives, and a Reset button restores the defaults. The
scoring prompt is the same full text in both modes, because it is the demo.
The argument and verifier prompts differ: teachers get the full text, and
students start from one line per box, with a red edge on those boxes to
mark them as the ones the homework asks them to change. A student who adds
a rule to WHAT TO DO sees the model's argument change on the next run.

## One vocabulary

For each paper the model returns two things. The first is whether the
paper's own work touches the question at all. The second is a score from 1
to 10, where 1 is the strongest evidence against the gap and 10 is the
strongest evidence for it. The bands of the scale are spelled out in the
prompt. The stance is then computed in code. A paper that does not touch the
question is neutral. A paper scored 5 or below is antithesis. A paper scored
above 5 is thesis. The cutoff is `STANCE_CUTOFF` in `lit_helper.py`.

The model also tests each refutation condition against the abstract and marks
it yes, no or not reported, with a quote. The gap relation is computed from
those checks, as counterexample, partial, no or unclear. A flag marks any
paper whose score and checks disagree, so that the reader trusts the quotes
over the number.

The verifier marks each claim as supported, not supported, or can't tell. The
verdict on the gap is one of gap stands, gap narrowed, gap closed, or can't
tell. In every place, "can't tell" means that the material does not decide
the question.

## The report

The report is written in the order a student would write the argument. It
begins with the claim and the three starting papers. It then lists the
evidence for, against and neutral, each with its score, reason and quote,
and describes how the corpus was searched. The argument follows with its
verdict, then the verification counts and table. An appendix holds the
prompts as used, every scored record, the model and the cost. The student's
comments sit under the moves they belong to.

## What happens when a prompt breaks

If the scoring prompt no longer produces a JSON array, the app shows the raw
reply with a yellow note. The note says when the reply was cut off at the
model's output limit. If the records carry neither a score nor a relevance
field, the argument uses every paper and says so. Editing the claim, the
PMIDs or a prompt clears that move and the moves after it. Comments survive.

## Adding a pool or a question set

```bash
python build_pool.py my_pool '<PubMed query>'
python pick_anchor.py my_pool '<question>' '<gap>' '<disproof>'
```

The first command fetches, filters and embeds a pool. The second scores the
ten papers closest to the question and suggests an anchor.

A question set is one JSON file in `question_sets/`. It has the fields
`name`, `question`, `gap`, `disproof`, `query` and `pmids`, with the anchor
first, and optionally `prompts`. Copy one of the shipped files and edit it.

## Files

| File | What it is |
|---|---|
| `app.py` | The Streamlit page. |
| `lit_helper.py` | The corpus, the two retrievals, scoring and stance, the argument, verification, and the stance panels. |
| `pubmed_ingest.py` | Fetching and parsing from PubMed, and citation counts from NIH iCite. |
| `build_pool.py`, `pick_anchor.py` | Instructor tools. |
| `pools/<name>/` | A cached pool: the query, the PMIDs, the fetch metadata, the abstracts, the vectors, and the anchor. |
| `question_sets/` | The shipped question sets. |
| `homework/homework.md` | What students submit. |
| `UI-MOCKUP.md` | The page layout as a text sketch. |
| `../tests/test_v4.py` | Headless tests with a faked model. |
| `../tests/live_run.py` | One run of the four moves against the real models, printed. |
| `../tests/scenarios.py` | Ten scenarios against the real models, with quality checks. |

## Cost

A full run on the lung CT set makes about 12 model calls and costs about 1.2
cents on Haiku. There are two scoring calls, one for the three starting
papers and one for up to 12 new papers, one argument call, and one
verification call for each cited sentence. The default cap is 10 cents.

## Two things to know

Haiku 5.5 emits a thinking block before its text even at low effort, and the
thinking counts against `max_tokens`. Scoring 17 papers with quotes and
condition checks produces around 8,000 output tokens, so the scoring call
allows 16,384.

The verifier checks one sentence against one abstract. A sentence that
summarises several papers can only be judged for this paper's share. Read
the issue column before trusting a "not supported".
