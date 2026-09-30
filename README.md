# LUAD Trial Matcher

Uncertainty-aware, evidence-grounded clinical trial matching for lung adenocarcinoma.

**Status**: Phase 2 of 5. Retrieval + per-criterion LLM matcher with hybrid local/API routing.

## What this does (planned)

Takes a patient profile (mutations from the LUAD variant pipeline + clinical baseline + treatment history) and a corpus of recruiting clinical trials, and produces a ranked list of trials the patient is likely eligible for, with:

- Per-criterion calibrated confidence, not just a trial-level score
- Selective abstention when evidence is insufficient (a "refer to human" flag)
- Versioned evidence chain: every eligibility decision cites the specific criterion clause, patient feature, and any external source

Differentiates from [TrialMatchAI](https://arxiv.org/abs/2505.08508) and [TrialGPT](https://www.nature.com/articles/s41467-024-53081-z) on those three attributes.

## Layout

```
trial_matcher/
├── fetch.py       # ClinicalTrials.gov API v2 ingestion
├── parse.py       # eligibility text -> structured criteria list
├── patient.py     # patient profile schema + example
├── retrieve.py    # BM25 + MiniLM hybrid retrieval, RRF (k=60), cached indexes
├── prompts.py     # THE criterion prompt (only place prompt text lives)
├── llm.py         # OllamaBackend (phi4), ClaudeBackend (Haiku 4.5), parse/repair, retries, JSONL log
├── match.py       # per-criterion routing + trial aggregation + ranking
└── cli.py         # luad-match retrieve | evaluate | rank | example-patient
data/
├── raw/           # raw JSON from ClinicalTrials.gov
└── processed/     # parsed trials, patient profiles, indexes, llm_log.jsonl
notebooks/
└── run_evaluation.ipynb   # Colab T4: Ollama + phi4 end-to-end run
reports/           # matches.md
tests/             # pytest suite; LLMs mocked
```

## Phase 1 (done)

- Project scaffold
- ClinicalTrials.gov API v2 fetcher (`fetch.py`) — filters recruiting LUAD trials
- Eligibility parser (`parse.py`) — splits inclusion/exclusion criteria, extracts biomarkers + mutations + ECOG + age with regex
- Patient profile schema (`patient.py`) — mutations, clinical baseline, treatment timeline
- Example patient derived from TCGA-05-4418 (KRAS G12C, progressed on first-line chemo + pembrolizumab)

## Phase 2 (done)

- Retrieval: BM25 (titles + conditions + interventions) + `all-MiniLM-L6-v2` cosine, fused with RRF k=60
- Criterion-level LLM matcher: one criterion per call, one-shot JSON prompt, temperature 0.0
- Hybrid routing: local phi4 first; confidence in [0.35, 0.75] is re-asked via Claude Haiku 4.5
  (`claude-haiku-4-5-20251001`) when `ANTHROPIC_API_KEY` is set. Both judgments are kept.
- Basic evidence chain: each decision cites the criterion text and the patient feature used
- Every LLM call logged to `data/processed/llm_log.jsonl` (tokens, latency, prompt version, runtime/GPU)

### Local model: phi4 (14B) on a Colab T4 GPU

The local backend is **phi4 (14B)** served by Ollama. On a laptop CPU (i7-1250U, no discrete GPU)
it takes roughly 30-40 s per criterion, so a 50-trial run (~850 criteria) would take ~9 hours.
The supported way to run evaluation is [`notebooks/run_evaluation.ipynb`](notebooks/run_evaluation.ipynb)
on Google Colab with a T4 GPU. `OllamaBackend` honours `OLLAMA_HOST`, so the same code path runs
locally (`localhost:11434`) and on Colab (`127.0.0.1:11434`). Each `llm_log.jsonl` row records
`model` and `runtime` (`platform`, `gpu`), so latency and cost numbers can be traced to where
inference ran. If Ollama is unreachable or phi4 is not pulled, `evaluate` stops with an actionable
error; it never silently falls back to the API.

### How a criterion is judged

The model is asked whether the criterion's *statement* is true for the patient
(`eligible: true | false | null`), without being told whether it is an inclusion or exclusion rule;
`match.py` applies polarity (`passes`). A trial is:

- **ineligible** if any criterion confidently (> 0.5) fails
- **eligible** if every criterion confidently (> 0.5) passes
- **unresolved** otherwise (typically criteria the profile cannot answer, e.g. organ function)

Score = mean of per-criterion *pass confidence* (confidence if passed, 1 - confidence if failed,
0.5 if unknown), with biomarker criteria weighted 2.0. Ranking: eligible, unresolved, ineligible;
then score.

## Phase 3 (after that)

- Calibration + abstention: conformal prediction over LLM confidence, selective refusal
- Longitudinal state module: prior therapy influences criterion evaluation
- Verification module: cross-check LLM output against retrieved evidence

## Phase 4

- Baselines: reproduce TrialMatchAI, run TrialGPT via NIH API where possible
- Metrics: nDCG@10, ECE, coverage-risk curves, evidence traceability rate
- Benchmark on TREC-CDS 2023 + hand-annotated LUAD subset

## Phase 5

- Package as CLI, tests, GitHub, paper draft

## Running Phase 1 locally

```bash
# fetch real trials (sandbox network is restricted; run on your machine)
python -m trial_matcher.fetch --condition "Lung Adenocarcinoma" --limit 200

# parse
python -m trial_matcher.parse --in data/raw/trials.json --out data/processed/trials.json

# example patient
python -m trial_matcher.patient
```

## Running Phase 2

```bash
pip install -e ".[dev]"
pytest tests/                          # LLMs mocked; downloads MiniLM once

luad-match example-patient --out data/processed/patient_example.json
luad-match retrieve --patient data/processed/patient_example.json     --trials data/processed/trials_parsed.json --top-k 50 --out data/processed/candidates.json
luad-match evaluate --patient data/processed/patient_example.json     --candidates data/processed/candidates.json --out data/processed/evaluated.json   # needs Ollama + phi4
luad-match rank --evaluated data/processed/evaluated.json --out reports/matches.md
```

`evaluate` checkpoints after each trial; `--resume` continues an interrupted run.
Set `ANTHROPIC_API_KEY` to enable API re-queries, or pass `--no-api`.

## Five open decisions (unresolved)

1. Longitudinal data source: TCGA only, synthesized trajectories, or MSK-IMPACT
2. Own oncology benchmark: yes and hand-annotate 150-200 judgments, or TREC only
3. LLM strategy: API, local, or hybrid routing
4. Calibration method: temperature scaling, isotonic, or conformal
5. Baselines: reproduce TrialMatchAI, cite, or both

## Not for clinical use
