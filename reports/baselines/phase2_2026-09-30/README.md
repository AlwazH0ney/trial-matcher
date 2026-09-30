# Phase 2 baseline — 2026-09-30

The first end-to-end run of the Phase 2 matcher with a real local LLM. Phase 3 (calibration,
verification, abstention) is measured against this run and must not regress it.

## Files

| File | Contents |
|---|---|
| `matches.md` | Ranked output with per-criterion verdicts, confidence, route and cited evidence |
| `run_summary.json` | Speed, routing, model identity, corpus size, ranking, known failures, and all 43 per-criterion verdicts (the regression gate) |
| `llm_log_sample.jsonl` | First 10 of 43 LLM call records. The full log is not committed; regenerate it by re-running the notebook |

## Run conditions

| | |
|---|---|
| Patient | `TCGA-05-4418-derived`: stage IV lung adenocarcinoma, KRAS G12C (+ STK11, KEAP1, RBM10), ECOG 1, progressed on first-line carboplatin + pemetrexed + pembrolizumab |
| Corpus | `data/processed/trials_parsed.json`: 3 trials (sotorasib, adagrasib, osimertinib), 43 criteria, all passed to the LLM |
| Local model | phi4, 14.66B, Q4_K_M, Ollama 0.35.0, blob `sha256-fd7b6731…5df20`, temperature 0.0, seed 0, JSON-schema-constrained output |
| API model | Disabled (no `ANTHROPIC_API_KEY`); criteria in the [0.35, 0.75] band kept the local verdict (`route = api_unavailable`) |
| Hardware | Google Colab, Tesla T4, all 41/41 layers on GPU |
| Code | commit `01be495`; prompt version `3fe32ef28a11`; reference date 2026-09-30 |
| Calibration / verification / abstention | None (Phase 3) |

## Results

| Rank | Trial | Status | Score | Pass / Fail / Unknown |
|---:|---|---|---:|---|
| 1 | NCT04303780 adagrasib (KRYSTAL-1) | unresolved | 0.863 | 9 / 0 / 4 |
| 2 | NCT03600883 sotorasib (CodeBreaK 100) | unresolved | 0.732 | 7 / 0 / 10 |
| 3 | NCT03778229 osimertinib | ineligible | 0.588 | 6 / 2 / 5 |

- **Speed:** 3.28 s mean, 3.04 s median, 4.60 s p95 per criterion; 43 criteria in 141 s (18.3/min).
- **Routing:** 3/43 criteria (7.0%) in the re-query band; 0 sent to the API.
- **Output quality:** 0 parse errors, 0 repaired outputs; 19/43 criteria unknown.
- **Status:** no trial is "eligible" because unrecorded facts (organ function, consent, infections,
  cardiac history) are correctly left unknown.

The top-3 acceptance criterion (sotorasib or adagrasib in the top 3) is met, but with a 3-trial
corpus it is not discriminative. Retrieval on the 203-trial corpus ranks adagrasib 1st and
sotorasib 5th.

## Known failures (Phase 3 targets)

1. **Overconfidence on absent information.** NCT03778229 exclusion #3, "Concurrent malignancy
   requiring active therapy": answered *not met, confidence 1.00*, though the profile says
   "Prior malignancy: not recorded". It should be null. Target: calibration + prompt fix.
2. **Failure to compose facts.** NCT03778229 inclusion #3, "Radiographic progression on prior
   EGFR TKI therapy": answered *unknown, 0.90*, though the patient has no EGFR mutation and no
   prior EGFR TKI, so it is provably not met. Target: verification.

Neither failure changed the ranking: osimertinib is already excluded by inclusions #1 and #2.

## Not measurable yet

ECE, coverage-risk and abstention utility need ground-truth labels. They are computed once
`data/annotations/annotations_v1.jsonl` is filled in (Phase 3).
