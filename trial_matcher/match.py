"""
Per-criterion judgments, hybrid local/API routing, and trial-level aggregation.

Terminology used in outputs:
  eligible       the LLM verdict: is the criterion's statement TRUE for the
                 patient (true / false / null = unknown). Same meaning for
                 inclusion and exclusion criteria.
  passes         whether the patient clears the criterion: eligible for an
                 inclusion, NOT eligible (the exclusion does not apply) for an
                 exclusion. null when the verdict is unknown.
  pass_confidence  probability-like support for "patient clears this criterion":
                 confidence if passes, 1 - confidence if not, 0.5 if unknown.

Trial status:
  ineligible   at least one criterion confidently (> 0.5) not passed
  eligible     every criterion confidently (> 0.5) passed
  unresolved   nothing confidently fails, but some criteria are unknown or weak
"""

from __future__ import annotations

from typing import Any, Optional

from trial_matcher.llm import JudgmentResult, LLMBackend
from trial_matcher.patient import PatientProfile
from trial_matcher.prompts import summarize_patient

ROUTE_LOW = 0.35
ROUTE_HIGH = 0.75
DECISION_THRESHOLD = 0.5
BIOMARKER_WEIGHT = 2.0
DEFAULT_WEIGHT = 1.0

STATUS_ORDER = {"eligible": 0, "unresolved": 1, "ineligible": 2}


def needs_api_review(confidence: float) -> bool:
    """True when a local judgment is in the uncertain band [0.35, 0.75] and should be re-asked."""
    return ROUTE_LOW <= confidence <= ROUTE_HIGH


def is_biomarker_criterion(criterion: dict[str, Any]) -> bool:
    """Biomarker criteria are the ones Phase 1 tagged with a gene or variant."""
    feats = criterion.get("features") or {}
    return bool(feats.get("biomarkers") or feats.get("mutations"))


def criterion_weight(criterion: dict[str, Any]) -> float:
    """Biomarker criteria count double: they are the most discriminative for targeted-therapy trials."""
    return BIOMARKER_WEIGHT if is_biomarker_criterion(criterion) else DEFAULT_WEIGHT


def _passes(kind: str, eligible: Optional[bool]) -> Optional[bool]:
    """Apply inclusion/exclusion polarity to the literal verdict."""
    if eligible is None:
        return None
    return eligible if kind == "inclusion" else not eligible


def _pass_confidence(passes: Optional[bool], confidence: float) -> float:
    """Support for 'patient clears this criterion', so confident failures lower the score."""
    if passes is None:
        return 0.5
    return confidence if passes else 1.0 - confidence


def evaluate_criterion(patient: PatientProfile, criterion: dict[str, Any],
                       backend_local: LLMBackend, backend_api: Optional[LLMBackend],
                       *, patient_summary: Optional[str] = None,
                       nct_id: Optional[str] = None) -> dict[str, Any]:
    """Judge one criterion with the local model, escalating uncertain cases to the API.

    Both judgments are kept in the output so routing impact (agreement rate,
    flips, cost) can be measured later. With no API backend, uncertain cases
    keep the local verdict and are marked route='api_unavailable'.
    """
    summary = patient_summary or summarize_patient(patient)
    kind = criterion.get("kind", "inclusion")
    meta = {"patient_id": patient.patient_id, "nct_id": nct_id,
            "criterion_kind": kind, "criterion_idx": criterion.get("idx")}

    local: JudgmentResult = backend_local.judge(summary, criterion["text"], meta)
    api: Optional[JudgmentResult] = None
    if needs_api_review(local.confidence):
        if backend_api is not None:
            api = backend_api.judge(summary, criterion["text"], meta)
            route = "api"
        else:
            route = "api_unavailable"
    else:
        route = "local"
    final = api or local

    passes = _passes(kind, final.eligible)
    return {
        "kind": kind,
        "idx": criterion.get("idx"),
        "text": criterion["text"],
        "is_biomarker": is_biomarker_criterion(criterion),
        "weight": criterion_weight(criterion),
        "eligible": final.eligible,
        "confidence": final.confidence,
        "reasoning": final.reasoning,
        "evidence": {"patient_feature": final.patient_feature,
                     "criterion_text": criterion["text"]},
        "passes": passes,
        "pass_confidence": _pass_confidence(passes, final.confidence),
        "backend": final.backend,
        "route": route,
        "local": local.to_dict(),
        "api": api.to_dict() if api else None,
    }


def aggregate(criteria: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn per-criterion results into a trial status and weighted score."""
    if not criteria:
        return {"status": "unresolved", "score": 0.0, "n_pass": 0, "n_fail": 0,
                "n_unknown": 0, "blocking": [], "note": "no criteria parsed"}

    def confident(c: dict[str, Any]) -> bool:
        return c["confidence"] > DECISION_THRESHOLD

    blocking = [c for c in criteria if c["passes"] is False and confident(c)]
    all_pass = all(c["passes"] is True and confident(c) for c in criteria)
    status = "ineligible" if blocking else "eligible" if all_pass else "unresolved"

    total_w = sum(c["weight"] for c in criteria)
    score = sum(c["weight"] * c["pass_confidence"] for c in criteria) / total_w
    return {
        "status": status,
        "score": round(score, 4),
        "n_pass": sum(1 for c in criteria if c["passes"] is True),
        "n_fail": sum(1 for c in criteria if c["passes"] is False),
        "n_unknown": sum(1 for c in criteria if c["passes"] is None),
        "blocking": [{"kind": c["kind"], "idx": c["idx"], "text": c["text"],
                      "reasoning": c["reasoning"], "evidence": c["evidence"]}
                     for c in blocking],
    }


def evaluate_trial(patient: PatientProfile, trial: dict[str, Any],
                   backend_local: LLMBackend, backend_api: Optional[LLMBackend],
                   *, patient_summary: Optional[str] = None) -> dict[str, Any]:
    """Evaluate every criterion of one trial and aggregate to a status and score.

    Every criterion is evaluated even after a blocker is found: the complete
    per-criterion record is what later calibration and routing analysis need.
    """
    summary = patient_summary or summarize_patient(patient)
    results = [evaluate_criterion(patient, c, backend_local, backend_api,
                                  patient_summary=summary, nct_id=trial.get("nct_id"))
               for c in trial.get("criteria") or []]
    return {
        "nct_id": trial.get("nct_id"),
        "title": trial.get("title"),
        "phase": trial.get("phase"),
        "interventions": [i.get("name") for i in trial.get("interventions") or []],
        "retrieval_score": trial.get("retrieval_score"),
        **aggregate(results),
        "n_criteria": len(results),
        "n_routed_api": sum(1 for r in results if r["route"] == "api"),
        "n_api_unavailable": sum(1 for r in results if r["route"] == "api_unavailable"),
        "criteria": results,
    }


def rank_trials(evaluated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order trials: eligible, then unresolved, then ineligible; by score, then retrieval score."""
    return sorted(evaluated, key=lambda t: (
        STATUS_ORDER.get(t.get("status"), 9),
        -(t.get("score") or 0.0),
        -(t.get("retrieval_score") or 0.0),
    ))
