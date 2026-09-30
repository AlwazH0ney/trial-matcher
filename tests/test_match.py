"""Routing and aggregation tests with scripted LLM backends."""

from __future__ import annotations

import json

import pytest

from conftest import ScriptedBackend
from trial_matcher.llm import Completion, JsonlLogger, LLMBackend, TransientLLMError
from trial_matcher.match import (aggregate, evaluate_criterion, evaluate_trial,
                                 needs_api_review, rank_trials)

CRIT = {"kind": "inclusion", "idx": 0, "text": "ECOG performance status of 0 or 1.",
        "features": {"ecog_max": 1}}


# ---- Hybrid router ------------------------------------------------------------

def test_confidence_05_is_flagged_for_api_requery(patient):
    local = ScriptedBackend("local", default=(True, 0.5))
    api = ScriptedBackend("api", default=(False, 0.92))
    res = evaluate_criterion(patient, CRIT, local, api)
    assert needs_api_review(0.5)
    assert res["route"] == "api"
    assert len(api.calls) == 1
    # API result is used, and both are recorded for later routing analysis
    assert res["backend"] == "api"
    assert res["eligible"] is False and res["confidence"] == 0.92
    assert res["local"]["eligible"] is True and res["local"]["confidence"] == 0.5
    assert res["api"]["eligible"] is False


@pytest.mark.parametrize("conf", [0.0, 0.3, 0.34, 0.76, 0.9, 1.0])
def test_confident_local_results_are_kept(patient, conf):
    local = ScriptedBackend("local", default=(True, conf))
    api = ScriptedBackend("api", default=(False, 0.99))
    res = evaluate_criterion(patient, CRIT, local, api)
    assert res["route"] == "local" and res["backend"] == "local"
    assert api.calls == []
    assert res["api"] is None


@pytest.mark.parametrize("conf", [0.35, 0.75])
def test_band_edges_are_inclusive(conf):
    assert needs_api_review(conf)


def test_missing_api_backend_keeps_local_and_marks_route(patient):
    local = ScriptedBackend("local", default=(True, 0.6))
    res = evaluate_criterion(patient, CRIT, local, None)
    assert res["route"] == "api_unavailable"
    assert res["eligible"] is True and res["backend"] == "local"


# ---- Trial evaluation -----------------------------------------------------------

def _kras_inhibitor_rule(summary: str, criterion: str):
    """A mock model that 'knows' sotorasib/adagrasib are KRAS G12C inhibitors."""
    if "KRAS G12C" in criterion and "inhibitor" in criterion and criterion.lower().startswith("prior"):
        took = any(d in summary.lower() for d in ("sotorasib", "adagrasib"))
        return (took, 0.95)
    return None


def _kras_positive_rule(summary: str, criterion: str):
    if "KRAS" in criterion and "G12C" in criterion and "mutation" in criterion.lower():
        return (True, 0.97)
    return None


def test_prior_g12c_inhibitor_patient_is_excluded(patient_prior_g12c, trial_by_id):
    local = ScriptedBackend("local", rules=[_kras_inhibitor_rule, _kras_positive_rule],
                            default=(None, 0.9))
    res = evaluate_trial(patient_prior_g12c, trial_by_id("NCT03600883"), local, None)
    assert res["status"] == "ineligible"
    blocking = res["blocking"]
    assert len(blocking) == 1
    assert blocking[0]["kind"] == "exclusion"
    assert "KRAS G12C inhibitor" in blocking[0]["text"]
    crit = next(c for c in res["criteria"] if c["kind"] == "exclusion" and c["idx"] == 4)
    assert crit["eligible"] is True        # the exclusion statement is true for this patient
    assert crit["passes"] is False         # ... so the patient does not clear it
    assert crit["evidence"]["criterion_text"] == crit["text"]
    assert crit["evidence"]["patient_feature"]


def test_inhibitor_naive_patient_is_not_excluded(patient, trial_by_id):
    local = ScriptedBackend("local", rules=[_kras_inhibitor_rule, _kras_positive_rule],
                            default=(None, 0.9))
    res = evaluate_trial(patient, trial_by_id("NCT03600883"), local, None)
    assert res["status"] == "unresolved"   # unknowns (organ function etc.) remain, no blockers
    assert res["blocking"] == []


def test_biomarker_criteria_weigh_double():
    crits = [
        {"kind": "inclusion", "idx": 0, "text": "KRAS G12C", "weight": 2.0,
         "passes": True, "confidence": 1.0, "pass_confidence": 1.0, "reasoning": "", "evidence": {}},
        {"kind": "inclusion", "idx": 1, "text": "Adequate organ function", "weight": 1.0,
         "passes": None, "confidence": 0.9, "pass_confidence": 0.5, "reasoning": "", "evidence": {}},
    ]
    # score is rounded to 4 dp in the output
    assert aggregate(crits)["score"] == pytest.approx((2 * 1.0 + 1 * 0.5) / 3, abs=1e-4)


def test_all_confident_passes_make_trial_eligible(patient, trial_by_id):
    t = trial_by_id("NCT04303780")
    rule = lambda s, c: (False, 0.9) if any(  # noqa: E731
        c == x["text"] for x in t["criteria"] if x["kind"] == "exclusion") else (True, 0.9)
    res = evaluate_trial(patient, t, ScriptedBackend("local", rules=[rule]), None)
    assert res["status"] == "eligible"
    assert res["score"] == pytest.approx(0.9)


def test_ranking_orders_by_status_then_score():
    ts = [{"nct_id": "a", "status": "ineligible", "score": 0.99},
          {"nct_id": "b", "status": "unresolved", "score": 0.6},
          {"nct_id": "c", "status": "eligible", "score": 0.7},
          {"nct_id": "d", "status": "unresolved", "score": 0.8}]
    assert [t["nct_id"] for t in rank_trials(ts)] == ["c", "d", "b", "a"]


# ---- Logging + retry through the real LLMBackend pipeline -----------------------

class FlakyBackend(LLMBackend):
    """Fails transiently once, then returns fenced JSON, to exercise retry, repair and logging."""

    role, model = "local", "flaky"

    def __init__(self, **kw):
        super().__init__(base_delay=0.0, **kw)
        self.n = 0

    def _complete(self, system, messages):
        self.n += 1
        if self.n == 1:
            raise TransientLLMError("simulated 503")
        return Completion('```json\n{"eligible": true, "confidence": 0.9, "reasoning": "r", '
                          '"patient_feature": "ECOG 1",}\n```', input_tokens=700, output_tokens=40)


def test_every_call_is_logged_with_retry_and_repair(patient, tmp_path):
    log = tmp_path / "llm_log.jsonl"
    backend = FlakyBackend(logger=JsonlLogger(log))
    res = evaluate_criterion(patient, CRIT, backend, None, nct_id="NCT00000001")
    assert res["eligible"] is True and res["local"]["attempts"] == 2
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["nct_id"] == "NCT00000001" and row["criterion_idx"] == 0
    assert row["attempts"] == 2 and row["repaired"] is True
    assert row["input_tokens"] == 700 and row["prompt_version"]
    assert {"platform", "gpu"} <= set(row["runtime"])
