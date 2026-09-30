"""Shared fixtures: example patient, trial corpora, and a scripted LLM backend (no Ollama/API needed)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Optional

import pytest

from trial_matcher.llm import JudgmentResult, LLMBackend
from trial_matcher.patient import PatientProfile, TreatmentEvent, example_patient_kras_g12c

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TRIALS = ROOT / "data" / "processed" / "trials_parsed.json"
RETRIEVAL_CORPUS = Path(__file__).parent / "fixtures" / "retrieval_corpus.json"

Rule = Callable[[str, str], Optional[tuple[Optional[bool], float]]]


class ScriptedBackend(LLMBackend):
    """Deterministic stand-in for an LLM: rules map (patient_summary, criterion) to a verdict.

    The first rule returning a (eligible, confidence) tuple wins; otherwise the
    default applies. Every call is recorded so tests can assert on routing.
    """

    def __init__(self, role: str, rules: list[Rule] | None = None,
                 default: tuple[Optional[bool], float] = (None, 0.9)) -> None:
        super().__init__()
        self.role = role
        self.model = f"scripted-{role}"
        self.rules = rules or []
        self.default = default
        self.calls: list[str] = []

    def judge(self, patient_summary: str, criterion_text: str,
              meta: dict[str, Any] | None = None) -> JudgmentResult:
        self.calls.append(criterion_text)
        verdict = next((v for r in self.rules if (v := r(patient_summary, criterion_text))),
                       self.default)
        eligible, conf = verdict
        return JudgmentResult(eligible=eligible, confidence=conf,
                              reasoning=f"scripted {self.role}", patient_feature="scripted",
                              backend=self.role, model=self.model, latency_s=0.01)


@pytest.fixture
def patient() -> PatientProfile:
    """The Phase 1 KRAS G12C example patient (post first-line chemo + pembrolizumab)."""
    return example_patient_kras_g12c()


@pytest.fixture
def patient_prior_g12c(patient: PatientProfile) -> PatientProfile:
    """Same patient, but with second-line sotorasib already given."""
    p = copy.deepcopy(patient)
    p.treatment_history.append(TreatmentEvent(
        line=2, drug="sotorasib", start_date="2026-07-01", end_date="2026-09-01",
        best_response="PD", discontinuation_reason="progression"))
    return p


@pytest.fixture
def sample_trials() -> list[dict[str, Any]]:
    """The 3 parsed Phase 1 sample trials (sotorasib, adagrasib, osimertinib)."""
    return json.loads(SAMPLE_TRIALS.read_text(encoding="utf-8"))


@pytest.fixture
def retrieval_corpus() -> list[dict[str, Any]]:
    """203 trials: 200 recruiting LUAD trials from ClinicalTrials.gov plus the 3 samples."""
    return json.loads(RETRIEVAL_CORPUS.read_text(encoding="utf-8"))


@pytest.fixture
def trial_by_id(sample_trials: list[dict[str, Any]]) -> Callable[[str], dict[str, Any]]:
    return lambda nct: next(t for t in sample_trials if t["nct_id"] == nct)
