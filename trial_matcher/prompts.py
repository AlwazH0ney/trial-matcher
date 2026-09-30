"""
The criterion-screening prompt. This is the ONLY place prompt text lives.

Design notes (see README, Phase 2):
  - The model judges whether a criterion's statement is TRUE for the patient,
    not whether the patient is "eligible". It is never told whether the
    criterion is an inclusion or exclusion rule; match.py applies polarity.
    This avoids the double-negative errors models make on exclusion criteria.
  - The treatment history is declared complete, so "no prior KRAS G12C
    inhibitor" resolves to false rather than unknown.
  - Confidence on a null answer means "how sure the information is absent",
    so obvious gaps (organ function, consent) do not trigger API re-queries.
  - The one-shot example uses a fictional EGFR patient so it cannot leak
    answers for the KRAS G12C evaluation case.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Optional

from trial_matcher.patient import PatientProfile

TEMPERATURE = 0.0

SYSTEM_PROMPT = """\
You are screening one patient against ONE clinical-trial eligibility criterion.

Decide whether the criterion's statement is TRUE for this patient:
  "eligible": true   -> the patient meets the statement as written
  "eligible": false  -> the patient does not meet it
  "eligible": null   -> the profile lacks the information needed to decide
Judge the statement literally, even when it describes something undesirable
(e.g. "Active brain metastases" is true if the patient has them). Do not
consider whether it is an inclusion or exclusion rule.

Evidence rules:
1. Use only the patient profile. Do not assume lab values, organ function,
   consent, contraception, life expectancy, or anything not listed; those are null.
2. The treatment history is the COMPLETE list of prior systemic therapies.
   A drug or drug class absent from it was not received.
3. Apply standard oncology drug-class knowledge (e.g. pembrolizumab is an
   anti-PD-1 antibody; carboplatin is a platinum agent).
4. Mutations list every reported variant. A variant not listed is not present.

Confidence (0.0-1.0) is how sure you are of your answer:
  0.9-1.0  the profile states the fact directly
  0.6-0.8  follows from the profile with one reasonable inference
  <0.6     the criterion is ambiguous or only partly addressed
For null answers, confidence is how sure you are the information is truly absent.

Reply with ONLY this JSON object:
{"eligible": true|false|null, "confidence": <0.0-1.0>,
 "reasoning": "<one sentence>",
 "patient_feature": "<the profile field and value you relied on, or 'none'>"}"""

EXAMPLE_PATIENT_SUMMARY = """\
PATIENT PROFILE (as of 2026-01-15)
Diagnosis: Lung adenocarcinoma, stage IV
Age 58, female. ECOG 0. Smoking: never. Brain metastases: yes (treated, stable).
Mutations: EGFR p.L858R; TP53 p.R273H
TMB 2.1 mut/Mb; PD-L1 TPS 5%; MSI: MSS
Treatment history (complete):
  Line 1: osimertinib, 2024-03-01 to 2025-11-20, best response PR, stopped for progression"""

EXAMPLE_CRITERION = "Prior treatment with an EGFR tyrosine kinase inhibitor."

EXAMPLE_RESPONSE = (
    '{"eligible": true, "confidence": 0.95, "reasoning": "Osimertinib, a '
    'third-generation EGFR TKI, was given as first-line therapy.", '
    '"patient_feature": "treatment_history line 1: osimertinib"}'
)

# Changes whenever any fixed prompt text changes, so llm_log.jsonl rows can be
# grouped by prompt revision when measuring drift.
PROMPT_VERSION = hashlib.sha256(
    "\x1f".join([SYSTEM_PROMPT, EXAMPLE_PATIENT_SUMMARY, EXAMPLE_CRITERION,
                 EXAMPLE_RESPONSE]).encode("utf-8")
).hexdigest()[:12]


def _fmt(value: object, suffix: str = "", missing: str = "not recorded") -> str:
    """Render an optional profile value so absence is explicit, never blank."""
    if value is None:
        return missing
    if isinstance(value, float):
        return f"{value:g}{suffix}"   # 15.0 -> "15", 4.6 -> "4.6"
    return f"{value}{suffix}"


def summarize_patient(patient: PatientProfile, as_of: Optional[date] = None) -> str:
    """Render the compact patient block the model reasons over.

    Every field the matcher might need is spelled out, including explicit
    "not recorded" markers, so the model can tell missing data from negatives.
    """
    as_of = as_of or date.today()
    histology = f" (histology: {patient.histology})" if patient.histology else ""
    if patient.brain_metastases is None:
        brain = "not recorded"
    else:
        brain = "yes" if patient.brain_metastases else "no"

    muts = "; ".join(
        f"{m.gene} {m.protein_change}" + (" (hotspot)" if m.is_hotspot else "")
        for m in patient.mutations
    ) or "none reported"

    lines = [
        f"PATIENT PROFILE (as of {as_of.isoformat()})",
        f"Diagnosis: {patient.diagnosis}, stage {patient.stage}{histology}",
        f"Age {patient.age_years}, {patient.sex.lower()}. ECOG {patient.ecog}. "
        f"Smoking: {_fmt(patient.smoking_status)}. Brain metastases: {brain}. "
        f"Prior malignancy: {_fmt(patient.prior_malignancy)}.",
        f"Mutations: {muts}",
        f"TMB {_fmt(patient.tmb, ' mut/Mb')}; PD-L1 TPS {_fmt(patient.pd_l1_tps, '%')}; "
        f"MSI: {_fmt(patient.msi_status)}",
        "Treatment history (complete):",
    ]
    if not patient.treatment_history:
        lines.append("  none (treatment-naive)")
    for t in sorted(patient.treatment_history, key=lambda e: e.line):
        span = f"{t.start_date} to {t.end_date or 'ongoing'}"
        resp = f", best response {t.best_response}" if t.best_response else ""
        stop = (f", stopped for {t.discontinuation_reason}"
                if t.discontinuation_reason else "")
        lines.append(f"  Line {t.line}: {t.drug}, {span}{resp}{stop}")
    lines.append(f"Next line of therapy: {patient.current_line()}")
    return "\n".join(lines)


def _user_turn(patient_summary: str, criterion_text: str) -> str:
    """Join profile and criterion in the exact layout the one-shot example uses."""
    return f"{patient_summary}\n\nCRITERION\n{criterion_text.strip()}"


def build_messages(patient_summary: str, criterion_text: str) -> list[dict[str, str]]:
    """Build the chat turns (one-shot example, then the real query).

    Returned without the system prompt because Ollama and the Anthropic API
    take it in different places; each backend attaches SYSTEM_PROMPT itself.
    """
    return [
        {"role": "user", "content": _user_turn(EXAMPLE_PATIENT_SUMMARY, EXAMPLE_CRITERION)},
        {"role": "assistant", "content": EXAMPLE_RESPONSE},
        {"role": "user", "content": _user_turn(patient_summary, criterion_text)},
    ]
