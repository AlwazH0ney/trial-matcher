"""
Patient profile: the schema the matcher consumes as input.

Two layers:
  - baseline: clinical facts true at diagnosis
  - timeline: treatment history (line, drug, response, dates)

The matcher evaluates each trial criterion against this profile.
"""

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional
import json


@dataclass
class Mutation:
    gene: str                   # HGNC symbol, e.g. "KRAS"
    protein_change: str         # HGVS short, e.g. "p.G12C"
    variant_class: str          # "Missense_Mutation", "Frame_Shift_Del", ...
    tumor_vaf: Optional[float] = None
    is_hotspot: bool = False
    evidence_score: Optional[int] = None    # 0-6 from Phase 5


@dataclass
class TreatmentEvent:
    line: int                   # 1 = first-line, 2 = second-line, ...
    drug: str                   # e.g. "carboplatin + pemetrexed"
    start_date: str             # ISO date
    end_date: Optional[str] = None
    best_response: Optional[str] = None       # "CR" | "PR" | "SD" | "PD" | "NE"
    discontinuation_reason: Optional[str] = None  # "progression" | "toxicity" | "completed"


@dataclass
class PatientProfile:
    patient_id: str
    diagnosis: str              # e.g. "Lung adenocarcinoma"
    stage: str                  # e.g. "IV"
    age_years: int
    sex: str                    # "MALE" | "FEMALE"
    ecog: int                   # 0-4
    smoking_status: Optional[str] = None      # "never" | "former" | "current"
    histology: Optional[str] = None
    brain_metastases: Optional[bool] = None
    prior_malignancy: Optional[str] = None

    mutations: list[Mutation] = field(default_factory=list)
    tmb: Optional[float] = None                # mutations/Mb
    pd_l1_tps: Optional[float] = None          # PD-L1 tumor proportion score (0-100)
    msi_status: Optional[str] = None           # "MSS" | "MSI-H" | "MSI-L"

    treatment_history: list[TreatmentEvent] = field(default_factory=list)

    def current_line(self) -> int:
        """Next line of therapy is (max prior line) + 1."""
        if not self.treatment_history:
            return 1
        return max(t.line for t in self.treatment_history) + 1

    def prior_drugs(self) -> list[str]:
        return [t.drug for t in self.treatment_history]

    def prior_response_to(self, drug_keyword: str) -> Optional[str]:
        """Find the best response on any prior therapy containing a drug keyword."""
        matches = [t for t in self.treatment_history
                   if drug_keyword.lower() in t.drug.lower()]
        return matches[-1].best_response if matches else None

    def has_mutation(self, gene: str, protein_change: str = None) -> bool:
        for m in self.mutations:
            if m.gene.upper() != gene.upper():
                continue
            if protein_change is None:
                return True
            if m.protein_change.upper().replace("P.", "") == \
               protein_change.upper().replace("P.", ""):
                return True
        return False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        muts = [Mutation(**m) for m in d.pop("mutations", [])]
        hist = [TreatmentEvent(**t) for t in d.pop("treatment_history", [])]
        return cls(**d, mutations=muts, treatment_history=hist)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# ---- Example patient derived from Phase 5 output ---------------------------

def example_patient_kras_g12c():
    """
    Modeled on TCGA-05-4418 from the LUAD pipeline:
      KRAS G12C + STK11 loss + KEAP1 loss + RBM10 frameshift
      TMB 4.6 mut/Mb (low)
      Now progressed on first-line chemotherapy.
    """
    return PatientProfile(
        patient_id="TCGA-05-4418-derived",
        diagnosis="Lung adenocarcinoma",
        stage="IV",
        age_years=63,
        sex="MALE",
        ecog=1,
        smoking_status="former",
        histology="adenocarcinoma",
        brain_metastases=False,
        mutations=[
            Mutation("KRAS", "p.G12C", "Missense_Mutation",
                     tumor_vaf=0.41, is_hotspot=True, evidence_score=5),
            Mutation("STK11", "p.G56V", "Missense_Mutation",
                     tumor_vaf=0.39, is_hotspot=False, evidence_score=3),
            Mutation("KEAP1", "p.F280Y", "Missense_Mutation",
                     tumor_vaf=0.43, is_hotspot=False, evidence_score=3),
            Mutation("RBM10", "p.G17Wfs*4", "Frame_Shift_Ins",
                     tumor_vaf=0.42, is_hotspot=False, evidence_score=4),
        ],
        tmb=4.6,
        pd_l1_tps=15.0,
        msi_status="MSS",
        treatment_history=[
            TreatmentEvent(
                line=1,
                drug="carboplatin + pemetrexed + pembrolizumab",
                start_date="2025-11-10",
                end_date="2026-06-20",
                best_response="PD",
                discontinuation_reason="progression",
            ),
        ],
    )


if __name__ == "__main__":
    p = example_patient_kras_g12c()
    print(f"Patient: {p.patient_id}")
    print(f"Diagnosis: {p.diagnosis}, stage {p.stage}")
    print(f"Age {p.age_years}, sex {p.sex}, ECOG {p.ecog}, smoking {p.smoking_status}")
    print(f"Brain mets: {p.brain_metastases}")
    print(f"TMB: {p.tmb} mut/Mb, PD-L1 TPS: {p.pd_l1_tps}%")
    print()
    print("Mutations:")
    for m in p.mutations:
        h = " (HOTSPOT)" if m.is_hotspot else ""
        print(f"  {m.gene:8} {m.protein_change:14} VAF={m.tumor_vaf:.2f}  "
              f"evidence={m.evidence_score}/6{h}")
    print()
    print(f"Treatment history (next line: {p.current_line()}):")
    for t in p.treatment_history:
        print(f"  line {t.line}: {t.drug}")
        print(f"    {t.start_date} to {t.end_date}, best response: {t.best_response}, "
              f"discontinued: {t.discontinuation_reason}")
    print()
    print(f"Has KRAS G12C: {p.has_mutation('KRAS', 'G12C')}")
    print(f"Has EGFR L858R: {p.has_mutation('EGFR', 'L858R')}")
    print(f"Prior response to pembrolizumab: {p.prior_response_to('pembrolizumab')}")
