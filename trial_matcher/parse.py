"""
Parse raw ClinicalTrials.gov trials into a normalized structure.

Every trial becomes:
  {
    nct_id, title, phase, conditions, interventions,
    min_age, max_age, sex, ecog_range,  # extracted structured facts
    criteria: [
      {kind: "inclusion" | "exclusion", text: "...", idx: 0, ...structured extractions},
      ...
    ]
  }

Structured extraction here is deliberately shallow: regex for age, sex, ECOG,
biomarkers, prior therapy flags. Anything more complex is left to the LLM
matcher downstream. The point of this stage is to (1) split criteria into
addressable units and (2) attach cheap features for retrieval and for the
matcher to condition on.
"""

import json
import re
from pathlib import Path

# ---- Structured feature extractors -----------------------------------------

BIOMARKER_RE = re.compile(
    r"\b(KRAS|EGFR|BRAF|ALK|ROS1|MET|RET|HER2|ERBB2|PIK3CA|STK11|KEAP1|"
    r"NTRK\d|TP53|NRAS|BRCA1|BRCA2|PD-L1|PDL1|TMB|MSI)\b",
    re.IGNORECASE,
)

# Match specific mutations like G12C, L858R, T790M, V600E, exon 19 deletion
MUTATION_RE = re.compile(
    r"\b([A-Z]\d{1,4}[A-Z*]|exon\s*\d+\s*(?:deletion|insertion|skipping)|"
    r"amplification|fusion|rearrangement)\b",
    re.IGNORECASE,
)

ECOG_RE = re.compile(
    r"ECOG.{0,40}?(?:performance status\s*)?(?:of\s*)?"
    r"(?:<=|≤|less than or equal to|of at most)?\s*"
    r"(\d)(?:\s*(?:to|-|or)\s*(\d))?",
    re.IGNORECASE,
)

AGE_RE = re.compile(r"(?:age|aged)\s*(?:>=|≥|at least|of)\s*(\d+)\s*years?", re.IGNORECASE)

# Detect keywords the matcher cares about
PRIOR_THERAPY_RE = re.compile(
    r"\b(prior|previous|previously)\b.{0,50}?"
    r"\b(treatment|therapy|line|systemic|chemotherapy|TKI|immunotherapy|inhibitor)\b",
    re.IGNORECASE,
)

BRAIN_METS_RE = re.compile(r"\bbrain\s+metasta[sc]e[sd]?\b|CNS metasta", re.IGNORECASE)


def extract_features(text):
    """Cheap, deterministic feature extraction from criterion text."""
    feats = {}

    genes = list({m.group(1).upper() for m in BIOMARKER_RE.finditer(text)})
    if genes:
        feats["biomarkers"] = genes

    mutations = list({m.group(1).upper() for m in MUTATION_RE.finditer(text)})
    if mutations:
        feats["mutations"] = mutations

    ecog = ECOG_RE.search(text)
    if ecog:
        low = int(ecog.group(1))
        high = int(ecog.group(2)) if ecog.group(2) else low
        feats["ecog_max"] = max(low, high)

    age = AGE_RE.search(text)
    if age:
        feats["min_age_years"] = int(age.group(1))

    if PRIOR_THERAPY_RE.search(text):
        feats["prior_therapy_relevant"] = True

    if BRAIN_METS_RE.search(text):
        feats["mentions_brain_mets"] = True

    return feats


# ---- Criterion splitter ----------------------------------------------------

def split_criteria(eligibility_text):
    """
    Split the raw eligibility block into inclusion and exclusion criteria.
    ClinicalTrials.gov usually formats these as:
        Inclusion Criteria:
        * item 1
        * item 2
        Exclusion Criteria:
        * item 1
        ...
    but formatting varies. This handles the common cases.
    """
    if not eligibility_text:
        return []

    # Normalise bullets
    text = eligibility_text.replace("•", "*").replace("•", "*")

    # Find inclusion / exclusion section markers
    inc_pat = re.compile(r"inclusion\s+criteria\s*:?", re.IGNORECASE)
    exc_pat = re.compile(r"exclusion\s+criteria\s*:?", re.IGNORECASE)

    inc_m = inc_pat.search(text)
    exc_m = exc_pat.search(text)

    if inc_m and exc_m:
        inc_text = text[inc_m.end():exc_m.start()]
        exc_text = text[exc_m.end():]
    elif inc_m:
        inc_text = text[inc_m.end():]
        exc_text = ""
    elif exc_m:
        inc_text = ""
        exc_text = text[exc_m.end():]
    else:
        # No section headers; treat everything as unlabelled criteria
        inc_text = text
        exc_text = ""

    def split_bullets(block):
        """Split a block into individual criterion strings on bullets or newlines."""
        # Prefer bullet-split; if no bullets, fall back to blank-line split
        parts = re.split(r"(?:^|\n)\s*\*\s*", block)
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) <= 1:
            parts = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
        # Drop empty and clean up whitespace
        return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]

    criteria = []
    for text_block, kind in [(inc_text, "inclusion"), (exc_text, "exclusion")]:
        for i, txt in enumerate(split_bullets(text_block)):
            criteria.append({
                "kind": kind,
                "idx": i,
                "text": txt,
                "features": extract_features(txt),
            })
    return criteria


# ---- Trial-level parser ----------------------------------------------------

def parse_trial(raw):
    """Convert one raw ClinicalTrials.gov study record into the normalized shape."""
    proto = raw.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    cond = proto.get("conditionsModule", {})
    arms = proto.get("armsInterventionsModule", {})
    elig = proto.get("eligibilityModule", {})
    locs = proto.get("contactsLocationsModule", {}).get("locations", [])

    criteria = split_criteria(elig.get("eligibilityCriteria", ""))

    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "status": status.get("overallStatus"),
        "phase": design.get("phases", []),
        "study_type": design.get("studyType"),
        "conditions": cond.get("conditions", []),
        "interventions": [
            {"type": i.get("type"), "name": i.get("name")}
            for i in arms.get("interventions", [])
        ],
        "min_age": elig.get("minimumAge"),
        "max_age": elig.get("maximumAge"),
        "sex": elig.get("sex"),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "countries": sorted({loc.get("country") for loc in locs if loc.get("country")}),
        "n_criteria": len(criteria),
        "criteria": criteria,
    }


def parse_file(in_path, out_path):
    raw = json.loads(Path(in_path).read_text())
    parsed = [parse_trial(t) for t in raw]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(parsed, indent=2))
    return parsed


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default="data/raw/sample_trials.json")
    ap.add_argument("--out", default="data/processed/trials_parsed.json")
    args = ap.parse_args()

    parsed = parse_file(args.in_path, args.out)
    print(f"Parsed {len(parsed)} trials")
    print(f"Total criteria: {sum(t['n_criteria'] for t in parsed)}")
    print(f"Saved: {args.out}")

    # Show a summary
    print("\nPer-trial breakdown:")
    for t in parsed:
        inc = sum(1 for c in t["criteria"] if c["kind"] == "inclusion")
        exc = sum(1 for c in t["criteria"] if c["kind"] == "exclusion")
        genes = sorted({g for c in t["criteria"] for g in c["features"].get("biomarkers", [])})
        print(f"  {t['nct_id']:>14}  inc={inc:2d} exc={exc:2d}  "
              f"biomarkers={genes if genes else '-'}")


if __name__ == "__main__":
    main()
