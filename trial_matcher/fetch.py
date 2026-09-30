"""
Fetch recruiting trials from ClinicalTrials.gov API v2.

Usage:
    python -m trial_matcher.fetch --condition "Lung Adenocarcinoma" --limit 200
    python -m trial_matcher.fetch --condition "Lung Adenocarcinoma" --status RECRUITING --limit 500
    python -m trial_matcher.fetch --help

API docs: https://clinicaltrials.gov/data-api/api
"""

import argparse
import json
import time
from pathlib import Path

import requests

API = "https://clinicaltrials.gov/api/v2/studies"

# The fields we actually need. Requesting all fields returns ~500 KB per trial;
# this list keeps a fetch of 500 trials under 10 MB.
FIELDS = [
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "OverallStatus",
    "StudyType",
    "Phase",
    "Condition",
    "InterventionName",
    "InterventionType",
    "EligibilityCriteria",
    "MinimumAge",
    "MaximumAge",
    "Sex",
    "HealthyVolunteers",
    "StdAge",
    "LocationCountry",
    "LocationCity",
    "OrgFullName",
    "LastUpdatePostDate",
    "StartDate",
    "PrimaryCompletionDate",
]


def fetch_page(condition, status, page_token=None, page_size=100):
    params = {
        "query.cond": condition,
        "filter.overallStatus": status,
        "fields": ",".join(FIELDS),
        "pageSize": page_size,
        "format": "json",
    }
    if page_token:
        params["pageToken"] = page_token
    r = requests.get(API, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_all(condition, status, limit):
    out = []
    token = None
    while len(out) < limit:
        page_size = min(100, limit - len(out))
        page = fetch_page(condition, status, token, page_size)
        studies = page.get("studies", [])
        if not studies:
            break
        out.extend(studies)
        token = page.get("nextPageToken")
        print(f"  fetched {len(out)} / {limit} trials")
        if not token:
            break
        time.sleep(0.3)  # be nice to the API
    return out[:limit]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="Lung Adenocarcinoma",
                    help="Disease name for query.cond (default: Lung Adenocarcinoma)")
    ap.add_argument("--status", default="RECRUITING",
                    choices=["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED",
                             "NOT_YET_RECRUITING", "TERMINATED"],
                    help="Trial status filter (default: RECRUITING)")
    ap.add_argument("--limit", type=int, default=200,
                    help="Max trials to fetch (default: 200)")
    ap.add_argument("--out", default="data/raw/trials.json",
                    help="Output JSON path")
    args = ap.parse_args()

    print(f"Fetching {args.status} trials for '{args.condition}' (max {args.limit})...")
    trials = fetch_all(args.condition, args.status, args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(trials, fh, indent=2)

    print(f"\nSaved {len(trials)} trials to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
