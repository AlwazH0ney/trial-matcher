"""
luad-match: command-line entry point for Phase 2.

    luad-match example-patient --out data/processed/patient_example.json
    luad-match retrieve --patient PATH --trials PATH --top-k 50 --out PATH
    luad-match evaluate --patient PATH --candidates PATH --out PATH
    luad-match rank --evaluated PATH --out reports/matches.md

Every path is an argument; defaults are relative to the working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from trial_matcher import __version__, prompts
from trial_matcher.llm import (DEFAULT_CLAUDE_MODEL, DEFAULT_LOG_PATH, DEFAULT_OLLAMA_MODEL,
                               ClaudeBackend, JsonlLogger, LLMError, OllamaBackend,
                               OllamaUnavailableError)
from trial_matcher.match import evaluate_trial, rank_trials
from trial_matcher.patient import PatientProfile, example_patient_kras_g12c
from trial_matcher.retrieve import (DEFAULT_EMBED_MODEL, DEFAULT_INDEX_DIR, build_query,
                                    retrieve_candidates)


def _now() -> str:
    """UTC timestamp for output provenance."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path | str) -> Any:
    """Read a JSON file, failing with the path in the message if it is missing."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: Path | str, data: Any) -> None:
    """Write JSON atomically so an interrupted run (e.g. a Colab disconnect) never leaves a torn file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, p)


def load_patient(path: str) -> PatientProfile:
    """Load a patient JSON, or the built-in example when path is 'example'."""
    if path == "example":
        return example_patient_kras_g12c()
    return PatientProfile.from_dict(_read_json(path))


def _load_trials(path: str) -> list[dict[str, Any]]:
    """Accept either a bare list of trials or a candidates file from `retrieve`."""
    data = _read_json(path)
    trials = data["candidates"] if isinstance(data, dict) and "candidates" in data else data
    if not isinstance(trials, list):
        raise SystemExit(f"error: {path} is neither a trial list nor a candidates file")
    return trials


# ---- Subcommands ------------------------------------------------------------

def cmd_example_patient(args: argparse.Namespace) -> int:
    """Write the built-in KRAS G12C example patient to JSON for use with --patient."""
    p = example_patient_kras_g12c()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    p.save(args.out)
    print(f"Saved example patient {p.patient_id} to {args.out}")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    """Hybrid-retrieve candidate trials for a patient and save them."""
    patient = load_patient(args.patient)
    trials = _load_trials(args.trials)
    query = build_query(patient)
    t0 = time.perf_counter()
    cands = retrieve_candidates(patient, trials, top_k=args.top_k,
                                index_dir=args.index_dir, model_name=args.embed_model)
    _write_json(args.out, {
        "patient_id": patient.patient_id, "query": query, "generated_at": _now(),
        "trials_file": str(args.trials), "corpus_size": len(trials),
        "top_k": args.top_k, "embed_model": args.embed_model, "candidates": cands,
    })
    print(f"Query: {query}")
    print(f"Retrieved {len(cands)} of {len(trials)} trials in {time.perf_counter() - t0:.1f}s "
          f"-> {args.out}")
    for i, c in enumerate(cands[:10], 1):
        r = c["retrieval"]
        print(f"  {i:2d}. {c['nct_id']}  rrf={c['retrieval_score']:.4f}  "
              f"bm25#{r['bm25_rank']:<3d} dense#{r['dense_rank']:<3d} {(c.get('title') or '')[:70]}")
    return 0


def run_summary(trials: list[dict[str, Any]], wall_time_s: float) -> dict[str, Any]:
    """Headline numbers for a run: volume, speed, routing rate, unresolved count."""
    crits = [c for t in trials for c in t["criteria"]]
    n = len(crits)
    local_lat = [c["local"]["latency_s"] for c in crits]
    total_lat = [c["local"]["latency_s"] + (c["api"]["latency_s"] if c["api"] else 0.0)
                 for c in crits]
    in_band = sum(1 for c in crits if c["route"] in ("api", "api_unavailable"))
    return {
        "n_trials": len(trials),
        "n_criteria": n,
        "n_trials_by_status": {s: sum(1 for t in trials if t["status"] == s)
                               for s in ("eligible", "unresolved", "ineligible")},
        "n_unresolved_trials": sum(1 for t in trials if t["status"] == "unresolved"),
        "n_unknown_criteria": sum(1 for c in crits if c["eligible"] is None),
        "mean_local_latency_s": round(sum(local_lat) / n, 3) if n else None,
        "mean_criterion_time_s": round(sum(total_lat) / n, 3) if n else None,
        "routing_band_rate": round(in_band / n, 4) if n else None,
        "n_routed_api": sum(1 for c in crits if c["route"] == "api"),
        "api_routing_rate": round(sum(1 for c in crits if c["route"] == "api") / n, 4) if n else None,
        "n_api_unavailable": sum(1 for c in crits if c["route"] == "api_unavailable"),
        "n_api_flipped": sum(1 for c in crits if c["api"] and
                             c["api"]["eligible"] != c["local"]["eligible"]),
        "n_parse_errors": sum(1 for c in crits
                              if c["local"]["parse_error"] or (c["api"] or {}).get("parse_error")),
        "input_tokens": {
            "local": sum(c["local"]["input_tokens"] or 0 for c in crits),
            "api": sum((c["api"] or {}).get("input_tokens") or 0 for c in crits)},
        "output_tokens": {
            "local": sum(c["local"]["output_tokens"] or 0 for c in crits),
            "api": sum((c["api"] or {}).get("output_tokens") or 0 for c in crits)},
        "wall_time_s": round(wall_time_s, 1),
    }


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Judge every criterion of every candidate trial and save per-trial results.

    Results are checkpointed after each trial; --resume skips trials already in
    --out, so a dropped Colab session loses at most one trial of work.
    """
    patient = load_patient(args.patient)
    trials = _load_trials(args.candidates)
    if args.max_trials:
        trials = trials[:args.max_trials]
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    summary_text = prompts.summarize_patient(patient, as_of)
    logger = JsonlLogger(args.log)

    try:
        local = OllamaBackend(model=args.ollama_model, logger=logger)
    except OllamaUnavailableError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    api = None if args.no_api else ClaudeBackend.from_env(model=args.api_model, logger=logger)
    if api is None:
        why = "--no-api given" if args.no_api else "ANTHROPIC_API_KEY is not set"
        print(f"warning: API backend disabled ({why}); criteria with local confidence in "
              f"[0.35, 0.75] keep the local verdict, marked route='api_unavailable'.",
              file=sys.stderr)

    done: list[dict[str, Any]] = []
    prior_wall = 0.0
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        prev = _read_json(out_path)
        if prev.get("patient_id") != patient.patient_id or \
           prev.get("prompt_version") != prompts.PROMPT_VERSION:
            raise SystemExit(f"error: {out_path} was produced for a different patient or prompt "
                             f"version; use a new --out or drop --resume")
        done = prev.get("trials", [])
        prior_wall = prev.get("summary", {}).get("wall_time_s", 0.0)
        print(f"Resuming: {len(done)} trials already evaluated in {out_path}")
    done_ids = {t["nct_id"] for t in done}
    todo = [t for t in trials if t.get("nct_id") not in done_ids]

    header = {
        "patient_id": patient.patient_id, "generated_at": _now(), "version": __version__,
        "prompt_version": prompts.PROMPT_VERSION, "as_of": as_of.isoformat(),
        "candidates_file": str(args.candidates),
        "backends": {"local": local.describe(), "api": api.describe() if api else None},
        "routing_band": [0.35, 0.75], "patient_summary": summary_text,
    }
    n_crit = sum(len(t.get("criteria") or []) for t in todo)
    print(f"Evaluating {len(todo)} trials ({n_crit} criteria) with {local.model} @ {local.host}"
          + (f", API re-query via {api.model}" if api else ""))

    t0 = time.perf_counter()
    try:
        for i, trial in enumerate(todo, 1):
            ts = time.perf_counter()
            res = evaluate_trial(patient, trial, local, api, patient_summary=summary_text)
            done.append(res)
            wall = prior_wall + time.perf_counter() - t0
            _write_json(out_path, {**header, "summary": run_summary(done, wall), "trials": done})
            print(f"  [{i}/{len(todo)}] {res['nct_id']}: {res['status']:<10} "
                  f"score={res['score']:.3f}  criteria={res['n_criteria']} "
                  f"routed={res['n_routed_api']}  ({time.perf_counter() - ts:.1f}s)", flush=True)
    except OllamaUnavailableError as e:
        print(f"error: {e}\nPartial results saved to {out_path}; re-run with --resume.",
              file=sys.stderr)
        return 2
    except LLMError as e:
        print(f"error: LLM call failed after retries: {e}\nPartial results saved to {out_path}; "
              f"re-run with --resume.", file=sys.stderr)
        return 3

    summary = run_summary(done, prior_wall + time.perf_counter() - t0)
    _write_json(out_path, {**header, "summary": summary, "trials": done})
    print(f"\nSaved {len(done)} evaluated trials -> {out_path}")
    print(json.dumps(summary, indent=2))
    return 0


def _md_escape(s: Optional[str]) -> str:
    """Keep table cells intact when text contains pipes or newlines."""
    return (s or "").replace("|", "\\|").replace("\n", " ")


def render_markdown(ev: dict[str, Any], ranked: list[dict[str, Any]], top: int) -> str:
    """Human-readable match report: ranked table, then evidence per recommended trial."""
    s = ev.get("summary", {})
    api = (ev.get("backends") or {}).get("api")
    local = (ev.get("backends") or {}).get("local") or {}
    lines = [
        f"# Trial matches for {ev.get('patient_id')}",
        "",
        f"Generated {ev.get('generated_at')} · prompt `{ev.get('prompt_version')}` · "
        f"local `{local.get('model')}` · API `{api['model'] if api else 'disabled'}`",
        "",
        "> Research prototype. Not for clinical use.",
        "",
        "```", ev.get("patient_summary", ""), "```",
        "",
        f"{s.get('n_trials')} trials, {s.get('n_criteria')} criteria · "
        f"API routing rate {s.get('api_routing_rate')} · "
        f"unknown criteria {s.get('n_unknown_criteria')} · "
        f"unresolved trials {s.get('n_unresolved_trials')}",
        "",
        "| Rank | NCT ID | Status | Score | Retrieval | Pass/Fail/Unknown | Title |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for i, t in enumerate(ranked[:top], 1):
        lines.append(
            f"| {i} | [{t['nct_id']}](https://clinicaltrials.gov/study/{t['nct_id']}) | "
            f"{t['status']} | {t['score']:.3f} | {t.get('retrieval_score') or 0:.4f} | "
            f"{t['n_pass']}/{t['n_fail']}/{t['n_unknown']} | {_md_escape(t.get('title'))} |")
    for i, t in enumerate(ranked[:top], 1):
        lines += ["", f"## {i}. {t['nct_id']} — {t['status']} (score {t['score']:.3f})", "",
                  f"{_md_escape(t.get('title'))}", ""]
        if t.get("blocking"):
            lines.append("**Blocking criteria**")
            for b in t["blocking"]:
                lines.append(f"- *{b['kind']} #{b['idx']}*: {_md_escape(b['text'])}  \n"
                             f"  → {_md_escape(b['reasoning'])} "
                             f"(evidence: {_md_escape(b['evidence']['patient_feature'])})")
            lines.append("")
        lines += ["| Kind | # | Criterion | Verdict | Conf | Passes | Route | Evidence |",
                  "|---|---:|---|---|---:|---|---|---|"]
        for c in t["criteria"]:
            verdict = {True: "met", False: "not met", None: "unknown"}[c["eligible"]]
            passes = {True: "✓", False: "✗", None: "?"}[c["passes"]]
            lines.append(
                f"| {c['kind'][:3]} | {c['idx']} | {_md_escape(c['text'][:140])} | {verdict} | "
                f"{c['confidence']:.2f} | {passes} | {c['route']} | "
                f"{_md_escape(c['evidence']['patient_feature'])} |")
    return "\n".join(lines) + "\n"


def cmd_rank(args: argparse.Namespace) -> int:
    """Rank evaluated trials and write the markdown report."""
    ev = _read_json(args.evaluated)
    ranked = rank_trials(ev.get("trials", []))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(ev, ranked, args.top), encoding="utf-8")
    print(f"Ranked {len(ranked)} trials -> {out}\n")
    print(f"{'#':>3}  {'NCT ID':<12} {'status':<11} {'score':>6}  pass/fail/unk  title")
    for i, t in enumerate(ranked[:args.top], 1):
        print(f"{i:>3}  {t['nct_id']:<12} {t['status']:<11} {t['score']:>6.3f}  "
              f"{t['n_pass']:>4}/{t['n_fail']}/{t['n_unknown']:<5} {(t.get('title') or '')[:60]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define the luad-match subcommands and their path arguments."""
    ap = argparse.ArgumentParser(prog="luad-match", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("example-patient", help="write the built-in KRAS G12C example patient")
    p.add_argument("--out", default="data/processed/patient_example.json")
    p.set_defaults(func=cmd_example_patient)

    p = sub.add_parser("retrieve", help="hybrid BM25 + embedding candidate retrieval")
    p.add_argument("--patient", required=True, help="patient JSON path, or 'example'")
    p.add_argument("--trials", required=True, help="parsed trials JSON")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--out", default="data/processed/candidates.json")
    p.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("evaluate", help="per-criterion LLM judgments with hybrid routing")
    p.add_argument("--patient", required=True, help="patient JSON path, or 'example'")
    p.add_argument("--candidates", required=True, help="output of `retrieve`, or a trial list")
    p.add_argument("--out", default="data/processed/evaluated.json")
    p.add_argument("--log", default=str(DEFAULT_LOG_PATH), help="JSONL call log")
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    p.add_argument("--api-model", default=DEFAULT_CLAUDE_MODEL)
    p.add_argument("--no-api", action="store_true", help="never re-query via the API")
    p.add_argument("--as-of", help="reference date for the patient summary (YYYY-MM-DD)")
    p.add_argument("--max-trials", type=int, help="evaluate only the first N candidates")
    p.add_argument("--resume", action="store_true", help="skip trials already in --out")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("rank", help="rank evaluated trials and write a markdown report")
    p.add_argument("--evaluated", required=True)
    p.add_argument("--out", default="reports/matches.md")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_rank)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console-script entry point (luad-match)."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
