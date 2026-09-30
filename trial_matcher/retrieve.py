"""
Hybrid trial retrieval: BM25 + dense embeddings fused with reciprocal rank fusion.

BM25 catches exact biomarker tokens ("G12C", "KRAS") that embeddings blur;
MiniLM embeddings catch paraphrase ("non-small cell lung cancer" vs
"adenocarcinoma of the lung") that BM25 misses. RRF fuses ranks rather than
raw scores, so the two scales never need calibrating against each other.

Indexes are cached under --index-dir, keyed by a hash of the corpus text and
embedding model, so a different trials file never reuses a stale index.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from trial_matcher.patient import PatientProfile

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = Path("data/processed/index")
RRF_K = 60

_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the",
    "to", "with", "without", "vs", "versus", "study", "trial", "phase",
    "patients", "participants", "subjects",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords dropped; keeps variant tokens like 'g12c' intact."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOPWORDS and len(t) > 1]


def trial_document(trial: dict[str, Any]) -> str:
    """The text a trial is indexed by: titles, conditions and intervention names."""
    parts = [trial.get("title") or "", trial.get("official_title") or ""]
    parts += trial.get("conditions") or []
    parts += [i.get("name") or "" for i in trial.get("interventions") or []]
    return " . ".join(p for p in parts if p)


def build_query(patient: PatientProfile) -> str:
    """Turn a patient into a retrieval query focused on disease and actionable biology.

    Prior drugs are deliberately left out: including "pembrolizumab" would pull
    trials of the therapy the patient already progressed on.
    """
    terms = [patient.diagnosis, f"stage {patient.stage}"]
    if patient.histology and patient.histology.lower() not in patient.diagnosis.lower():
        terms.append(patient.histology)
    dx = patient.diagnosis.lower()
    if "lung" in dx and "adenocarcinoma" in dx:
        terms.append("non-small cell lung cancer NSCLC")
    for m in patient.mutations:
        change = m.protein_change.removeprefix("p.")
        terms.append(f"{m.gene} {change}" if m.is_hotspot else m.gene)
        if m.is_hotspot:
            terms.append(f"{m.gene} {change} mutation")   # weight actionable hotspots
    if patient.treatment_history:
        terms.append("previously treated advanced metastatic")
    return " ".join(terms)


@lru_cache(maxsize=2)
def _load_encoder(model_name: str):
    """Load a sentence-transformers model once per process (slow to construct)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _encode(model_name: str, texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings, so cosine similarity is a plain dot product."""
    emb = _load_encoder(model_name).encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32)


def corpus_fingerprint(trials: list[dict[str, Any]], model_name: str) -> str:
    """Stable hash of the indexed text and embedding model; names the cache files."""
    h = hashlib.sha256(model_name.encode())
    for t in trials:
        h.update(f"{t.get('nct_id')}\x1f{trial_document(t)}\x1e".encode("utf-8"))
    return h.hexdigest()[:16]


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank), rank from 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class HybridIndex:
    """BM25 + embedding index over one trial corpus."""

    fingerprint: str
    nct_ids: list[str]
    bm25: BM25Okapi
    embeddings: np.ndarray
    model_name: str

    @classmethod
    def build(cls, trials: list[dict[str, Any]], model_name: str = DEFAULT_EMBED_MODEL) -> "HybridIndex":
        """Index a corpus from scratch (embedding is the slow part)."""
        docs = [trial_document(t) for t in trials]
        return cls(
            fingerprint=corpus_fingerprint(trials, model_name),
            nct_ids=[t.get("nct_id") for t in trials],
            bm25=BM25Okapi([tokenize(d) or ["_empty_"] for d in docs]),
            embeddings=_encode(model_name, docs),
            model_name=model_name,
        )

    def _paths(self, index_dir: Path) -> tuple[Path, Path, Path]:
        """File names for this corpus's BM25 pickle, embeddings and metadata."""
        stem = index_dir / f"hybrid_{self.fingerprint}"
        return stem.with_suffix(".bm25.pkl"), stem.with_suffix(".emb.npy"), stem.with_suffix(".meta.json")

    def save(self, index_dir: Path) -> None:
        """Persist both indexes so later runs skip re-embedding."""
        index_dir.mkdir(parents=True, exist_ok=True)
        bm25_p, emb_p, meta_p = self._paths(index_dir)
        with open(bm25_p, "wb") as fh:
            pickle.dump(self.bm25, fh)
        np.save(emb_p, self.embeddings)
        meta_p.write_text(json.dumps({"model_name": self.model_name, "nct_ids": self.nct_ids}))

    @classmethod
    def load_or_build(cls, trials: list[dict[str, Any]], index_dir: Path | str = DEFAULT_INDEX_DIR,
                      model_name: str = DEFAULT_EMBED_MODEL) -> tuple["HybridIndex", bool]:
        """Load the cached index for this exact corpus, or build and save it.

        Returns (index, built) so callers and tests can tell a cache hit from a rebuild.
        """
        index_dir = Path(index_dir)
        fp = corpus_fingerprint(trials, model_name)
        stub = cls(fp, [], None, np.empty(0), model_name)  # type: ignore[arg-type]
        bm25_p, emb_p, meta_p = stub._paths(index_dir)
        if bm25_p.exists() and emb_p.exists() and meta_p.exists():
            meta = json.loads(meta_p.read_text())
            with open(bm25_p, "rb") as fh:
                bm25 = pickle.load(fh)
            return cls(fp, meta["nct_ids"], bm25, np.load(emb_p), model_name), False
        index = cls.build(trials, model_name)
        index.save(index_dir)
        return index, True

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Rank the whole corpus by each method, fuse with RRF, return the top_k hits."""
        bm25_scores = self.bm25.get_scores(tokenize(query))
        dense_scores = self.embeddings @ _encode(self.model_name, [query])[0]
        bm25_rank = list(np.argsort(-bm25_scores, kind="stable"))
        dense_rank = list(np.argsort(-dense_scores, kind="stable"))
        fused = rrf_fuse([bm25_rank, dense_rank])
        bm25_pos = {int(d): r for r, d in enumerate(bm25_rank, start=1)}
        dense_pos = {int(d): r for r, d in enumerate(dense_rank, start=1)}
        order = sorted(fused, key=lambda d: (-fused[d], bm25_pos[int(d)]))[:top_k]
        return [{
            "doc": int(d),
            "rrf": fused[d],
            "bm25_rank": bm25_pos[int(d)],
            "dense_rank": dense_pos[int(d)],
            "bm25_score": float(bm25_scores[d]),
            "dense_score": float(dense_scores[d]),
        } for d in order]


def retrieve_candidates(patient: PatientProfile, trials: list[dict[str, Any]], top_k: int = 50,
                        index_dir: Path | str = DEFAULT_INDEX_DIR,
                        model_name: str = DEFAULT_EMBED_MODEL,
                        query: Optional[str] = None) -> list[dict[str, Any]]:
    """Return the top_k trials for a patient, each copied with a retrieval_score.

    This is a recall stage only; eligibility is decided criterion by criterion
    in match.py, so top_k should stay generous.
    """
    if not trials:
        return []
    index, _ = HybridIndex.load_or_build(trials, index_dir, model_name)
    hits = index.search(query or build_query(patient), min(top_k, len(trials)))
    out = []
    for h in hits:
        t = dict(trials[h["doc"]])
        t["retrieval_score"] = round(h["rrf"], 6)
        t["retrieval"] = {k: h[k] for k in ("bm25_rank", "dense_rank")} | {
            "bm25_score": round(h["bm25_score"], 4), "dense_score": round(h["dense_score"], 4)}
        out.append(t)
    return out
