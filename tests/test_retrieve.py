"""Retrieval tests. Uses the real MiniLM model (downloaded once, ~90 MB) but no LLM."""

from __future__ import annotations

from trial_matcher.retrieve import HybridIndex, build_query, retrieve_candidates, rrf_fuse

SOTORASIB = "NCT03600883"


def test_kras_g12c_patient_gets_sotorasib_in_top5(patient, retrieval_corpus, tmp_path):
    cands = retrieve_candidates(patient, retrieval_corpus, top_k=5, index_dir=tmp_path)
    ids = [c["nct_id"] for c in cands]
    assert len(cands) == 5
    assert SOTORASIB in ids, f"top 5 was {ids}"
    assert all("retrieval_score" in c for c in cands)
    scores = [c["retrieval_score"] for c in cands]
    assert scores == sorted(scores, reverse=True)


def test_query_carries_hotspot_variant_but_not_prior_drugs(patient):
    q = build_query(patient)
    assert "KRAS G12C" in q
    assert "pembrolizumab" not in q.lower()


def test_rrf_fusion_math():
    # doc 0 is 1st and 3rd; doc 1 is 2nd and 1st; doc 2 is 3rd and 2nd
    fused = rrf_fuse([[0, 1, 2], [1, 2, 0]], k=60)
    assert abs(fused[0] - (1 / 61 + 1 / 63)) < 1e-12
    assert abs(fused[1] - (1 / 62 + 1 / 61)) < 1e-12
    assert max(fused, key=fused.get) == 1


def test_index_is_persisted_and_reused(sample_trials, tmp_path):
    _, built_first = HybridIndex.load_or_build(sample_trials, tmp_path)
    _, built_second = HybridIndex.load_or_build(sample_trials, tmp_path)
    assert built_first is True
    assert built_second is False
    assert any(p.name.endswith(".emb.npy") for p in tmp_path.iterdir())


def test_changed_corpus_rebuilds_index(sample_trials, tmp_path):
    HybridIndex.load_or_build(sample_trials, tmp_path)
    _, rebuilt = HybridIndex.load_or_build(sample_trials[:2], tmp_path)
    assert rebuilt is True
