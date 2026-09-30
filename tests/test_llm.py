"""Parse/repair, prompt, and Ollama-availability tests. No model is ever called."""

from __future__ import annotations

import pytest

from trial_matcher import prompts
from trial_matcher.llm import (JudgmentParseError, OllamaBackend, OllamaUnavailableError,
                               parse_judgment, resolve_ollama_host, with_retries,
                               TransientLLMError)


@pytest.mark.parametrize("raw, eligible, conf", [
    ('{"eligible": true, "confidence": 0.9, "reasoning": "r", "patient_feature": "f"}', True, 0.9),
    ('```json\n{"eligible": false, "confidence": 0.8, "reasoning": "r"}\n```', False, 0.8),
    ('Sure! Here is my answer: {"eligible": null, "confidence": 0.7, "reasoning": "r"} Hope it helps',
     None, 0.7),
    ('{"eligible": "unknown", "confidence": "85%", "reasoning": "r",}', None, 0.85),
    ('{"eligible": True, "confidence": 95, "reasoning": "r"}', True, 0.95),
    ('{"verdict": "ineligible", "confidence": 0.6, "rationale": "r"}', False, 0.6),
    ('{"eligible": unknown, "confidence": 0.5, "reasoning": "has {braces} inside"}', None, 0.5),
])
def test_parse_and_repair(raw, eligible, conf):
    j, _ = parse_judgment(raw)
    assert j.eligible is eligible
    assert j.confidence == pytest.approx(conf)


def test_clean_json_is_not_marked_repaired():
    _, repaired = parse_judgment('{"eligible": true, "confidence": 0.9, "reasoning": "r", '
                                 '"patient_feature": "f"}')
    assert repaired is False


@pytest.mark.parametrize("raw", ["I cannot determine this.", '{"eligible": "perhaps", "confidence": 0.5}'])
def test_unparseable_output_raises(raw):
    with pytest.raises(JudgmentParseError):
        parse_judgment(raw)


def test_one_shot_example_parses_with_our_own_parser():
    j, repaired = parse_judgment(prompts.EXAMPLE_RESPONSE)
    assert j.eligible is True and not repaired


def test_patient_summary_contents(patient):
    s = prompts.summarize_patient(patient)
    for needle in ("KRAS p.G12C (hotspot)", "TMB 4.6 mut/Mb", "PD-L1 TPS 15%", "ECOG 1",
                   "Treatment history (complete)", "pembrolizumab", "Next line of therapy: 2",
                   "Prior malignancy: not recorded"):
        assert needle in s, needle


def test_messages_are_one_shot_then_query(patient):
    msgs = prompts.build_messages(prompts.summarize_patient(patient), "Age >= 18 years.")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == prompts.EXAMPLE_RESPONSE
    assert msgs[2]["content"].endswith("CRITERION\nAge >= 18 years.")


def test_retries_back_off_then_succeed():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise TransientLLMError("503")
        return "ok"

    assert with_retries(fn, max_attempts=4, base_delay=1.0, sleep=sleeps.append) == ("ok", 3)
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


@pytest.mark.parametrize("env, expected", [
    (None, "http://localhost:11434"),
    ("127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("http://10.0.0.5:11434/", "http://10.0.0.5:11434"),
])
def test_ollama_host_resolution(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", env)
    assert resolve_ollama_host() == expected


def test_unreachable_ollama_fails_loudly(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")   # discard port: nothing listens
    with pytest.raises(OllamaUnavailableError) as exc:
        OllamaBackend(timeout_s=5)
    msg = str(exc.value)
    assert "127.0.0.1:9" in msg
    assert "ollama serve" in msg and "ollama pull phi4" in msg
    assert "Not falling back" in msg
