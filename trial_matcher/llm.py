"""
LLM backends for per-criterion judgments.

Two backends share one prompt (prompts.py), one output schema
(CriterionJudgment), one parse + repair path, one retry policy, and one
JSONL call log:

  OllamaBackend  local phi4 via Ollama. Honors OLLAMA_HOST, so the same code
                 runs on a laptop and on a Colab T4 VM. Fails loudly if the
                 server or model is missing; it never falls back to the API.
  ClaudeBackend  claude-haiku-4-5-20251001 via the Anthropic API; used by the
                 router in match.py for mid-confidence re-queries.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from trial_matcher import prompts

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "phi4"
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_LOG_PATH = Path("data/processed/llm_log.jsonl")

T = TypeVar("T")


# ---- Errors ----------------------------------------------------------------

class LLMError(RuntimeError):
    """Base class for backend failures the caller should not silently absorb."""


class OllamaUnavailableError(LLMError):
    """Ollama is unreachable or lacks the model; raised so runs stop instead of stubbing."""


class TransientLLMError(LLMError):
    """A failure worth retrying (timeouts, 429, 5xx)."""


# ---- Output schema + parse/repair -------------------------------------------

_TRUE_WORDS = {"true", "yes", "eligible", "met", "meets", "y"}
_FALSE_WORDS = {"false", "no", "ineligible", "not met", "not_met", "n"}
_NULL_WORDS = {"null", "none", "unknown", "n/a", "na", "", "undetermined", "unclear"}


class CriterionJudgment(BaseModel):
    """The one output schema both backends must produce.

    `eligible` means "the criterion's statement is true for the patient";
    exclusion polarity is applied later in match.py.
    """

    eligible: Optional[bool]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    patient_feature: str = "none"

    @field_validator("eligible", mode="before")
    @classmethod
    def _coerce_eligible(cls, v: Any) -> Optional[bool]:
        """Accept the verdict spellings small models actually emit."""
        if v is None or isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in _TRUE_WORDS:
            return True
        if s in _FALSE_WORDS:
            return False
        if s in _NULL_WORDS:
            return None
        raise ValueError(f"unrecognised verdict {v!r}")

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: Any) -> float:
        """Normalise '85%', '0.85' and 85 to 0.85 and clamp to [0, 1]."""
        if isinstance(v, str):
            s = v.strip()
            pct = s.endswith("%")
            f = float(s.rstrip("%").strip())
            if pct:
                f /= 100.0
        else:
            f = float(v)
        if f > 1.0:
            f /= 100.0
        return min(max(f, 0.0), 1.0)

    @field_validator("reasoning", "patient_feature", mode="before")
    @classmethod
    def _coerce_text(cls, v: Any) -> str:
        """Tolerate null or non-string explanation fields."""
        return "" if v is None else str(v).strip()


JUDGMENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "eligible": {"type": ["boolean", "null"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "patient_feature": {"type": "string"},
    },
    "required": ["eligible", "confidence", "reasoning", "patient_feature"],
}


class JudgmentParseError(ValueError):
    """Model text could not be coerced into a CriterionJudgment."""


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} span, ignoring braces inside strings."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _repair_json(s: str) -> str:
    """Fix the common near-JSON mistakes: trailing commas, Python literals, bare unknown."""
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r'(:\s*)(unknown|undetermined)\b', r'\1null', s, flags=re.IGNORECASE)
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    return s


def parse_judgment(raw: str) -> tuple[CriterionJudgment, bool]:
    """Coerce raw model text into a CriterionJudgment.

    Returns (judgment, repaired) where `repaired` is True when the text needed
    any fix-up beyond plain json.loads; logged so format drift is visible.
    """
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    candidate = _extract_json_object(text)
    if candidate is None:
        raise JudgmentParseError(f"no JSON object in model output: {raw[:200]!r}")

    repaired = candidate.strip() != raw.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            data = json.loads(_repair_json(candidate))
            repaired = True
        except json.JSONDecodeError as e:
            raise JudgmentParseError(f"unrepairable JSON: {e}; output={raw[:200]!r}") from e
    if not isinstance(data, dict):
        raise JudgmentParseError(f"expected a JSON object, got {type(data).__name__}")

    # Map near-miss key names onto the schema.
    aliases = {"verdict": "eligible", "met": "eligible", "answer": "eligible",
               "reason": "reasoning", "rationale": "reasoning",
               "evidence": "patient_feature", "feature": "patient_feature"}
    for alt, canon in aliases.items():
        if alt in data and canon not in data:
            data[canon] = data.pop(alt)
            repaired = True
    try:
        return CriterionJudgment.model_validate(data), repaired
    except ValidationError as e:
        raise JudgmentParseError(f"schema mismatch: {e.errors()[:2]}; output={raw[:200]!r}") from e


# ---- Call result + logging ----------------------------------------------------

@dataclass
class JudgmentResult:
    """A judgment plus the provenance needed for routing analysis and cost tracking."""

    eligible: Optional[bool]
    confidence: float
    reasoning: str
    patient_feature: str
    backend: str                      # "local" | "api"
    model: str
    latency_s: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    attempts: int = 1
    repaired: bool = False
    parse_error: Optional[str] = None
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON output files."""
        return asdict(self)


@dataclass
class Completion:
    """Raw text plus token usage from one successful backend call."""

    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class JsonlLogger:
    """Append-only JSONL call log, one line per LLM call, for cost and drift analysis."""

    def __init__(self, path: Path | str | None = DEFAULT_LOG_PATH) -> None:
        """Create the log's parent directory up front so a long run cannot fail at the first write."""
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        """Append one record; a None path disables logging (used by some tests)."""
        if not self.path:
            return
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


@lru_cache(maxsize=1)
def detect_runtime() -> dict[str, Optional[str]]:
    """Describe where inference runs (laptop vs Colab, GPU name) for the call log.

    Latency numbers are meaningless without this: phi4 on a Colab T4 and phi4
    on a laptop CPU differ by an order of magnitude.
    """
    on_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            gpu = out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
        except (OSError, subprocess.SubprocessError):
            gpu = None
    return {"platform": "colab" if on_colab else sys.platform, "gpu": gpu}


# ---- Retry ------------------------------------------------------------------

def with_retries(fn: Callable[[], T], max_attempts: int = 4, base_delay: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep) -> tuple[T, int]:
    """Run fn, retrying TransientLLMError with exponential backoff and jitter.

    Returns (result, attempts). Non-transient errors propagate immediately so
    configuration mistakes (bad key, missing model) surface on the first call.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(), attempt
        except TransientLLMError:
            if attempt == max_attempts:
                raise
            sleep(base_delay * (2 ** (attempt - 1)) * (1 + random.random() * 0.25))
    raise AssertionError("unreachable")


# ---- Backends ---------------------------------------------------------------

class LLMBackend:
    """Shared judge() pipeline: prompt -> call with retries -> parse -> log.

    Subclasses implement only _complete(), so both backends are guaranteed to
    see the identical prompt and produce the identical output schema.
    """

    role: str = "local"
    model: str = ""

    def __init__(self, logger: JsonlLogger | None = None, max_attempts: int = 4,
                 base_delay: float = 1.0) -> None:
        """Hold the logger and retry policy shared by every call this backend makes."""
        self.logger = logger or JsonlLogger(None)
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    def _complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        """Send one chat request; raise TransientLLMError for retryable failures."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Backend identity for output files and logs."""
        return {"role": self.role, "model": self.model, "class": type(self).__name__}

    def judge(self, patient_summary: str, criterion_text: str,
              meta: dict[str, Any] | None = None) -> JudgmentResult:
        """Judge one criterion for one patient and log the call.

        Unparseable output becomes an unknown verdict with confidence 0.0 and a
        parse_error, so one malformed reply cannot abort a multi-hour run; the
        count of such rows is reported in the evaluation summary.
        """
        messages = prompts.build_messages(patient_summary, criterion_text)
        t0 = time.perf_counter()
        error: Optional[str] = None
        try:
            completion, attempts = with_retries(
                lambda: self._complete(prompts.SYSTEM_PROMPT, messages),
                max_attempts=self.max_attempts, base_delay=self.base_delay,
            )
        except LLMError as e:
            error = f"{type(e).__name__}: {e}"
            self._log(meta, None, time.perf_counter() - t0, self.max_attempts, error=error)
            raise
        latency = time.perf_counter() - t0

        try:
            judgment, repaired = parse_judgment(completion.text)
            result = JudgmentResult(
                **judgment.model_dump(), backend=self.role, model=self.model,
                latency_s=latency, input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens, attempts=attempts,
                repaired=repaired, raw_output=completion.text,
            )
        except JudgmentParseError as e:
            result = JudgmentResult(
                eligible=None, confidence=0.0, reasoning="unparseable model output",
                patient_feature="none", backend=self.role, model=self.model,
                latency_s=latency, input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens, attempts=attempts,
                parse_error=str(e), raw_output=completion.text,
            )
        self._log(meta, result, latency, attempts)
        return result

    def _log(self, meta: dict[str, Any] | None, result: Optional[JudgmentResult],
             latency: float, attempts: int, error: Optional[str] = None) -> None:
        """Write one call record; success and failure rows share a shape."""
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "backend": self.role,
            "model": self.model,
            "prompt_version": prompts.PROMPT_VERSION,
            "temperature": prompts.TEMPERATURE,
            "runtime": detect_runtime(),
            **(meta or {}),
            "latency_s": round(latency, 3),
            "attempts": attempts,
            "error": error,
        }
        if result is not None:
            record.update({
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "eligible": result.eligible,
                "confidence": result.confidence,
                "repaired": result.repaired,
                "parse_error": result.parse_error,
                "raw_output": result.raw_output,
            })
        self.logger.write(record)


def resolve_ollama_host(host: Optional[str] = None) -> str:
    """Pick the Ollama URL: explicit arg, then OLLAMA_HOST, then localhost:11434.

    OLLAMA_HOST is often set without a scheme ("127.0.0.1:11434"), so one is
    added; that keeps laptop and Colab on the same code path.
    """
    h = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).strip().rstrip("/")
    if not re.match(r"^https?://", h):
        h = "http://" + h
    return h


class OllamaBackend(LLMBackend):
    """Local phi4 via Ollama; checks server and model at construction so a run fails in seconds, not hours."""

    role = "local"

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, host: Optional[str] = None,
                 timeout_s: float = 300.0, keep_alive: str = "30m",
                 logger: JsonlLogger | None = None, check: bool = True, **kw: Any) -> None:
        """Connect to Ollama and verify the model is pulled (unless check=False)."""
        super().__init__(logger=logger, **kw)
        import ollama  # imported here so tests that never touch Ollama don't need it loaded

        self._ollama = ollama
        self.model = model
        self.host = resolve_ollama_host(host)
        self.keep_alive = keep_alive
        self.client = ollama.Client(host=self.host, timeout=timeout_s)
        if check:
            self._check_ready()

    def describe(self) -> dict[str, Any]:
        """Include the host and runtime so results record where phi4 actually ran."""
        return {**super().describe(), "host": self.host, "runtime": detect_runtime()}

    def _unavailable(self, detail: str) -> OllamaUnavailableError:
        """Build the actionable error shown when Ollama cannot be used."""
        return OllamaUnavailableError(
            f"Cannot use Ollama at {self.host} ({detail}).\n"
            f"  1. Install Ollama: https://ollama.com/download "
            f"(Linux/Colab: curl -fsSL https://ollama.com/install.sh | sh)\n"
            f"  2. Start the server: ollama serve\n"
            f"  3. Pull the model:   ollama pull {self.model}\n"
            f"  4. If it runs elsewhere, set OLLAMA_HOST (currently "
            f"{os.environ.get('OLLAMA_HOST', 'unset')}).\n"
            f"Not falling back to the API backend: local results are required for routing."
        )

    def _check_ready(self) -> None:
        """Fail loudly if the server is down or the model has not been pulled."""
        try:
            listing = self.client.list()
        except (ConnectionError, httpx.ConnectError, httpx.TimeoutException) as e:
            raise self._unavailable(f"server unreachable: {e}") from None
        names = {m.model for m in listing.models if m.model}
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            raise self._unavailable(
                f"model '{self.model}' not pulled; available: {sorted(names) or 'none'}")

    def _complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        """One non-streaming chat call constrained to the judgment JSON schema."""
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                format=JUDGMENT_JSON_SCHEMA,
                options={"temperature": prompts.TEMPERATURE, "seed": 0},
                keep_alive=self.keep_alive,
            )
        except (ConnectionError, httpx.ConnectError) as e:
            raise self._unavailable(f"connection lost mid-run: {e}") from None
        except httpx.TimeoutException as e:
            raise TransientLLMError(f"ollama timeout: {e}") from e
        except self._ollama.ResponseError as e:
            if e.status_code in (429, 500, 502, 503, 504):
                raise TransientLLMError(f"ollama {e.status_code}: {e.error}") from e
            raise LLMError(f"ollama {e.status_code}: {e.error}") from e
        return Completion(text=resp.message.content or "",
                          input_tokens=resp.prompt_eval_count,
                          output_tokens=resp.eval_count)


class ClaudeBackend(LLMBackend):
    """Claude Haiku 4.5 via the Anthropic API, for mid-confidence re-queries."""

    role = "api"

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, api_key: Optional[str] = None,
                 max_tokens: int = 512, logger: JsonlLogger | None = None, **kw: Any) -> None:
        """Create the client with SDK retries off, so with_retries is the single retry policy."""
        super().__init__(logger=logger, **kw)
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=60.0)

    @classmethod
    def from_env(cls, **kw: Any) -> Optional["ClaudeBackend"]:
        """Return a backend only if ANTHROPIC_API_KEY is set; routing records 'api_unavailable' otherwise."""
        return cls(**kw) if os.environ.get("ANTHROPIC_API_KEY") else None

    def _complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        """One Messages API call; maps SDK errors onto retryable vs fatal."""
        a = self._anthropic
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                temperature=prompts.TEMPERATURE, system=system, messages=messages,
            )
        except (a.RateLimitError, a.APIConnectionError) as e:   # includes APITimeoutError
            raise TransientLLMError(f"anthropic: {e}") from e
        except a.APIStatusError as e:
            if e.status_code >= 500:
                raise TransientLLMError(f"anthropic {e.status_code}: {e}") from e
            raise LLMError(f"anthropic {e.status_code}: {e}") from e
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(text=text, input_tokens=resp.usage.input_tokens,
                          output_tokens=resp.usage.output_tokens)
