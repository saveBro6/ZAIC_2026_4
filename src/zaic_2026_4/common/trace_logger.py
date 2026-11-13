"""Trace logging utilities for capturing LLM calls, prompts, reasoning, and stage metrics."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def build_call_id() -> str:
    """Generate a short unique identifier for an LLM call."""
    return uuid.uuid4().hex[:12]


def extract_thinking_trace(raw_text: str) -> str:
    """Extract content inside <think>...</think> blocks, handling unclosed truncated tags."""
    if not raw_text:
        return ""
    matches = _THINK_TAG_RE.findall(raw_text)
    if matches:
        return "\n".join(m.strip() for m in matches if m.strip())
    lower = raw_text.lower()
    open_idx = lower.find("<think>")
    if open_idx != -1 and "</think>" not in lower[open_idx:]:
        return raw_text[open_idx + len("<think>") :].strip()
    return ""


def sanitize_sampling_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Convert sampling parameters into a JSON-serializable dictionary."""
    if not params:
        return {}
    clean: dict[str, Any] = {}
    for key, val in params.items():
        if val is None or isinstance(val, (str, int, float, bool)):
            clean[key] = val
        elif isinstance(val, (list, tuple)):
            clean[key] = [
                item if isinstance(item, (str, int, float, bool, type(None))) else str(item)
                for item in val
            ]
        elif isinstance(val, dict):
            clean[key] = sanitize_sampling_params(val)
        elif key == "structured_outputs":
            clean["has_structured_outputs"] = True
            clean["structured_outputs_type"] = type(val).__name__
        else:
            clean[key] = str(val)
    return clean


def extract_llm_engine_call_traces(llm_engine: Any) -> list[dict[str, Any]]:
    """Safely extract per-prompt metadata captured during the latest llm_engine.generate() call."""
    traces = getattr(llm_engine, "last_call_traces", None)
    if not isinstance(traces, list):
        return []
    return [dict(item) if isinstance(item, dict) else {} for item in traces]


@dataclass
class LLMCallRecord:
    """Structured record of a single LLM prompt-completion invocation."""

    stage: str
    prompt: str
    raw_output: str
    sampling_params: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    stop_reason: Any | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=build_call_id)
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to a JSON-compatible dictionary."""
        return asdict(self)


def make_llm_call_record(
    stage: str,
    prompt: str,
    raw_output: str,
    engine_trace: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> LLMCallRecord:
    """Build an LLMCallRecord combining caller context with engine-level trace metrics."""
    trace = engine_trace or {}
    return LLMCallRecord(
        stage=stage,
        prompt=prompt,
        raw_output=raw_output,
        sampling_params=dict(trace.get("sampling_params") or {}),
        prompt_tokens=trace.get("prompt_tokens"),
        completion_tokens=trace.get("completion_tokens"),
        finish_reason=trace.get("finish_reason"),
        stop_reason=trace.get("stop_reason"),
        latency_ms=trace.get("latency_ms"),
        metadata=dict(metadata or {}),
    )


def append_llm_calls(
    cache_dir: Path | str | None,
    records: Sequence[LLMCallRecord | dict[str, Any]],
    filename: str = "llm_calls.jsonl",
) -> None:
    """Append LLM call records to cache_dir/llm_calls.jsonl without interrupting pipeline execution."""
    if cache_dir is None or not records:
        return
    try:
        c_dir = Path(cache_dir)
        c_dir.mkdir(parents=True, exist_ok=True)
        out_path = c_dir / filename
        with open(out_path, "a", encoding="utf-8") as f:
            for rec in records:
                payload = rec.to_dict() if isinstance(rec, LLMCallRecord) else rec
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to append LLM call traces to %s: %s", cache_dir, exc)


def write_json_trace(path: Path | str, payload: dict[str, Any]) -> None:
    """Persist a structured JSON trace file safely."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to write JSON trace %s: %s", path, exc)


__all__ = [
    "LLMCallRecord",
    "append_llm_calls",
    "build_call_id",
    "extract_llm_engine_call_traces",
    "extract_thinking_trace",
    "make_llm_call_record",
    "sanitize_sampling_params",
    "utc_now_iso",
    "write_json_trace",
]
