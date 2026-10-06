from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.run_manager import RunManager
from src.threat_model import (
    ADVERSARY_DIRECT,
    ADVERSARY_INDIRECT,
    ADVERSARY_MIXED,
    FORENSIC_QUESTION_KEYS,
)

# Path to the canonical clean corpus.  Used for corpus-diff signal.
_CLEAN_CORPUS_JSONL = Path("data/processed/hansard/chunks.jsonl")
_CLEAN_SOURCE_DOMAIN = "parliament.gh"


@lru_cache(maxsize=1)
def _load_clean_chunk_ids() -> frozenset[str]:
    """
    Return the set of chunk IDs in the unmodified Hansard corpus.
    Cached after first load.  Returns an empty frozenset if the file
    is absent (e.g. unit tests without corpus data).
    """
    if not _CLEAN_CORPUS_JSONL.exists():
        return frozenset()
    ids: set[str] = set()
    for line in _CLEAN_CORPUS_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            ids.add(str(row.get("chunk_id", "")))
    return frozenset(ids)


SUSPICIOUS_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+override",
    r"reveal\s+the\s+system\s+prompt",
    r"attack\s+success",
    r"jailbreak",
    r"override\s+normal\s+evidence",
    r"do\s+not\s+use\s+retrieved\s+evidence",
    r"without\s+citations?",
    r"abolished\s+education\s+rights",
    r"cancelled\s+education",
    r"secretly\s+cancelled",
    r"hidden\s+unicode",
    r"zero\s+width",
]


def load_events(run_dir: str | Path) -> list[dict[str, Any]]:
    events_path = Path(run_dir) / "events.jsonl"
    events: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_config(run_dir: str | Path) -> dict[str, Any]:
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def write_reconstruction(run_dir: str | Path, result: dict[str, Any]) -> Path:
    out = Path(run_dir) / "reconstruction.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def reconstruct_run(run_dir: str | Path, write: bool = True) -> dict[str, Any]:
    """Reconstruct one run using the artifacts present in its run folder."""

    run_dir = Path(run_dir)
    events = load_events(run_dir)
    config = load_config(run_dir)
    event_by_name = _index_events(events)

    query_payload = _last_payload(event_by_name, "query.received")
    retrieval_payload = _last_payload(event_by_name, "retrieval.completed")
    prompt_payload = _last_payload(event_by_name, "prompt.built")
    generation_payload = _last_payload(event_by_name, "generation.completed")

    query = str(query_payload.get("query", ""))
    topk = retrieval_payload.get("topk") or []

    direct_detected = _looks_suspicious(query)
    suspicious_chunk = _find_suspicious_chunk(topk)
    indirect_detected = suspicious_chunk is not None

    if direct_detected and indirect_detected:
        reconstructed_attack_type = ADVERSARY_MIXED
    elif direct_detected:
        reconstructed_attack_type = ADVERSARY_DIRECT
    elif indirect_detected:
        reconstructed_attack_type = ADVERSARY_INDIRECT
    else:
        reconstructed_attack_type = "none"

    attributed_source = "user_query" if direct_detected else None
    attributed_chunk_id = None
    if suspicious_chunk is not None:
        attributed_chunk_id = suspicious_chunk.get("chunk_id")
        if not direct_detected:
            attributed_source = "retrieved_chunk"

    integrity_result = RunManager.verify_hash_chain(run_dir / "events.jsonl")
    has_generation_params = all(
        key in generation_payload for key in ("model", "temperature", "top_p")
    )
    has_full_prompt = "system_prompt" in prompt_payload and "user_prompt" in prompt_payload
    has_retrieval_sequence = bool(topk) and all("rank" in row and "chunk_id" in row for row in topk)
    has_retrieved_text = any("text" in row for row in topk)
    # Label-free provenance check: is any retrieved chunk foreign to the clean corpus?
    # Uses corpus-diff, source-domain, and ingestion-stage signals — NOT attacker flags.
    clean_ids = _load_clean_chunk_ids()
    has_foreign_chunk = any(
        _is_corpus_foreign(str(row.get("chunk_id") or ""), clean_ids)
        or _is_source_anomalous(row.get("metadata") or {})
        or _has_ingestion_stage(row.get("metadata") or {})
        for row in topk
    )
    can_assess_injection = (
        direct_detected
        or indirect_detected
        or has_retrieved_text
        or has_full_prompt
        or has_foreign_chunk
    )

    questions_answered = {
        "injection_present": can_assess_injection,
        "attack_type": can_assess_injection,
        "source_attribution": attributed_source is not None
        or (can_assess_injection and reconstructed_attack_type == "none"),
        "retrieval_sequence": has_retrieval_sequence,
        "prompt_reconstruction": has_full_prompt,
        "generation_replay": has_full_prompt and has_generation_params,
        "integrity_verification": bool(integrity_result.get("ok")),
    }
    # Enforce: question keys must stay aligned with the canonical threat model.
    assert set(questions_answered.keys()) == set(FORENSIC_QUESTION_KEYS), (
        f"questions_answered keys diverged from FORENSIC_QUESTION_KEYS. "
        f"Check src/threat_model.py and src/forensic_reconstruction.py."
    )

    required_artifacts = {
        "query": bool(query),
        "retrieval_sequence": has_retrieval_sequence,
        "retrieved_text_or_metadata": any(
            "text" in row or "metadata" in row for row in topk
        ),
        "full_prompt": has_full_prompt,
        "generation_params": has_generation_params,
        "answer": "answer" in generation_payload,
        "hash_chain": bool(integrity_result.get("ok")),
    }

    result = {
        "run_id": events[0]["run_id"] if events else config.get("run_id", ""),
        "run_dir": str(run_dir),
        "observability_level": config.get("observability_level"),
        "attack_id": config.get("attack_id"),
        "expected_attack_type": config.get("attack_type"),
        "reconstructed_attack_type": reconstructed_attack_type,
        "attributed_source": attributed_source,
        "attributed_chunk_id": attributed_chunk_id,
        "questions_answered": questions_answered,
        "required_artifacts": required_artifacts,
        "retrieved_chunk_ids": [row.get("chunk_id") for row in topk],
        "integrity": integrity_result,
        "notes": _build_notes(config, questions_answered),
    }

    if write:
        write_reconstruction(run_dir, result)
    return result


def _index_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        indexed.setdefault(event.get("event", ""), []).append(event)
    return indexed


def _last_payload(indexed: dict[str, list[dict[str, Any]]], event_name: str) -> dict[str, Any]:
    events = indexed.get(event_name) or []
    if not events:
        return {}
    payload = events[-1].get("payload") or {}
    return dict(payload)


def _looks_suspicious(text: str) -> bool:
    text = text or ""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SUSPICIOUS_PATTERNS)


def _is_corpus_foreign(chunk_id: str, clean_ids: frozenset[str]) -> bool:
    """True if chunk_id is absent from the clean corpus — i.e. was injected."""
    return bool(chunk_id) and bool(clean_ids) and chunk_id not in clean_ids


def _is_source_anomalous(metadata: dict[str, Any]) -> bool:
    """True if source_url does not belong to the official parliament.gh domain."""
    source = str(metadata.get("source_url") or "")
    return bool(source) and _CLEAN_SOURCE_DOMAIN not in source


def _has_ingestion_stage(metadata: dict[str, Any]) -> bool:
    """True if ingestion_stage is logged — a field absent from all clean corpus chunks."""
    return bool(metadata.get("ingestion_stage"))


def _find_suspicious_chunk(topk: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Identify a potentially injected chunk using label-free provenance signals only.

    Signals checked (no attacker-provided flags are read):
      1. text_content   — retrieved text matches injection patterns        (L4+)
      2. corpus_diff    — chunk_id absent from clean Hansard corpus         (L2+)
      3. source_anomaly — source_url outside parliament.gh                  (L3+)
      4. ingestion_stage— ingestion_stage field present in metadata         (L3+)
    """
    clean_ids = _load_clean_chunk_ids()
    for row in topk:
        metadata = row.get("metadata") or {}
        chunk_id = str(row.get("chunk_id") or "")
        text = str(row.get("text") or "")

        if (
            _looks_suspicious(text)
            or _is_corpus_foreign(chunk_id, clean_ids)
            or _is_source_anomalous(metadata)
            or _has_ingestion_stage(metadata)
        ):
            return row
    return None


def _build_notes(config: dict[str, Any], questions_answered: dict[str, bool]) -> list[str]:
    notes: list[str] = []
    level = config.get("observability_level")
    if level is not None and int(level) < 5 and not questions_answered["prompt_reconstruction"]:
        notes.append("Full prompt was not logged at this observability level.")
    if level is not None and int(level) < 4 and not questions_answered["source_attribution"]:
        notes.append("Retrieved text was not available for content-based attribution.")
    return notes
