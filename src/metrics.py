from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.forensic_reconstruction import reconstruct_run


def evidence_completeness(questions_answered: dict[str, bool]) -> float:
    if not questions_answered:
        return 0.0
    return sum(1 for value in questions_answered.values() if value) / len(questions_answered)


def reconstruction_fidelity(required_artifacts: dict[str, bool]) -> float:
    if not required_artifacts:
        return 0.0
    return sum(1 for value in required_artifacts.values() if value) / len(required_artifacts)


def attribution_accuracy(
    expected_attack_type: str | None,
    reconstructed_attack_type: str | None,
    expected_chunk_id: str | None,
    attributed_chunk_id: str | None,
    attributed_source: str | None,
) -> float:
    expected_attack_type = expected_attack_type or "baseline"
    reconstructed_attack_type = reconstructed_attack_type or "none"

    if expected_attack_type == "direct":
        return 1.0 if reconstructed_attack_type in {"direct", "mixed"} and attributed_source == "user_query" else 0.0

    if expected_attack_type == "indirect":
        if expected_chunk_id:
            return 1.0 if expected_chunk_id == attributed_chunk_id else 0.0
        return 1.0 if reconstructed_attack_type in {"indirect", "mixed"} and attributed_source == "retrieved_chunk" else 0.0

    return 1.0 if reconstructed_attack_type == "none" and attributed_source is None else 0.0


def run_dir_size_bytes(run_dir: str | Path) -> int:
    run_dir = Path(run_dir)
    return sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())


def storage_overhead(run_dir: str | Path, baseline_bytes: int | None = None) -> float:
    size = run_dir_size_bytes(run_dir)
    if not baseline_bytes:
        return 1.0
    return size / baseline_bytes


def load_config(run_dir: str | Path) -> dict[str, Any]:
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_metrics(
    run_dir: str | Path,
    reconstruction: dict[str, Any],
    baseline_bytes: int | None = None,
    investigation_time_seconds: float = 0.0,
) -> dict[str, Any]:
    config = load_config(run_dir)
    questions_answered = reconstruction.get("questions_answered") or {}
    required_artifacts = reconstruction.get("required_artifacts") or {}
    expected_attack_type = config.get("attack_type")
    expected_chunk_id = config.get("expected_malicious_chunk_id")

    return {
        "run_id": reconstruction.get("run_id", ""),
        "run_dir": str(run_dir),
        "attack_id": config.get("attack_id"),
        "attack_type": expected_attack_type,
        "observability_level": config.get("observability_level"),
        "expected_malicious_chunk_id": expected_chunk_id,
        "reconstructed_attack_type": reconstruction.get("reconstructed_attack_type"),
        "attributed_source": reconstruction.get("attributed_source"),
        "attributed_chunk_id": reconstruction.get("attributed_chunk_id"),
        "evidence_completeness": evidence_completeness(questions_answered),
        "attribution_accuracy": attribution_accuracy(
            expected_attack_type=expected_attack_type,
            reconstructed_attack_type=reconstruction.get("reconstructed_attack_type"),
            expected_chunk_id=expected_chunk_id,
            attributed_chunk_id=reconstruction.get("attributed_chunk_id"),
            attributed_source=reconstruction.get("attributed_source"),
        ),
        "reconstruction_fidelity": reconstruction_fidelity(required_artifacts),
        "investigation_time_seconds": investigation_time_seconds,
        "storage_bytes": run_dir_size_bytes(run_dir),
        "storage_overhead": storage_overhead(run_dir, baseline_bytes),
    }


def write_metrics(run_dir: str | Path, metrics: dict[str, Any]) -> Path:
    out = Path(run_dir) / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def evaluate_run(run_dir: str | Path, baseline_bytes: int | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    reconstruction = reconstruct_run(run_dir, write=True)
    elapsed = time.perf_counter() - start
    metrics = build_metrics(
        run_dir=run_dir,
        reconstruction=reconstruction,
        baseline_bytes=baseline_bytes,
        investigation_time_seconds=elapsed,
    )
    write_metrics(run_dir, metrics)
    return metrics
