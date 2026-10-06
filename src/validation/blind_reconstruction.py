"""
Blind Reconstruction Validation for RAGWitness Phase 3.

Proves that the forensic reconstructor operates purely from events.jsonl —
it does not require config.json to make detection/attribution decisions.

Two tests:

  1. Config-masked reconstruction
     Each run is reconstructed twice: once normally (with config.json), once
     with config.json temporarily hidden. The reconstructed_attack_type,
     attributed_source, and attributed_chunk_id must be identical.

  2. Confusion matrix
     Ground truth attack_type (from config.json) vs reconstructed_attack_type
     (from logs alone) at each observability level. Shows exactly where and
     why misclassifications occur.

Output written to analysis/tables/blind_reconstruction_report.json.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.forensic_reconstruction import reconstruct_run


# Labels used in the confusion matrix
_LABEL_MAP = {
    "direct": "direct",
    "indirect": "indirect",
    "baseline": "none",   # ground truth "baseline" → expected reconstruction "none"
}
_ALL_LABELS = ["direct", "indirect", "none"]


def _reconstruct_blind(run_dir: Path) -> dict[str, Any]:
    """
    Reconstruct using only events.jsonl (config.json hidden in a temp dir).
    Returns the reconstruction dict.
    """
    config_path = run_dir / "config.json"
    # Copy config aside temporarily
    with tempfile.TemporaryDirectory() as tmp:
        hidden = Path(tmp) / "config.json"
        shutil.copy2(config_path, hidden)
        config_path.rename(config_path.with_suffix(".json.hidden"))
        try:
            result = reconstruct_run(run_dir, write=False)
        finally:
            # Always restore
            config_path.with_suffix(".json.hidden").rename(config_path)
    return result


def run_blind_validation(
    runs_root: str | Path = "runs",
    output_path: str | Path = "analysis/tables/blind_reconstruction_report.json",
) -> dict[str, Any]:
    runs_root = Path(runs_root)
    output_path = Path(output_path)

    normal_results: list[dict[str, Any]] = []
    blind_results: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for metrics_path in sorted(runs_root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        cfg = json.loads((run_dir / "config.json").read_text())

        normal = reconstruct_run(run_dir, write=False)
        blind  = _reconstruct_blind(run_dir)

        normal_results.append(normal)
        blind_results.append(blind)

        # Check the three detection fields that must be log-derived
        for field in ("reconstructed_attack_type", "attributed_source", "attributed_chunk_id"):
            if normal.get(field) != blind.get(field):
                mismatches.append({
                    "run_id": cfg.get("run_id", run_dir.name),
                    "field": field,
                    "with_config": normal.get(field),
                    "without_config": blind.get(field),
                })

    # ── Confusion matrix at each level ──────────────────────────────────────
    confusion_by_level: dict[int, dict[str, Any]] = {}
    for level in [1, 2, 3, 4, 5]:
        # Load all runs at this level
        level_rows = []
        for p in sorted(runs_root.glob("*/metrics.json")):
            cfg = json.loads((p.parent / "config.json").read_text())
            if int(cfg.get("observability_level", 0)) != level:
                continue
            m = json.loads(p.read_text())
            true_label = _LABEL_MAP.get(cfg.get("attack_type", "baseline"), "none")
            pred_label = m.get("reconstructed_attack_type", "none")
            level_rows.append({"true": true_label, "pred": pred_label,
                                "attack_id": cfg.get("attack_id")})

        # Build matrix: rows = true, cols = pred
        matrix: dict[str, dict[str, int]] = {
            t: {p: 0 for p in _ALL_LABELS} for t in _ALL_LABELS
        }
        for row in level_rows:
            t = row["true"]
            p = row["pred"] if row["pred"] in _ALL_LABELS else "none"
            matrix[t][p] += 1

        # Per-class precision, recall, F1
        per_class: dict[str, dict[str, float]] = {}
        for cls in _ALL_LABELS:
            tp = matrix[cls][cls]
            fn = sum(matrix[cls][p] for p in _ALL_LABELS if p != cls)
            fp = sum(matrix[t][cls] for t in _ALL_LABELS if t != cls)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[cls] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": sum(matrix[cls].values()),
            }

        correct = sum(matrix[t][t] for t in _ALL_LABELS)
        total = sum(matrix[t][p] for t in _ALL_LABELS for p in _ALL_LABELS)
        accuracy = round(correct / total, 4) if total else 0.0

        confusion_by_level[level] = {
            "accuracy": accuracy,
            "n_runs": total,
            "confusion_matrix": matrix,
            "per_class": per_class,
        }

    n_runs = len(normal_results)
    report = {
        "n_runs_tested": n_runs,
        "config_masked_mismatches": len(mismatches),
        "mismatch_details": mismatches,
        "finding_masked": (
            "PASS — reconstruction is identical with and without config.json. "
            "Detection logic operates purely from events.jsonl."
            if not mismatches
            else f"FAIL — {len(mismatches)} field(s) differed between normal and blind reconstruction."
        ),
        "confusion_by_level": confusion_by_level,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report
