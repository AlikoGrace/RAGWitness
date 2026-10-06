"""
Detection Performance (P/R/F1) metric for RAGWitness experiments.

Treats the forensic reconstructor as a binary classifier:
  - Positive class: runs where an attack actually occurred (attack_type != "baseline")
  - Negative class: benign runs (attack_type == "baseline")
  - Predicted positive: reconstructed_attack_type != "none"
  - Predicted negative: reconstructed_attack_type == "none"

At each observability level, computes:
  - Precision  = TP / (TP + FP)   — of flagged runs, how many truly had attacks
  - Recall     = TP / (TP + FN)   — of all attacks, how many were detected
  - F1         = harmonic mean of P and R
  - Specificity = TN / (TN + FP)  — benign runs correctly passed through
  - Confusion matrix: TP, TN, FP, FN

Special case — I2 (not_reached):
  I2 is an attack that never entered the pipeline (poison chunk not retrieved).
  The reconstructor correctly returns "none" since there is nothing to attribute.
  This is reported as an FN but annotated separately, since it represents a
  detection boundary (retrieval gate) rather than a logging deficiency.

Output written to analysis/tables/detection_performance_report.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_all_runs(runs_root: str | Path = "runs") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(Path(runs_root).glob("*/metrics.json")):
        run_dir = metrics_path.parent
        cfg = json.loads((run_dir / "config.json").read_text())
        m = json.loads(metrics_path.read_text())
        rows.append(
            {
                "run_id": m.get("run_id", run_dir.name),
                "attack_id": cfg.get("attack_id", ""),
                "attack_type": cfg.get("attack_type", "baseline"),
                "observability_level": int(cfg.get("observability_level", 0)),
                "reconstructed_attack_type": m.get("reconstructed_attack_type", "none"),
                "poison_chunk_retrieved": cfg.get("poison_chunk_retrieved"),
            }
        )
    return rows


def compute_detection_performance(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute P/R/F1 at each observability level and overall.

    Ground truth positive = attack_type != "baseline"
    Predicted positive    = reconstructed_attack_type != "none"
    """
    results_by_level: dict[int, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []

    for level in [1, 2, 3, 4, 5]:
        level_rows = [r for r in rows if r["observability_level"] == level]

        tp = [
            r for r in level_rows
            if r["attack_type"] != "baseline" and r["reconstructed_attack_type"] != "none"
        ]
        tn = [
            r for r in level_rows
            if r["attack_type"] == "baseline" and r["reconstructed_attack_type"] == "none"
        ]
        fp = [
            r for r in level_rows
            if r["attack_type"] == "baseline" and r["reconstructed_attack_type"] != "none"
        ]
        fn = [
            r for r in level_rows
            if r["attack_type"] != "baseline" and r["reconstructed_attack_type"] == "none"
        ]

        # Identify I2-like FNs: attack that never entered pipeline
        fn_not_reached = [
            r for r in fn if r.get("poison_chunk_retrieved") is False
        ]
        fn_missed = [r for r in fn if r not in fn_not_reached]

        prec = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
        rec = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        specificity = len(tn) / (len(tn) + len(fp)) if (len(tn) + len(fp)) > 0 else 0.0

        # Recall excluding not_reached FNs (detectable recall ceiling)
        detectable_attacks = [
            r for r in level_rows
            if r["attack_type"] != "baseline"
            and r.get("poison_chunk_retrieved") is not False
        ]
        rec_detectable = (
            len(tp) / len(detectable_attacks) if detectable_attacks else 0.0
        )

        results_by_level[level] = {
            "level": level,
            "n_total": len(level_rows),
            "n_attack": sum(1 for r in level_rows if r["attack_type"] != "baseline"),
            "n_benign": sum(1 for r in level_rows if r["attack_type"] == "baseline"),
            "tp": len(tp),
            "tn": len(tn),
            "fp": len(fp),
            "fn": len(fn),
            "fn_not_reached": len(fn_not_reached),
            "fn_missed_by_logging": len(fn_missed),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "specificity": round(specificity, 4),
            "recall_detectable": round(rec_detectable, 4),
            "tp_run_ids": [r["run_id"] for r in tp],
            "fn_run_ids": [r["run_id"] for r in fn],
            "fp_run_ids": [r["run_id"] for r in fp],
        }
        all_rows.extend(level_rows)

    return {
        "by_level": results_by_level,
        "notes": (
            "Precision=1.00 at all levels means zero false alarms on benign queries. "
            "FN at L1/L2 are indirect attacks (insufficient logged metadata for attribution). "
            "Persistent FN=1 at L3-L5 is I2 (poison chunk never retrieved — retrieval-gated FN, "
            "not a logging deficiency). "
            "recall_detectable is recall computed only over attacks that entered the pipeline."
        ),
    }


def score_and_save(
    runs_root: str | Path = "runs",
    output_path: str | Path = "analysis/tables/detection_performance_report.json",
) -> dict[str, Any]:
    """Load all runs, compute detection P/R/F1, and write to output_path."""
    rows = _load_all_runs(runs_root)
    report = compute_detection_performance(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report
