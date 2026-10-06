"""
Ablation Study Analysis for RAGWitness Phase 3.

Reads completed ablation runs from runs_ablation/ and produces a report
showing how EC, AA, and RF change when individual log fields are removed
from the Level 3 configuration.

The L3_reference config (standard L3) serves as the baseline. Each ablation
config removes one field and the delta shows which fields are critical.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def analyse_ablation(
    runs_root: str | Path = "runs_ablation",
    output_path: str | Path = "analysis/tables/ablation_report.json",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for p in sorted(Path(runs_root).glob("*/metrics.json")):
        cfg = json.loads((p.parent / "config.json").read_text())
        m = json.loads(p.read_text())
        abl = m.get("ablation_config", "unknown")
        rows.append({
            "ablation_config": abl,
            "attack_id": cfg.get("attack_id"),
            "attack_type": cfg.get("attack_type"),
            "level": int(cfg.get("observability_level", 0)),
            "ec": m["evidence_completeness"],
            "aa": m["attribution_accuracy"],
            "rf": m["reconstruction_fidelity"],
        })

    if not rows:
        return {"error": "no ablation runs found", "n": 0}

    # Aggregate mean EC/AA/RF per ablation config
    configs = sorted(set(r["ablation_config"] for r in rows))
    summary: dict[str, dict[str, Any]] = {}
    for cfg_name in configs:
        subset = [r for r in rows if r["ablation_config"] == cfg_name]
        n = len(subset)
        ec_mean = sum(r["ec"] for r in subset) / n
        aa_mean = sum(r["aa"] for r in subset) / n
        rf_mean = sum(r["rf"] for r in subset) / n

        # Split by attack type
        direct_rows = [r for r in subset if r["attack_type"] == "direct"]
        indirect_rows = [r for r in subset if r["attack_type"] == "indirect"]

        summary[cfg_name] = {
            "n": n,
            "ec_mean": round(ec_mean, 4),
            "aa_mean": round(aa_mean, 4),
            "rf_mean": round(rf_mean, 4),
            "direct": {
                "n": len(direct_rows),
                "ec": round(sum(r["ec"] for r in direct_rows) / len(direct_rows), 4) if direct_rows else None,
                "aa": round(sum(r["aa"] for r in direct_rows) / len(direct_rows), 4) if direct_rows else None,
                "rf": round(sum(r["rf"] for r in direct_rows) / len(direct_rows), 4) if direct_rows else None,
            },
            "indirect": {
                "n": len(indirect_rows),
                "ec": round(sum(r["ec"] for r in indirect_rows) / len(indirect_rows), 4) if indirect_rows else None,
                "aa": round(sum(r["aa"] for r in indirect_rows) / len(indirect_rows), 4) if indirect_rows else None,
                "rf": round(sum(r["rf"] for r in indirect_rows) / len(indirect_rows), 4) if indirect_rows else None,
            },
        }

    # Compute deltas vs L3_reference
    ref = summary.get("L3_reference", {})
    deltas: dict[str, dict[str, float]] = {}
    for cfg_name, stats in summary.items():
        if cfg_name == "L3_reference":
            continue
        deltas[cfg_name] = {
            "delta_ec": round(stats["ec_mean"] - ref.get("ec_mean", 0), 4),
            "delta_aa": round(stats["aa_mean"] - ref.get("aa_mean", 0), 4),
            "delta_rf": round(stats["rf_mean"] - ref.get("rf_mean", 0), 4),
        }

    ablation_descriptions = {
        "A1_no_metadata": "L3 without document metadata (log_document_metadata=False)",
        "A2_no_chunk_ids": "L3 without chunk IDs or scores (log_retrieved_ids=False)",
        "A3_no_scores": "L3 without retrieval scores (log_retrieval_scores=False)",
        "A4_no_query": "L3 without query logging (log_query=False)",
        "A5_no_response": "L3 without LLM response (log_response=False)",
        "L3_reference": "Standard L3 (all fields enabled)",
    }

    report = {
        "n_total_runs": len(rows),
        "configs_tested": configs,
        "summary_by_config": summary,
        "deltas_vs_l3_reference": deltas,
        "config_descriptions": ablation_descriptions,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2))
    return report
