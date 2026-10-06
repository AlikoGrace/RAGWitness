"""
Ablation Study at Level 5 for RAGWitness Phase 3 Item 10.

Tests which individual log fields at Level 5 (forensic) are responsible for
forensic capability gains beyond L4. Five custom ObservabilityConfigs are
created, each identical to L5 except that one field is ablated.

Ablation configs (L5-based):
  A6_no_full_prompt      — log_full_prompt=False        (field added at L5)
  A7_no_retrieved_text   — log_retrieved_text=False      (field added at L4)
  A8_no_document_hashes  — log_document_hashes=False     (field added at L4)
  A9_no_generation_params— log_generation_params=False   (field added at L5)
  A10_no_integrity       — log_integrity_details=False   (field added at L5)

All 10 attacks (D1–D5, I1–I5) at each config + L5 reference = 60 runs.
Output: runs_ablation_l5/
Report: analysis/tables/ablation_l5_report.json
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from src.corpus_snapshot import build_corpus_metadata, sha256_tree
from src.embeddings import Embedder
from src.experiments.experiment_indirect import _probe_poison_retrieval
from src.metrics import evaluate_run
from src.observability_config import ObservabilityConfig, get_observability_config
from src.poisoning import prepare_poisoned_corpus
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.retrieval import Retriever


def _ablation_configs_l5() -> dict[str, ObservabilityConfig]:
    """Return five L5 variants, each with one field ablated."""
    base = get_observability_config(5)

    return {
        "A6_no_full_prompt": replace(
            base, level=56, name="ablation_l5_no_full_prompt",
            log_full_prompt=False,
        ),
        "A7_no_retrieved_text": replace(
            base, level=57, name="ablation_l5_no_retrieved_text",
            log_retrieved_text=False,
        ),
        "A8_no_document_hashes": replace(
            base, level=58, name="ablation_l5_no_document_hashes",
            log_document_hashes=False,
        ),
        "A9_no_generation_params": replace(
            base, level=59, name="ablation_l5_no_generation_params",
            log_generation_params=False,
        ),
        "A10_no_integrity": replace(
            base, level=60, name="ablation_l5_no_integrity",
            log_integrity_details=False,
        ),
    }


def run_ablation_l5_experiments(
    direct_catalog: str | Path = "data/catalogs/direct_injection_variants.json",
    indirect_catalog: str | Path = "data/catalogs/indirect_injection_catalog.json",
    runs_root: str = "runs_ablation_l5",
    corpus_version: str = "v1.0_200docs_15482chunks",
) -> list[Path]:

    corpus_metadata = build_corpus_metadata(corpus_version=corpus_version)
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True)

    ablation_cfgs = _ablation_configs_l5()
    ablation_cfgs["L5_reference"] = get_observability_config(5)

    d_attacks = json.loads(Path(direct_catalog).read_text(encoding="utf-8"))
    i_attacks = json.loads(Path(indirect_catalog).read_text(encoding="utf-8"))

    retriever_clean = Retriever(
        chroma_path="indexes/chroma_hansard",
        collection_name="hansard_chunks",
        embedder=embedder,
    )

    # Pre-build poisoned retrievers (reuse existing indexes — no rebuild needed)
    poison_retrievers: dict[str, tuple] = {}
    for attack in i_attacks:
        aid = attack["attack_id"]
        poison_result = prepare_poisoned_corpus(
            attack, max_clean_chunks=0, rebuild_index=False, embedder=embedder,
        )
        ret = Retriever(
            chroma_path=str(poison_result.chroma_path),
            collection_name=poison_result.collection,
            embedder=embedder,
        )
        poison_retrieved = _probe_poison_retrieval(
            ret, attack["target_query"],
            poison_result.expected_malicious_chunk_id,
        )
        poison_retrievers[aid] = (attack, poison_result, ret, poison_retrieved)
        print(f"  Poison probe {aid}: retrieved={poison_retrieved}")

    run_dirs: list[Path] = []

    for cfg_name, obs_cfg in ablation_cfgs.items():
        print(f"\n── {cfg_name} ──")

        for attack in d_attacks:
            aid = attack["attack_id"]
            config = RAGPipelineConfig(
                query=attack["query"],
                observability_level=obs_cfg.level,
                attack_id=aid,
                attack_type="direct",
                **corpus_metadata,
                runs_root=runs_root,
            )
            run_dir = RAGPipeline(
                config, retriever=retriever_clean, observability=obs_cfg
            ).run()
            evaluate_run(run_dir)
            mp = run_dir / "metrics.json"
            m = json.loads(mp.read_text())
            m["ablation_config"] = cfg_name
            mp.write_text(json.dumps(m, indent=2))
            run_dirs.append(run_dir)
            print(f"  {cfg_name} {aid} → {run_dir.name}")

        for aid, (attack, poison_result, ret, poison_retrieved) in poison_retrievers.items():
            config_dict = dict(
                query=attack["target_query"],
                observability_level=obs_cfg.level,
                attack_id=aid,
                attack_type="indirect",
                poison_chunk_retrieved=poison_retrieved,
                expected_malicious_chunk_id=poison_result.expected_malicious_chunk_id,
                chroma_path=str(poison_result.chroma_path),
                collection=poison_result.collection,
                runs_root=runs_root,
                **corpus_metadata,
            )
            config_dict["effective_chroma_path"] = str(poison_result.chroma_path)
            config_dict["effective_chroma_snapshot_hash"] = sha256_tree(
                poison_result.chroma_path
            )
            config = RAGPipelineConfig(**config_dict)
            run_dir = RAGPipeline(
                config, retriever=ret, observability=obs_cfg
            ).run()
            evaluate_run(run_dir)
            mp = run_dir / "metrics.json"
            m = json.loads(mp.read_text())
            m["ablation_config"] = cfg_name
            mp.write_text(json.dumps(m, indent=2))
            run_dirs.append(run_dir)
            print(f"  {cfg_name} {aid} → {run_dir.name}")

    print(f"\nDone. {len(run_dirs)} ablation runs in {runs_root}/")
    return run_dirs


def analyse_ablation_l5(
    runs_root: str = "runs_ablation_l5",
    output_path: str = "analysis/tables/ablation_l5_report.json",
) -> dict:
    """Analyse L5 ablation runs, compute deltas vs L5_reference."""
    rows = []
    for p in sorted(Path(runs_root).glob("*/metrics.json")):
        cfg = json.loads((p.parent / "config.json").read_text())
        m = json.loads(p.read_text())
        rows.append({
            "ablation_config": m.get("ablation_config", "unknown"),
            "attack_id": cfg.get("attack_id"),
            "attack_type": cfg.get("attack_type"),
            "ec": m["evidence_completeness"],
            "aa": m["attribution_accuracy"],
            "rf": m["reconstruction_fidelity"],
        })

    configs = sorted(set(r["ablation_config"] for r in rows))
    summary = {}
    for cfg_name in configs:
        subset = [r for r in rows if r["ablation_config"] == cfg_name]
        n = len(subset)
        direct   = [r for r in subset if r["attack_type"] == "direct"]
        indirect = [r for r in subset if r["attack_type"] == "indirect"]
        summary[cfg_name] = {
            "n": n,
            "ec_mean": round(sum(r["ec"] for r in subset) / n, 4),
            "aa_mean": round(sum(r["aa"] for r in subset) / n, 4),
            "rf_mean": round(sum(r["rf"] for r in subset) / n, 4),
            "direct_aa":   round(sum(r["aa"] for r in direct)   / len(direct),   4) if direct   else None,
            "indirect_aa": round(sum(r["aa"] for r in indirect) / len(indirect), 4) if indirect else None,
        }

    ref = summary.get("L5_reference", {})
    deltas = {}
    for cfg_name, stats in summary.items():
        if cfg_name == "L5_reference":
            continue
        deltas[cfg_name] = {
            "delta_ec": round(stats["ec_mean"] - ref.get("ec_mean", 0), 4),
            "delta_aa": round(stats["aa_mean"] - ref.get("aa_mean", 0), 4),
            "delta_rf": round(stats["rf_mean"] - ref.get("rf_mean", 0), 4),
        }

    config_descriptions = {
        "A6_no_full_prompt":       "L5 without system prompt logging (log_full_prompt=False)",
        "A7_no_retrieved_text":    "L5 without chunk text logging (log_retrieved_text=False)",
        "A8_no_document_hashes":   "L5 without document SHA-256 hashes (log_document_hashes=False)",
        "A9_no_generation_params": "L5 without model/temperature params (log_generation_params=False)",
        "A10_no_integrity":        "L5 without hash chain / integrity (log_integrity_details=False)",
        "L5_reference":            "Standard L5 (all fields enabled)",
    }

    report = {
        "reference_level": 5,
        "n_total_runs": len(rows),
        "configs_tested": configs,
        "summary_by_config": summary,
        "deltas_vs_l5_reference": deltas,
        "config_descriptions": config_descriptions,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_ablation_l5_experiments()
    report = analyse_ablation_l5()

    ref = report["summary_by_config"].get("L5_reference", {})
    print(f"\nL5 reference: EC={ref['ec_mean']}  AA={ref['aa_mean']}  RF={ref['rf_mean']}")
    print(f"\n{'Config':<28} {'ΔEC':>7} {'ΔAA':>7} {'ΔRF':>7}  Description")
    print("-" * 85)
    descs = {
        "A6_no_full_prompt":       "remove system prompt",
        "A7_no_retrieved_text":    "remove chunk text",
        "A8_no_document_hashes":   "remove doc SHA-256 hashes",
        "A9_no_generation_params": "remove model/temp params",
        "A10_no_integrity":        "remove hash chain",
    }
    for cfg, delta in sorted(report["deltas_vs_l5_reference"].items()):
        print(
            f"{cfg:<28} {delta['delta_ec']:>+7.4f} {delta['delta_aa']:>+7.4f}"
            f" {delta['delta_rf']:>+7.4f}  {descs.get(cfg, '')}"
        )
    print(f"\nSaved → analysis/tables/ablation_l5_report.json")
