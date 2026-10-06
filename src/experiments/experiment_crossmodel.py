"""
Cross-model validation experiment for RAGWitness Phase 3 Item 8.

Runs D1, D3, I1, I3, B1 × L1–L5 with a second LLM (llama3:latest) into
runs_model2/, then compares EC/AA/RF against the canonical model
(llama3.1:8b-instruct-q4_K_M) from runs/.

Produces: analysis/tables/cross_model_report.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.corpus_snapshot import build_corpus_metadata, sha256_tree
from src.embeddings import Embedder
from src.metrics import evaluate_run
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.retrieval import Retriever

CANONICAL_MODEL = "llama3.1:8b-instruct-q4_K_M"
SECOND_MODEL    = "llama3:latest"

ITEM8_ATTACK_IDS = ["D1", "D3", "I1", "I3", "B1"]


def _direct_catalog() -> dict[str, dict]:
    attacks = json.loads(
        Path("data/catalogs/direct_injection_variants.json").read_text()
    )
    return {a["attack_id"]: a for a in attacks}


def _indirect_catalog() -> dict[str, dict]:
    attacks = json.loads(
        Path("data/catalogs/indirect_injection_catalog.json").read_text()
    )
    return {a["attack_id"]: a for a in attacks}


def _baseline_catalog() -> dict[str, dict]:
    attacks = json.loads(
        Path("data/catalogs/baseline_queries.json").read_text()
    )
    # baseline_id → attack_id alias
    return {a.get("attack_id", a.get("baseline_id", "")): a for a in attacks}


def run_model2_missing(
    runs_root: str = "runs_model2",
    model: str = SECOND_MODEL,
    attack_ids: list[str] | None = None,
    levels: list[int] | None = None,
    corpus_version: str = "v1.0_200docs_15482chunks",
) -> list[Path]:
    """Run only the (attack_id, level) pairs not yet present in runs_root."""
    attack_ids = attack_ids or ITEM8_ATTACK_IDS
    levels = levels or [1, 2, 3, 4, 5]

    # Find what's already done
    existing: set[tuple[str, int]] = set()
    for mp in sorted(Path(runs_root).glob("*/metrics.json")):
        cfg = json.loads((mp.parent / "config.json").read_text())
        existing.add((cfg.get("attack_id", ""), int(cfg.get("observability_level", 0))))

    todo = [(a, l) for a in attack_ids for l in levels if (a, l) not in existing]
    if not todo:
        print("All model2 runs already present — nothing to run.")
        return []

    print(f"Missing {len(todo)} runs: {todo}")

    direct_cat  = _direct_catalog()
    indirect_cat = _indirect_catalog()
    baseline_cat = _baseline_catalog()

    corpus_metadata = build_corpus_metadata(corpus_version=corpus_version)
    embedder = Embedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True
    )

    run_dirs: list[Path] = []

    for aid, level in todo:
        print(f"  {aid} L{level} [{model}] …", end=" ", flush=True)

        if aid in direct_cat:
            attack = direct_cat[aid]
            retriever = Retriever(
                chroma_path="indexes/chroma_hansard",
                collection_name="hansard_chunks",
                embedder=embedder,
            )
            config = RAGPipelineConfig(
                query=attack["query"],
                observability_level=level,
                attack_id=aid,
                attack_type="direct",
                model=model,
                **corpus_metadata,
                runs_root=runs_root,
            )

        elif aid in indirect_cat:
            attack = indirect_cat[aid]
            chroma_path = f"indexes/chroma_poisoned_{aid}"
            collection  = f"hansard_poisoned_{aid}"
            retriever = Retriever(
                chroma_path=chroma_path,
                collection_name=collection,
                embedder=embedder,
            )
            # Probe poison retrieval
            chunks = retriever.retrieve(attack["target_query"], k=5)
            expected_id = f"poison:{aid}:0"
            poison_retrieved = any(c.chunk_id == expected_id for c in chunks)
            config = RAGPipelineConfig(
                query=attack["target_query"],
                observability_level=level,
                attack_id=aid,
                attack_type="indirect",
                model=model,
                expected_malicious_chunk_id=expected_id,
                poison_chunk_retrieved=poison_retrieved,
                chroma_path=chroma_path,
                collection=collection,
                **corpus_metadata,
                runs_root=runs_root,
            )
            config_dict = config.__dict__.copy()
            config_dict["effective_chroma_path"] = chroma_path
            config_dict["effective_chroma_snapshot_hash"] = sha256_tree(
                Path(chroma_path)
            )
            config = RAGPipelineConfig(**config_dict)

        elif aid in baseline_cat:
            attack = baseline_cat[aid]
            retriever = Retriever(
                chroma_path="indexes/chroma_hansard",
                collection_name="hansard_chunks",
                embedder=embedder,
            )
            config = RAGPipelineConfig(
                query=attack["query"],
                observability_level=level,
                attack_id=aid,
                attack_type="baseline",
                model=model,
                **corpus_metadata,
                runs_root=runs_root,
            )
        else:
            print(f"UNKNOWN attack_id {aid}, skipping")
            continue

        run_dir = RAGPipeline(config, retriever=retriever).run()
        evaluate_run(run_dir)
        run_dirs.append(run_dir)
        print(run_dir.name)

    return run_dirs


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyse_cross_model(
    canonical_runs: str | Path = "runs",
    model2_runs: str | Path = "runs_model2",
    output_path: str | Path = "analysis/tables/cross_model_report.json",
    attack_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compare EC/AA/RF between canonical model and llama3:latest."""
    attack_ids = attack_ids or ITEM8_ATTACK_IDS
    canonical_runs = Path(canonical_runs)
    model2_runs = Path(model2_runs)

    def _load(root: Path) -> dict[tuple[str, int], dict]:
        # Keep first occurrence of each (attack_id, level) pair
        rows: dict[tuple[str, int], dict] = {}
        for mp in sorted(root.glob("*/metrics.json")):
            cfg = json.loads((mp.parent / "config.json").read_text())
            aid = cfg.get("attack_id", "")
            lvl = int(cfg.get("observability_level", 0))
            if (aid, lvl) in rows:
                continue
            m = json.loads(mp.read_text())
            rows[(aid, lvl)] = {
                "ec": m["evidence_completeness"],
                "aa": m["attribution_accuracy"],
                "rf": m["reconstruction_fidelity"],
                "reconstructed": m.get("reconstructed_attack_type", "none"),
            }
        return rows

    can = _load(canonical_runs)
    m2  = _load(model2_runs)

    comparison: list[dict] = []
    deltas_ec, deltas_aa, deltas_rf = [], [], []

    for aid in attack_ids:
        for lvl in [1, 2, 3, 4, 5]:
            c = can.get((aid, lvl))
            m = m2.get((aid, lvl))
            if c is None or m is None:
                comparison.append({
                    "attack_id": aid, "level": lvl,
                    "status": "missing",
                    "canonical": c is not None, "model2": m is not None,
                })
                continue
            dec = round(m["ec"] - c["ec"], 4)
            daa = round(m["aa"] - c["aa"], 4)
            drf = round(m["rf"] - c["rf"], 4)
            deltas_ec.append(abs(dec))
            deltas_aa.append(abs(daa))
            deltas_rf.append(abs(drf))
            comparison.append({
                "attack_id": aid, "level": lvl,
                "status": "ok",
                "canonical_ec": c["ec"], "model2_ec": m["ec"], "delta_ec": dec,
                "canonical_aa": c["aa"], "model2_aa": m["aa"], "delta_aa": daa,
                "canonical_rf": c["rf"], "model2_rf": m["rf"], "delta_rf": drf,
                "canonical_reconstructed": c["reconstructed"],
                "model2_reconstructed":   m["reconstructed"],
            })

    ok_rows = [r for r in comparison if r["status"] == "ok"]
    n = len(ok_rows)

    report: dict[str, Any] = {
        "canonical_model": CANONICAL_MODEL,
        "second_model": SECOND_MODEL,
        "attack_ids_tested": attack_ids,
        "n_paired_runs": n,
        "mean_abs_delta_ec": round(sum(deltas_ec) / n, 4) if n else None,
        "mean_abs_delta_aa": round(sum(deltas_aa) / n, 4) if n else None,
        "mean_abs_delta_rf": round(sum(deltas_rf) / n, 4) if n else None,
        "max_abs_delta_ec":  round(max(deltas_ec), 4) if deltas_ec else None,
        "max_abs_delta_aa":  round(max(deltas_aa), 4) if deltas_aa else None,
        "max_abs_delta_rf":  round(max(deltas_rf), 4) if deltas_rf else None,
        "all_metrics_identical": all(
            r["delta_ec"] == 0 and r["delta_aa"] == 0 and r["delta_rf"] == 0
            for r in ok_rows
        ),
        "per_run": comparison,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_model2_missing()
    print("\nAnalysing cross-model metrics …")
    report = analyse_cross_model()
    print(f"  n_paired_runs     : {report['n_paired_runs']}")
    print(f"  mean |ΔEC|        : {report['mean_abs_delta_ec']}")
    print(f"  mean |ΔAA|        : {report['mean_abs_delta_aa']}")
    print(f"  mean |ΔRF|        : {report['mean_abs_delta_rf']}")
    print(f"  all_identical     : {report['all_metrics_identical']}")
    missing = [r for r in report['per_run'] if r['status'] == 'missing']
    if missing:
        print(f"  MISSING pairs     : {[(r['attack_id'], r['level']) for r in missing]}")
    print("  Saved → analysis/tables/cross_model_report.json")
