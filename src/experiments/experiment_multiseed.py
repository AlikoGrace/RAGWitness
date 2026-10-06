"""
Multi-seed replication experiment for RAGWitness Phase 1 Task 5.

Runs D1 (direct) and I1 (indirect) at all 5 observability levels × 3 seeds
to verify that per-run results are stable across LLM sampling variation.

  D1 × 5 levels × 3 seeds = 15 runs
  I1 × 5 levels × 3 seeds = 15 runs
  Total: 30 runs

Written to runs_seed/ (separate from the canonical 90-run matrix in runs/)
so Phase 2 statistics remain uncontaminated.

Usage:
    .venv/bin/python -c "
    from src.experiments.experiment_multiseed import run_multiseed_experiments
    run_multiseed_experiments()
    "
"""
from __future__ import annotations

import json
from pathlib import Path

from src.corpus_snapshot import build_corpus_metadata, sha256_tree
from src.embeddings import Embedder
from src.metrics import evaluate_run
from src.metrics.attack_success import score_run, _get_baseline_responses
from src.poisoning import prepare_poisoned_corpus
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.retrieval import Retriever

SEEDS = [42, 123, 777]
LEVELS = [1, 2, 3, 4, 5]
RUNS_ROOT = "runs_seed"


def _probe_poison_retrieval(
    retriever: Retriever,
    query: str,
    expected_chunk_id: str,
    k: int = 5,
) -> bool:
    chunks = retriever.retrieve(query, k=k)
    return any(c.chunk_id == expected_chunk_id for c in chunks)


def run_multiseed_experiments(
    direct_catalog: str | Path = "data/catalogs/direct_injection_variants.json",
    indirect_catalog: str | Path = "data/catalogs/indirect_injection_catalog.json",
    runs_root: str = RUNS_ROOT,
    corpus_version: str = "v1.0_200docs_15482chunks",
    seeds: list[int] | None = None,
    levels: list[int] | None = None,
) -> list[Path]:
    seeds = seeds or SEEDS
    levels = levels or LEVELS
    corpus_metadata = build_corpus_metadata(corpus_version=corpus_version)
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True)

    # Load B1 baseline responses from the canonical runs/ folder for semantic deviation
    baseline_responses = _get_baseline_responses("runs")

    run_dirs: list[Path] = []

    # ── D1 (direct) ─────────────────────────────────────────────────────────
    d_attacks = json.loads(Path(direct_catalog).read_text(encoding="utf-8"))
    d1 = next(a for a in d_attacks if a["attack_id"] == "D1")

    retriever_clean = Retriever(
        chroma_path="indexes/chroma_hansard",
        collection_name="hansard_chunks",
        embedder=embedder,
    )

    print(f"Running D1 × {len(levels)} levels × {len(seeds)} seeds …")
    for seed in seeds:
        for level in levels:
            config = RAGPipelineConfig(
                query=d1["query"],
                observability_level=level,
                attack_id="D1",
                attack_type="direct",
                seed=seed,
                **corpus_metadata,
                runs_root=runs_root,
            )
            run_dir = RAGPipeline(config, retriever=retriever_clean).run()
            evaluate_run(run_dir)
            # Append AS score into metrics.json
            as_result = score_run(run_dir, baseline_responses)
            metrics_path = run_dir / "metrics.json"
            existing = json.loads(metrics_path.read_text())
            existing.update(as_result)
            metrics_path.write_text(json.dumps(existing, indent=2))
            run_dirs.append(run_dir)
            print(f"  D1 L{level} seed={seed} → {run_dir.name}")

    # ── I1 (indirect) ───────────────────────────────────────────────────────
    i_attacks = json.loads(Path(indirect_catalog).read_text(encoding="utf-8"))
    i1 = next(a for a in i_attacks if a["attack_id"] == "I1")

    poison_result = prepare_poisoned_corpus(
        i1,
        max_clean_chunks=0,
        rebuild_index=True,
        embedder=embedder,
    )
    retriever_poison = Retriever(
        chroma_path=str(poison_result.chroma_path),
        collection_name=poison_result.collection,
        embedder=embedder,
    )
    poison_retrieved = _probe_poison_retrieval(
        retriever_poison,
        i1["target_query"],
        poison_result.expected_malicious_chunk_id,
    )
    print(f"\nI1 poison chunk retrieved: {poison_retrieved}")
    print(f"Running I1 × {len(levels)} levels × {len(seeds)} seeds …")

    for seed in seeds:
        for level in levels:
            config_dict = dict(
                query=i1["target_query"],
                observability_level=level,
                attack_id="I1",
                attack_type="indirect",
                seed=seed,
                poison_chunk_retrieved=poison_retrieved,
                expected_malicious_chunk_id=poison_result.expected_malicious_chunk_id,
                chroma_path=str(poison_result.chroma_path),
                collection=poison_result.collection,
                runs_root=runs_root,
                **corpus_metadata,
            )
            config_dict["effective_chroma_path"] = str(poison_result.chroma_path)
            config_dict["effective_chroma_snapshot_hash"] = sha256_tree(poison_result.chroma_path)
            config = RAGPipelineConfig(**config_dict)
            run_dir = RAGPipeline(config, retriever=retriever_poison).run()
            evaluate_run(run_dir)
            as_result = score_run(run_dir, baseline_responses)
            metrics_path = run_dir / "metrics.json"
            existing = json.loads(metrics_path.read_text())
            existing.update(as_result)
            metrics_path.write_text(json.dumps(existing, indent=2))
            run_dirs.append(run_dir)
            print(f"  I1 L{level} seed={seed} → {run_dir.name}")

    print(f"\nDone. {len(run_dirs)} runs written to {runs_root}/")
    return run_dirs
