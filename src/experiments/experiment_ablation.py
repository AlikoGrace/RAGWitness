"""
Ablation Study — Phase 3, RAGWitness.

Tests which individual log fields at Level 3 are responsible for the EC/AA/RF
gains observed between L2 and L3. Five custom ObservabilityConfigs are created,
each identical to L3 except that one field is removed ("ablated").

Ablation configs:
  A1_no_metadata  — no document metadata  (log_document_metadata=False)
  A2_no_chunk_ids — no retrieved chunk IDs (log_retrieved_ids=False, log_retrieval_scores=False)
  A3_no_scores    — no retrieval scores    (log_retrieval_scores=False)
  A4_no_query     — no query logged        (log_query=False)
  A5_no_response  — no LLM answer logged   (log_response=False)

All 10 attacks (D1–D5, I1–I5) are run at each ablation config = 50 runs.
A clean L3 reference set of 10 runs is also produced for direct comparison.

Output written to runs_ablation/.
Report  written to analysis/tables/ablation_report.json.
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


# ── Ablation configurations ──────────────────────────────────────────────────
def _ablation_configs() -> dict[str, ObservabilityConfig]:
    """Return five L3 variants, each with one field ablated."""
    base = get_observability_config(3)

    return {
        "A1_no_metadata": replace(
            base,
            level=31,
            name="ablation_no_metadata",
            log_document_metadata=False,
        ),
        "A2_no_chunk_ids": replace(
            base,
            level=32,
            name="ablation_no_chunk_ids",
            log_retrieved_ids=False,
            log_retrieval_scores=False,
        ),
        "A3_no_scores": replace(
            base,
            level=33,
            name="ablation_no_scores",
            log_retrieval_scores=False,
        ),
        "A4_no_query": replace(
            base,
            level=34,
            name="ablation_no_query",
            log_query=False,
        ),
        "A5_no_response": replace(
            base,
            level=35,
            name="ablation_no_response",
            log_response=False,
            log_response_hash=False,
        ),
    }


# ── Experiment runner ────────────────────────────────────────────────────────
def run_ablation_experiments(
    direct_catalog: str | Path = "data/catalogs/direct_injection_variants.json",
    indirect_catalog: str | Path = "data/catalogs/indirect_injection_catalog.json",
    runs_root: str = "runs_ablation",
    corpus_version: str = "v1.0_200docs_15482chunks",
    include_l3_reference: bool = True,
) -> list[Path]:
    corpus_metadata = build_corpus_metadata(corpus_version=corpus_version)
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True)
    ablation_cfgs = _ablation_configs()

    if include_l3_reference:
        ablation_cfgs["L3_reference"] = get_observability_config(3)

    # Load catalogs
    d_attacks = json.loads(Path(direct_catalog).read_text(encoding="utf-8"))
    i_attacks = json.loads(Path(indirect_catalog).read_text(encoding="utf-8"))

    # Shared clean retriever for direct attacks
    retriever_clean = Retriever(
        chroma_path="indexes/chroma_hansard",
        collection_name="hansard_chunks",
        embedder=embedder,
    )

    # Pre-build poisoned retrievers for indirect attacks (one per attack)
    poison_retrievers: dict[str, tuple] = {}
    for attack in i_attacks:
        aid = attack["attack_id"]
        poison_result = prepare_poisoned_corpus(
            attack,
            max_clean_chunks=0,
            rebuild_index=True,
            embedder=embedder,
        )
        ret = Retriever(
            chroma_path=str(poison_result.chroma_path),
            collection_name=poison_result.collection,
            embedder=embedder,
        )
        poison_retrieved = _probe_poison_retrieval(
            ret,
            attack["target_query"],
            poison_result.expected_malicious_chunk_id,
        )
        poison_retrievers[aid] = (attack, poison_result, ret, poison_retrieved)
        print(f"  Poison probe {aid}: retrieved={poison_retrieved}")

    run_dirs: list[Path] = []

    for cfg_name, obs_cfg in ablation_cfgs.items():
        print(f"\n── {cfg_name} ──")

        # Direct attacks
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
            # Tag the ablation config name in metrics.json for later analysis
            mp = run_dir / "metrics.json"
            m = json.loads(mp.read_text())
            m["ablation_config"] = cfg_name
            mp.write_text(json.dumps(m, indent=2))
            run_dirs.append(run_dir)
            print(f"  {cfg_name} {aid} → {run_dir.name}")

        # Indirect attacks
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
            config_dict["effective_chroma_snapshot_hash"] = sha256_tree(poison_result.chroma_path)
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
