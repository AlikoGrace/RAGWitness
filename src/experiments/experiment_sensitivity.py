"""
Phase 4 §4.3 — Sensitivity analysis: top-k × embedding model.

Re-runs a fixed subset of attacks (D1, D3, I1, I3, B1) across all five
observability levels at each combination of:
  top_k  ∈ {3, 5, 10}
  embed  ∈ {all-MiniLM-L6-v2 (canonical), all-mpnet-base-v2 (probe)}

Key claim being validated: the L1→L2 AA jump is a consequence of the
logging configuration (chunk IDs appear at L2), not a function of how
many chunks are retrieved or which embedding model is used.

For the mpnet probe the clean index is rebuilt from the existing MiniLM
collection via chromadb .get() — no chunks.jsonl required.

Writes analysis/tables/sensitivity_report.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.corpus_snapshot import build_corpus_metadata
from src.embeddings import Embedder
from src.generation import StaticGenerator
from src.metrics import evaluate_run
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.retrieval import Retriever

ANALYSIS_DIR = Path("analysis/tables")
INDEXES_DIR = Path("indexes")
RUNS_ROOT = "runs_sensitivity"

# Attack subset chosen to cover both attack types and boundary case.
DIRECT_SUBSET = {"D1", "D3"}
INDIRECT_SUBSET = {"I1", "I3"}   # I2 excluded (retrieval-evasion boundary)
BASELINE_SUBSET = {"B1"}

TOP_K_VALUES = [3, 5, 10]

EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",
}

CANONICAL_CLEAN_INDEX = "indexes/chroma_hansard"
CANONICAL_COLLECTION = "hansard_chunks"


# ── Index helpers ─────────────────────────────────────────────────────────────

def _sensitivity_index_path(embed_key: str) -> Path:
    return INDEXES_DIR / f"sensitivity_{embed_key}_clean"


def _sensitivity_poisoned_path(embed_key: str, attack_id: str) -> Path:
    return INDEXES_DIR / f"sensitivity_{embed_key}_poisoned_{attack_id}"


def _build_clean_index(embed_key: str, embedder: Embedder) -> Path:
    """Export the canonical clean collection and re-embed with `embedder`."""
    import chromadb

    out_path = _sensitivity_index_path(embed_key)
    if out_path.exists():
        return out_path

    print(f"  Building clean index for {embed_key} …")
    src_client = chromadb.PersistentClient(path=CANONICAL_CLEAN_INDEX)
    src_col = src_client.get_collection(CANONICAL_COLLECTION)

    # Fetch in pages to avoid OOM on large collections
    PAGE = 2000
    offset = 0
    all_docs, all_ids, all_metas = [], [], []
    while True:
        page = src_col.get(limit=PAGE, offset=offset,
                           include=["documents", "metadatas"])
        batch_ids = page.get("ids") or []
        if not batch_ids:
            break
        all_ids.extend(batch_ids)
        all_docs.extend(page.get("documents") or [])
        all_metas.extend(page.get("metadatas") or [])
        offset += len(batch_ids)
        print(f"    exported {offset} / {src_col.count()} chunks …", end="\r")

    print(f"\n    re-embedding {len(all_docs)} chunks with {embed_key} …")
    vectors = embedder.embed_texts(all_docs, batch_size=64).tolist()

    dst_client = chromadb.PersistentClient(path=str(out_path))
    try:
        dst_client.delete_collection(CANONICAL_COLLECTION)
    except Exception:
        pass
    dst_col = dst_client.create_collection(CANONICAL_COLLECTION)

    UPSERT = 500
    for i in range(0, len(all_ids), UPSERT):
        dst_col.upsert(
            ids=all_ids[i:i + UPSERT],
            documents=all_docs[i:i + UPSERT],
            embeddings=vectors[i:i + UPSERT],
            metadatas=all_metas[i:i + UPSERT],
        )
    print(f"    clean index written → {out_path}")
    return out_path


def _build_poisoned_index(
    embed_key: str,
    attack: dict,
    embedder: Embedder,
    clean_index_path: Path,
) -> tuple[Path, str]:
    """
    Clone the clean index for `embed_key` and inject the poisoned chunk.
    Returns (chroma_path, poisoned_chunk_id).
    """
    import chromadb

    attack_id = attack["attack_id"]
    out_path = _sensitivity_poisoned_path(embed_key, attack_id)
    collection_name = f"hansard_poisoned_{attack_id}"
    poison_id = f"poison:{attack_id}:0"

    if out_path.exists():
        # Verify collection is actually present (guard against partial builds)
        try:
            _check = chromadb.PersistentClient(path=str(out_path))
            _check.get_collection(collection_name)
            return out_path, poison_id   # fully built
        except Exception:
            pass  # partial — fall through and rebuild

    print(f"  Building poisoned index {embed_key}/{attack_id} …")
    # Clone the clean index
    src_client = chromadb.PersistentClient(path=str(clean_index_path))
    src_col = src_client.get_collection(CANONICAL_COLLECTION)

    PAGE = 2000
    offset = 0
    all_docs, all_ids, all_metas, all_vecs = [], [], [], []
    while True:
        page = src_col.get(limit=PAGE, offset=offset,
                           include=["documents", "metadatas", "embeddings"])
        batch_ids = page.get("ids") or []
        if not batch_ids:
            break
        all_ids.extend(batch_ids)
        all_docs.extend(page.get("documents") or [])
        all_metas.extend(page.get("metadatas") or [])
        raw_vecs = page.get("embeddings")
        all_vecs.extend(raw_vecs if raw_vecs is not None else [])
        offset += len(batch_ids)

    # Add poisoned chunk (I3 uses poison_text_parts, others use poison_text)
    poison_text = (
        " ".join(attack["poison_text_parts"])
        if "poison_text_parts" in attack
        else attack["poison_text"]
    )
    poison_vec = embedder.embed_texts([poison_text], batch_size=1)[0].tolist()
    all_ids.append(poison_id)
    all_docs.append(poison_text)
    all_metas.append({"source_url": "http://attacker.example.com/poison",
                      "ingestion_stage": "adversarial"})
    all_vecs.append(poison_vec)

    # Normalise all vectors to plain Python lists (ChromaDB 1.4+ requires uniform type)
    all_vecs_list = [
        v.tolist() if hasattr(v, "tolist") else list(v)
        for v in all_vecs
    ]

    dst_client = chromadb.PersistentClient(path=str(out_path))
    try:
        dst_client.delete_collection(collection_name)
    except Exception:
        pass
    dst_col = dst_client.create_collection(collection_name)

    UPSERT = 500
    for i in range(0, len(all_ids), UPSERT):
        dst_col.upsert(
            ids=all_ids[i:i + UPSERT],
            documents=all_docs[i:i + UPSERT],
            embeddings=all_vecs_list[i:i + UPSERT],
            metadatas=all_metas[i:i + UPSERT],
        )
    print(f"    poisoned index written → {out_path}")
    return out_path, poison_id


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_sensitivity_experiments(
    mock_generation: bool = False,
    skip_index_rebuild: bool = False,
    corpus_version: str = "v1.0_200docs_15482chunks",
) -> dict[str, Any]:
    direct_catalog = json.loads(
        Path("data/catalogs/direct_injection_variants.json").read_text())
    indirect_catalog = json.loads(
        Path("data/catalogs/indirect_injection_catalog.json").read_text())
    baseline_catalog = json.loads(
        Path("data/catalogs/baseline_queries.json").read_text())

    direct_attacks  = [a for a in direct_catalog  if a["attack_id"] in DIRECT_SUBSET]
    indirect_attacks = [a for a in indirect_catalog if a["attack_id"] in INDIRECT_SUBSET]
    baselines       = [a for a in baseline_catalog
                       if a.get("attack_id", a.get("baseline_id")) in BASELINE_SUBSET]

    corpus_meta = build_corpus_metadata(corpus_version=corpus_version)
    results: list[dict] = []

    for embed_key, embed_model in EMBEDDING_MODELS.items():
        print(f"\n=== Embedding model: {embed_key} ({embed_model}) ===")
        embedder = Embedder(model_name=embed_model, normalize=True)

        # ── Build / locate indexes ──────────────────────────────────────────
        if embed_key == "minilm":
            clean_path = Path(CANONICAL_CLEAN_INDEX)
            clean_collection = CANONICAL_COLLECTION
        else:
            # _build_clean_index has its own exists-check; skip_index_rebuild
            # only suppresses the build when the index is already on disk.
            clean_path = (
                _sensitivity_index_path(embed_key)
                if (skip_index_rebuild and _sensitivity_index_path(embed_key).exists())
                else _build_clean_index(embed_key, embedder)
            )
            clean_collection = CANONICAL_COLLECTION

        # Pre-build poisoned indexes for indirect attacks.
        # _build_poisoned_index has its own exists-check so this is idempotent.
        # tuple: (chroma_path, collection_name, poison_chunk_id)
        poisoned_meta: dict[str, tuple[Path, str, str]] = {}
        for attack in indirect_attacks:
            if embed_key == "minilm":
                chroma_path = Path(f"indexes/chroma_poisoned_{attack['attack_id']}")
                coll = f"hansard_poisoned_{attack['attack_id']}"
                poisoned_meta[attack["attack_id"]] = (chroma_path, coll, f"poison:{attack['attack_id']}:0")
            else:
                p, pid = _build_poisoned_index(embed_key, attack, embedder, clean_path)
                poisoned_meta[attack["attack_id"]] = (p, f"hansard_poisoned_{attack['attack_id']}", pid)

        for k in TOP_K_VALUES:
            print(f"  top-k={k}")

            # ── Direct attacks ──────────────────────────────────────────────
            clean_retriever = Retriever(
                chroma_path=str(clean_path),
                collection_name=clean_collection,
                embedder=embedder,
            )
            for attack in direct_attacks:
                for level in [1, 2, 3, 4, 5]:
                    gen = StaticGenerator(f"MOCK_SENS_{attack['attack_id']}_L{level}_k{k}_{embed_key}") \
                          if mock_generation else None
                    cfg = RAGPipelineConfig(
                        query=attack["query"],
                        observability_level=level,
                        attack_id=attack["attack_id"],
                        attack_type="direct",
                        k=k,
                        **corpus_meta,
                        runs_root=f"{RUNS_ROOT}/{embed_key}/k{k}",
                    )
                    run_dir = RAGPipeline(cfg, retriever=clean_retriever, generator=gen).run()
                    metrics = evaluate_run(run_dir)
                    results.append(_row(metrics, embed_key, k, attack["attack_id"], level))

            # ── Indirect attacks ────────────────────────────────────────────
            for attack in indirect_attacks:
                poison_path, coll_name, poison_chunk_id = poisoned_meta[attack["attack_id"]]
                poison_retriever = Retriever(
                    chroma_path=str(poison_path),
                    collection_name=coll_name,
                    embedder=embedder,
                )
                chunks = poison_retriever.retrieve(attack["target_query"], k=k)
                poison_retrieved = any(c.chunk_id == poison_chunk_id for c in chunks)

                for level in [1, 2, 3, 4, 5]:
                    gen = StaticGenerator(f"MOCK_SENS_{attack['attack_id']}_L{level}_k{k}_{embed_key}") \
                          if mock_generation else None
                    cfg = RAGPipelineConfig(
                        query=attack["target_query"],
                        observability_level=level,
                        attack_id=attack["attack_id"],
                        attack_type="indirect",
                        expected_malicious_chunk_id=poison_chunk_id,
                        poison_chunk_retrieved=poison_retrieved,
                        k=k,
                        **corpus_meta,
                        runs_root=f"{RUNS_ROOT}/{embed_key}/k{k}",
                    )
                    run_dir = RAGPipeline(cfg, retriever=poison_retriever, generator=gen).run()
                    metrics = evaluate_run(run_dir)
                    results.append(_row(metrics, embed_key, k, attack["attack_id"], level))

            # ── Baselines ───────────────────────────────────────────────────
            for baseline in baselines:
                b_id = baseline.get("attack_id", baseline.get("baseline_id"))
                for level in [1, 2, 3, 4, 5]:
                    gen = StaticGenerator(f"MOCK_SENS_{b_id}_L{level}_k{k}_{embed_key}") \
                          if mock_generation else None
                    cfg = RAGPipelineConfig(
                        query=baseline["query"],
                        observability_level=level,
                        attack_id=b_id,
                        attack_type="baseline",
                        k=k,
                        **corpus_meta,
                        runs_root=f"{RUNS_ROOT}/{embed_key}/k{k}",
                    )
                    run_dir = RAGPipeline(cfg, retriever=clean_retriever, generator=gen).run()
                    metrics = evaluate_run(run_dir)
                    results.append(_row(metrics, embed_key, k, b_id, level))

    report = _summarise(results)
    out = ANALYSIS_DIR / "sensitivity_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSensitivity report written → {out}")
    return report


def _row(metrics: dict, embed_key: str, k: int,
         attack_id: str, level: int) -> dict:
    return {
        "embed_key": embed_key,
        "top_k": k,
        "attack_id": attack_id,
        "level": level,
        "AA": metrics.get("attribution_accuracy"),
        "EC": metrics.get("evidence_completeness"),
        "RF": metrics.get("reconstruction_fidelity"),
        "attack_type": metrics.get("attack_type"),
    }


def _summarise(rows: list[dict]) -> dict:
    """
    For each (embed_key, top_k): AA by level (attack runs only).
    Also compute the L1→L2 AA delta to confirm threshold persistence.
    """
    from collections import defaultdict

    cells: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["attack_type"] != "baseline" and r["AA"] is not None:
            cells[(r["embed_key"], r["top_k"], r["level"])].append(r["AA"])

    summary: list[dict] = []
    for (embed_key, k, level), vals in sorted(cells.items()):
        mean_aa = sum(vals) / len(vals)
        summary.append({
            "embed_key": embed_key,
            "top_k": k,
            "level": level,
            "n": len(vals),
            "mean_AA": round(mean_aa, 4),
        })

    # L1→L2 delta per (embed, k)
    deltas: list[dict] = []
    keyed = {(r["embed_key"], r["top_k"], r["level"]): r["mean_AA"]
             for r in summary}
    for embed_key in EMBEDDING_MODELS:
        for k in TOP_K_VALUES:
            l1 = keyed.get((embed_key, k, 1))
            l2 = keyed.get((embed_key, k, 2))
            if l1 is not None and l2 is not None:
                deltas.append({
                    "embed_key": embed_key,
                    "top_k": k,
                    "AA_L1": round(l1, 4),
                    "AA_L2": round(l2, 4),
                    "delta_L1_L2": round(l2 - l1, 4),
                })

    return {
        "by_embed_k_level": summary,
        "l1_l2_deltas": deltas,
        "raw_rows": rows,
    }
