"""
Phase 4 §3.2 part 2 — L4 content-hash query-time verification.

Demonstrates that content hashes logged at L4 detect in-place corpus
modification that the hash chain alone cannot catch.

Attack scenario (M1 — in-place modification):
  1. A normal RAG run at L4 logs text_sha256 for every retrieved chunk.
  2. An adversary with ChromaDB access updates one chunk's text in-place.
     The hash chain is untouched; the events.jsonl is not modified.
  3. At investigation time, verify_content_hashes() re-fetches the chunk
     from the live index and compares to the logged hash.
  4. The mismatch is detected and flagged as MODIFIED.

The experiment also runs M0 (no modification) as a control — all hashes
should match.

Writes analysis/tables/content_hash_report.json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from src.corpus_snapshot import build_corpus_metadata
from src.embeddings import Embedder
from src.generation import StaticGenerator
from src.metrics import evaluate_run
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.retrieval import Retriever

ANALYSIS_DIR = Path("analysis/tables")
RUNS_ROOT = "runs_content_hash"

# We run at L4 only — content hashes first appear there.
CONTENT_HASH_LEVEL = 4

# Use an existing indirect-attack query so we know exactly which chunk
# is retrieved — makes it easy to pick one to tamper with.
_TARGET_QUERY = "What did Parliament say about education policy?"
_CANONICAL_CLEAN_INDEX = "indexes/chroma_hansard"
_CANONICAL_COLLECTION = "hansard_chunks"

# Text injected into the modified chunk to simulate adversarial edit.
_TAMPERED_SUFFIX = " [MODIFIED BY ADVERSARY POST-HOC]"


# ── Content-hash verifier ─────────────────────────────────────────────────────

def verify_content_hashes(
    run_dir: str | Path,
    chroma_path: str,
    collection_name: str,
) -> dict[str, Any]:
    """
    Re-fetch retrieved chunks from ChromaDB and compare to logged hashes.

    Returns a dict with:
      verified    — True iff ALL hashes match
      mismatches  — list of {chunk_id, logged_hash, recomputed_hash}
      checked     — number of chunks with logged text_sha256
    """
    import chromadb

    run_dir = Path(run_dir)
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return {"verified": False, "error": "events.jsonl not found"}

    # Parse logged hashes from retrieval.completed event
    logged: dict[str, str] = {}  # chunk_id → text_sha256
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("event") == "retrieval.completed":
            for row in ev.get("payload", {}).get("topk", []):
                cid = str(row.get("chunk_id") or "")
                h = row.get("text_sha256")
                if cid and h:
                    logged[cid] = h

    if not logged:
        return {"verified": True, "checked": 0, "mismatches": [],
                "note": "No text_sha256 logged (level < 4?)"}

    # Re-fetch chunk texts from live index
    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_collection(collection_name)

    mismatches: list[dict] = []
    for chunk_id, logged_hash in logged.items():
        result = col.get(ids=[chunk_id], include=["documents"])
        docs = (result.get("documents") or [[]])[0] if result.get("documents") else None
        if not docs:
            mismatches.append({
                "chunk_id": chunk_id,
                "logged_hash": logged_hash,
                "recomputed_hash": None,
                "status": "CHUNK_MISSING",
            })
            continue
        live_text = docs if isinstance(docs, str) else docs[0]
        live_hash = hashlib.sha256(live_text.encode("utf-8")).hexdigest()
        if live_hash != logged_hash:
            mismatches.append({
                "chunk_id": chunk_id,
                "logged_hash": logged_hash,
                "recomputed_hash": live_hash,
                "status": "MODIFIED",
            })

    return {
        "verified": len(mismatches) == 0,
        "checked": len(logged),
        "mismatches": mismatches,
    }


# ── Experiment ────────────────────────────────────────────────────────────────

def run_content_hash_experiment(
    mock_generation: bool = False,
    corpus_version: str = "v1.0_200docs_15482chunks",
) -> dict[str, Any]:
    corpus_meta = build_corpus_metadata(corpus_version=corpus_version)
    embedder = Embedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True)

    # ── Scenario M0: clean run, no tampering ──────────────────────────────
    print("Running M0 (no tampering) …")
    clean_retriever = Retriever(
        chroma_path=_CANONICAL_CLEAN_INDEX,
        collection_name=_CANONICAL_COLLECTION,
        embedder=embedder,
    )
    gen = StaticGenerator("MOCK_CLEAN") if mock_generation else None
    cfg_m0 = RAGPipelineConfig(
        query=_TARGET_QUERY,
        observability_level=CONTENT_HASH_LEVEL,
        attack_id="M0",
        attack_type="baseline",
        k=5,
        **corpus_meta,
        runs_root=RUNS_ROOT,
    )
    run_dir_m0 = RAGPipeline(cfg_m0, retriever=clean_retriever, generator=gen).run()
    evaluate_run(run_dir_m0)

    result_m0 = verify_content_hashes(
        run_dir_m0, _CANONICAL_CLEAN_INDEX, _CANONICAL_COLLECTION)
    print(f"  M0 verified={result_m0['verified']}  "
          f"checked={result_m0['checked']}  "
          f"mismatches={len(result_m0['mismatches'])}")

    # ── Scenario M1: tamper one chunk, then re-verify ─────────────────────
    print("Running M1 (in-place modification) …")

    # Copy the clean index so we don't permanently corrupt it
    tampered_index = Path("indexes/content_hash_tampered")
    if tampered_index.exists():
        shutil.rmtree(tampered_index)
    shutil.copytree(_CANONICAL_CLEAN_INDEX, str(tampered_index))

    # Run the RAG pipeline BEFORE tampering (hashes are logged now)
    tampered_retriever = Retriever(
        chroma_path=str(tampered_index),
        collection_name=_CANONICAL_COLLECTION,
        embedder=embedder,
    )
    gen = StaticGenerator("MOCK_TAMPERED") if mock_generation else None
    cfg_m1 = RAGPipelineConfig(
        query=_TARGET_QUERY,
        observability_level=CONTENT_HASH_LEVEL,
        attack_id="M1",
        attack_type="baseline",
        k=5,
        **corpus_meta,
        runs_root=RUNS_ROOT,
    )
    run_dir_m1 = RAGPipeline(cfg_m1, retriever=tampered_retriever, generator=gen).run()
    evaluate_run(run_dir_m1)

    # Find a chunk that was retrieved and tamper its text in ChromaDB
    tampered_chunk_id = _tamper_one_retrieved_chunk(
        run_dir_m1, str(tampered_index), _CANONICAL_COLLECTION)

    # Now verify — mismatch should be detected
    result_m1 = verify_content_hashes(
        run_dir_m1, str(tampered_index), _CANONICAL_COLLECTION)
    print(f"  M1 verified={result_m1['verified']}  "
          f"checked={result_m1['checked']}  "
          f"mismatches={len(result_m1['mismatches'])}")
    print(f"  tampered chunk: {tampered_chunk_id}")

    report = {
        "M0_clean": {
            "description": "Normal run, no corpus modification",
            "verified": result_m0["verified"],
            "checked": result_m0["checked"],
            "mismatches": result_m0["mismatches"],
        },
        "M1_tampered": {
            "description": "Run at L4, then one chunk text modified in ChromaDB",
            "tampered_chunk_id": tampered_chunk_id,
            "verified": result_m1["verified"],
            "checked": result_m1["checked"],
            "mismatches": result_m1["mismatches"],
        },
        "detection_result": (
            "DETECTED" if not result_m1["verified"] and result_m0["verified"]
            else "FAILED"
        ),
    }

    out = ANALYSIS_DIR / "content_hash_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nContent hash report written → {out}")
    print(f"Detection result: {report['detection_result']}")
    return report


def _tamper_one_retrieved_chunk(
    run_dir: Path,
    chroma_path: str,
    collection_name: str,
) -> str | None:
    """Modify the first retrieved chunk's text in ChromaDB in-place."""
    import chromadb

    events_path = run_dir / "events.jsonl"
    chunk_id = None
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("event") == "retrieval.completed":
            topk = ev.get("payload", {}).get("topk", [])
            if topk:
                chunk_id = str(topk[0].get("chunk_id") or "")
                break

    if not chunk_id:
        return None

    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_collection(collection_name)

    result = col.get(ids=[chunk_id], include=["documents", "metadatas"])
    docs = (result.get("documents") or [[]])[0]
    original_text = docs if isinstance(docs, str) else (docs[0] if docs else "")
    meta = ((result.get("metadatas") or [[]])[0] or {})

    col.update(
        ids=[chunk_id],
        documents=[original_text + _TAMPERED_SUFFIX],
        metadatas=[meta],
    )
    return chunk_id
