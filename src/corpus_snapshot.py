from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CORPUS_VERSION = "v1.0_200docs_15482chunks"
DEFAULT_CHUNKS_PATH = Path("data/processed/hansard/chunks.jsonl")
DEFAULT_DOCS_PATH = Path("data/processed/hansard/docs.jsonl")
DEFAULT_STATS_PATH = Path("data/corpus_stats.json")
DEFAULT_CHROMA_PATH = Path("indexes/chroma_hansard")


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: str | Path) -> str:
    """Hash a directory snapshot by relative path and file bytes."""

    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_corpus_stats(stats_path: str | Path = DEFAULT_STATS_PATH) -> dict[str, Any]:
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus stats file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_corpus_metadata(
    corpus_version: str = DEFAULT_CORPUS_VERSION,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    docs_path: str | Path = DEFAULT_DOCS_PATH,
    stats_path: str | Path = DEFAULT_STATS_PATH,
    chroma_path: str | Path = DEFAULT_CHROMA_PATH,
) -> dict[str, Any]:
    chunks_path = Path(chunks_path)
    docs_path = Path(docs_path)
    stats_path = Path(stats_path)
    chroma_path = Path(chroma_path)
    stats = load_corpus_stats(stats_path)
    corpus_hash = sha256_file(chunks_path) if chunks_path.exists() else "MISSING"
    corpus_docs_hash = sha256_file(docs_path) if docs_path.exists() else "MISSING"
    chroma_collection = str(stats.get("chroma_collection") or "hansard_chunks")
    chroma_snapshot_hash = sha256_json(
        {
            "snapshot_type": "logical_chroma_collection",
            "chroma_path": str(chroma_path),
            "collection": chroma_collection,
            "corpus_hash": corpus_hash,
            "corpus_docs_hash": corpus_docs_hash,
            "docs": int(stats.get("docs", 0)),
            "chunks_total": int(stats.get("chunks_total", 0)),
        }
    )

    return {
        "corpus_version": corpus_version,
        "corpus_docs": int(stats.get("docs", 0)),
        "corpus_chunks": int(stats.get("chunks_total", 0)),
        "corpus_pages_total": int(stats.get("pages_total", 0)),
        "corpus_chunks_path": str(chunks_path),
        "corpus_docs_path": str(docs_path),
        "corpus_stats_path": str(stats_path),
        "corpus_hash": corpus_hash,
        "corpus_docs_hash": corpus_docs_hash,
        "corpus_stats_hash": sha256_file(stats_path),
        "chroma_snapshot_path": str(chroma_path),
        "chroma_snapshot_hash": chroma_snapshot_hash,
        "corpus_chroma_collection": chroma_collection,
        "effective_chroma_path": str(chroma_path),
        "effective_chroma_snapshot_hash": chroma_snapshot_hash,
    }
