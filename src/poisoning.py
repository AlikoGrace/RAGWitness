from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.embeddings import Embedder


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PoisoningResult:
    attack_id: str
    chunks_jsonl: Path
    chroma_path: Path
    collection: str
    expected_malicious_chunk_id: str


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def build_poison_chunks(attack: dict[str, Any]) -> list[dict[str, Any]]:
    attack_id = attack["attack_id"]
    if "poison_text_parts" in attack:
        texts = list(attack["poison_text_parts"])
    else:
        texts = [attack.get("poison_text", "")]

    chunks: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        chunk_id = f"poison:{attack_id}:{idx}"
        metadata = attack.get("poison_metadata") or {}
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": f"poison_doc_{attack_id}",
                "attack_id": attack_id,
                "poisoned": True,
                "chunk_index": idx,
                "chunk_text": text,
                "chunk_sha256": sha256_text(text),
                "title": metadata.get("title", f"Synthetic poison document {attack_id}"),
                "source_url": f"synthetic://poison/{attack_id}",
                "ingestion_stage": "synthetic_poison_insert",
                **metadata,
            }
        )
    return chunks


def prepare_poisoned_corpus(
    attack: dict[str, Any],
    clean_chunks_path: str | Path = "data/processed/hansard/chunks.jsonl",
    output_root: str | Path = "data/corpus/poisoned",
    chroma_root: str | Path = "indexes",
    collection: str | None = None,
    max_clean_chunks: int = 0,
    rebuild_index: bool = True,
    embedder: "Embedder | None" = None,
) -> PoisoningResult:
    attack_id = attack["attack_id"]
    clean_chunks = load_jsonl(clean_chunks_path)
    if max_clean_chunks > 0:
        clean_chunks = clean_chunks[:max_clean_chunks]

    poison_chunks = build_poison_chunks(attack)
    all_chunks = clean_chunks + poison_chunks

    out_dir = Path(output_root) / attack_id
    chunks_jsonl = write_jsonl(out_dir / "chunks.jsonl", all_chunks)
    write_jsonl(out_dir / "poison_only.jsonl", poison_chunks)

    chroma_path = Path(chroma_root) / f"chroma_poisoned_{attack_id}"
    collection_name = collection or f"hansard_poisoned_{attack_id}"
    if rebuild_index:
        build_chroma_index(
            all_chunks,
            chroma_path=chroma_path,
            collection=collection_name,
            embedder=embedder,
        )

    return PoisoningResult(
        attack_id=attack_id,
        chunks_jsonl=chunks_jsonl,
        chroma_path=chroma_path,
        collection=collection_name,
        expected_malicious_chunk_id=poison_chunks[0]["chunk_id"],
    )


def build_chroma_index(
    chunks: list[dict[str, Any]],
    chroma_path: str | Path,
    collection: str,
    embedder: "Embedder | None" = None,
    add_batch_size: int = 5000,
) -> None:
    import chromadb
    from src.embeddings import Embedder

    chroma_path = Path(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    col = client.get_or_create_collection(collection)

    embedder = embedder or Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True)
    ids = [str(row["chunk_id"]) for row in chunks]
    docs = [str(row.get("chunk_text") or row.get("text") or "") for row in chunks]
    metas = [_clean_metadata(row) for row in chunks]
    embeddings = embedder.embed_texts(docs, batch_size=64).tolist()
    for start in range(0, len(ids), add_batch_size):
        end = start + add_batch_size
        col.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
            embeddings=embeddings[start:end],
        )


def _clean_metadata(row: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    for key, value in row.items():
        if key in {"chunk_text", "text"}:
            continue
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = json.dumps(value, ensure_ascii=False)
    return metadata
