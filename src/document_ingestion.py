
from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
import chromadb

from src.run_manager import RunManager
from src.embeddings import Embedder


# -----------------------------
# Utilities
# -----------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def words(text: str) -> List[str]:
    # keep it simple: split on whitespace
    return text.split()


def chunk_words(tokens: List[str], chunk_size: int = 512, overlap: int = 50) -> List[List[str]]:
    assert chunk_size > overlap >= 0
    out = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_size, n)
        out.append(tokens[start:end])
        if end == n:
            break
        start = end - overlap
    return out


def extract_text_pymupdf(pdf_path: Path) -> Tuple[str, int]:
    doc = fitz.open(pdf_path)
    pages = doc.page_count
    page_texts = []
    for i in range(pages):
        page = doc.load_page(i)
        t = page.get_text("text") or ""
        page_texts.append(t.strip())
    doc.close()
    return normalize_whitespace("\n\n".join(page_texts)), pages


def load_metadata_map(metadata_jsonl: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not metadata_jsonl or not metadata_jsonl.exists():
        return {}
    m: Dict[str, Dict[str, Any]] = {}
    with metadata_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = None
            if rec.get("saved_path"):
                key = Path(rec["saved_path"]).name
            elif rec.get("scrape_saved_path"):
                key = Path(rec["scrape_saved_path"]).name
            elif rec.get("pdf_url"):
                key = Path(rec["pdf_url"]).name
            if key:
                m[key] = rec
    return m


# -----------------------------
# Core ingestion
# -----------------------------

@dataclass
class IngestionConfig:
    pdf_dir: Path
    metadata_jsonl: Optional[Path]
    out_docs_jsonl: Path
    out_chunks_jsonl: Path

    chunk_size_words: int = 512
    chunk_overlap_words: int = 50

    chroma_path: Path = Path("indexes/chroma_hansard")
    chroma_collection: str = "hansard_chunks"
    ingest_to_chroma: bool = True
    rebuild_chroma: bool = True

    max_docs: int = 0  # 0 = all


def ingest(cfg: IngestionConfig, run_prefix: str = "ingest_v1") -> Path:
    cfg.out_docs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_chunks_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cfg.chroma_path.mkdir(parents=True, exist_ok=True)

    meta_map = load_metadata_map(cfg.metadata_jsonl)

    rm = RunManager.create(
        runs_root=Path("runs"),
        prefix=run_prefix,
        config={
            "pdf_dir": str(cfg.pdf_dir),
            "metadata_jsonl": str(cfg.metadata_jsonl) if cfg.metadata_jsonl else None,
            "out_docs_jsonl": str(cfg.out_docs_jsonl),
            "out_chunks_jsonl": str(cfg.out_chunks_jsonl),
            "chunk_size_words": cfg.chunk_size_words,
            "chunk_overlap_words": cfg.chunk_overlap_words,
            "chroma_path": str(cfg.chroma_path),
            "chroma_collection": cfg.chroma_collection,
            "ingest_to_chroma": cfg.ingest_to_chroma,
            "rebuild_chroma": cfg.rebuild_chroma,
            "max_docs": cfg.max_docs,
        },
    )
    rm.log_event("ingest.started", {"pdf_dir": str(cfg.pdf_dir)})

    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True)
    rm.log_event("embedder.loaded", embedder.fingerprint())

    pdfs = sorted([p for p in cfg.pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if cfg.max_docs and cfg.max_docs > 0:
        pdfs = pdfs[: cfg.max_docs]

    # Setup Chroma (optional)
    col = None
    if cfg.ingest_to_chroma:
        client = chromadb.PersistentClient(path=str(cfg.chroma_path))
        if cfg.rebuild_chroma:
            try:
                client.delete_collection(name=cfg.chroma_collection)
            except Exception:
                pass
        col = client.get_or_create_collection(name=cfg.chroma_collection)

    docs_written = 0
    chunks_written = 0
    total_pages = 0
    chunk_lens = []

    with cfg.out_docs_jsonl.open("w", encoding="utf-8") as docs_f, cfg.out_chunks_jsonl.open("w", encoding="utf-8") as chunks_f:
        for idx, pdf_path in enumerate(pdfs, start=1):
            key = pdf_path.name
            meta = meta_map.get(key, {})
            doc_sha = sha256_file(pdf_path)
            text, pages = extract_text_pymupdf(pdf_path)
            total_pages += pages

            doc_id = meta.get("doc_id") or meta.get("title") or pdf_path.stem
            doc_rec = {
                "doc_id": doc_id,
                "pdf_path": str(pdf_path),
                "sha256_pdf": doc_sha,
                "pages": pages,
                "title": meta.get("title", ""),
                "date_text": meta.get("date_text", ""),
                "date_iso": meta.get("date_iso", ""),
                "source_url": meta.get("pdf_url", meta.get("source_url", "")),
                "text": text,
            }
            docs_f.write(json.dumps(doc_rec, ensure_ascii=False) + "\n")
            docs_written += 1

            toks = words(text)
            chunks = chunk_words(toks, chunk_size=cfg.chunk_size_words, overlap=cfg.chunk_overlap_words)

            # write chunks + optionally index
            chroma_ids = []
            chroma_docs = []
            chroma_metas = []

            for c_i, c_tokens in enumerate(chunks):
                chunk_text = " ".join(c_tokens)
                chunk_id = f"{doc_sha}:{c_i}"
                chunk_len = len(c_tokens)
                chunk_lens.append(chunk_len)

                chunk_rec = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "pdf_path": str(pdf_path),
                    "sha256_pdf": doc_sha,
                    "chunk_index": c_i,
                    "chunk_len_words": chunk_len,
                    "chunk_text": chunk_text,
                    "title": doc_rec["title"],
                    "date_iso": doc_rec["date_iso"],
                    "source_url": doc_rec["source_url"],
                }
                chunks_f.write(json.dumps(chunk_rec, ensure_ascii=False) + "\n")
                chunks_written += 1

                if col is not None:
                    chroma_ids.append(chunk_id)
                    chroma_docs.append(chunk_text)
                    chroma_metas.append({
                        "doc_id": doc_id,
                        "sha256_pdf": doc_sha,
                        "chunk_index": c_i,
                        "chunk_len_words": chunk_len,
                        "title": doc_rec["title"],
                        "date_iso": doc_rec["date_iso"],
                        "pdf_path": str(pdf_path),
                        "source_url": doc_rec["source_url"],
                    })

            if col is not None and chroma_ids:
                # Explicit, controlled embeddings (journal-grade reproducibility)
                vecs = embedder.embed_texts(chroma_docs, batch_size=64)
                col.add(ids=chroma_ids, documents=chroma_docs, metadatas=chroma_metas, embeddings=vecs.tolist())

            if idx % 5 == 0:
                rm.log_event("ingest.progress", {"docs_done": idx, "chunks_total": chunks_written, "last_pdf": str(pdf_path)})

    avg_chunk_len = (sum(chunk_lens) / len(chunk_lens)) if chunk_lens else 0.0
    summary = {
        "docs": docs_written,
        "pages_total": total_pages,
        "chunks_total": chunks_written,
        "avg_chunks_per_doc": (chunks_written / docs_written) if docs_written else 0.0,
        "avg_chunk_len_words": avg_chunk_len,
        "chunk_size_words": cfg.chunk_size_words,
        "chunk_overlap_words": cfg.chunk_overlap_words,
        "out_docs_jsonl": str(cfg.out_docs_jsonl),
        "out_chunks_jsonl": str(cfg.out_chunks_jsonl),
        "chroma_path": str(cfg.chroma_path),
        "chroma_collection": cfg.chroma_collection,
        "ingest_to_chroma": cfg.ingest_to_chroma,
        "rebuild_chroma": cfg.rebuild_chroma,
    }

    artifacts = rm.run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "ingestion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rm.log_event("ingest.completed", summary)
    rm.finalize()
    return rm.run_dir
