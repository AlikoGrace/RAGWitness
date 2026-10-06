#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.document_ingestion import IngestionConfig, ingest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the processed Hansard JSONL corpus and Chroma index."
    )
    parser.add_argument("--pdf-dir", default="data/corpus/clean/pdfs")
    parser.add_argument("--metadata-jsonl", default="data/corpus/clean/metadata.jsonl")
    parser.add_argument("--out-dir", default="data/processed/hansard")
    parser.add_argument("--max-docs", type=int, default=200)
    parser.add_argument("--chunk-size-words", type=int, default=512)
    parser.add_argument("--chunk-overlap-words", type=int, default=50)
    parser.add_argument("--chroma-path", default="indexes/chroma_hansard")
    parser.add_argument("--collection", default="hansard_chunks")
    parser.add_argument("--stats-path", default="data/corpus_stats.json")
    parser.add_argument("--run-prefix", default="ingest_hansard_200")
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Write JSONL outputs without rebuilding the Chroma index.",
    )
    parser.add_argument(
        "--append-index",
        action="store_true",
        help="Append to the existing Chroma collection instead of deleting it first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    cfg = IngestionConfig(
        pdf_dir=Path(args.pdf_dir),
        metadata_jsonl=Path(args.metadata_jsonl) if args.metadata_jsonl else None,
        out_docs_jsonl=out_dir / "docs.jsonl",
        out_chunks_jsonl=out_dir / "chunks.jsonl",
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        chroma_path=Path(args.chroma_path),
        chroma_collection=args.collection,
        ingest_to_chroma=not args.no_index,
        rebuild_chroma=not args.append_index,
        max_docs=args.max_docs,
    )

    available_pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    target = "all" if args.max_docs == 0 else str(args.max_docs)
    print(f"Preparing Hansard corpus from {len(available_pdfs)} available PDFs; target={target}.")
    run_dir = ingest(cfg, run_prefix=args.run_prefix)

    summary_path = run_dir / "artifacts" / "ingestion_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "run_dir": str(run_dir),
            "available_pdfs": len(available_pdfs),
            "requested_max_docs": args.max_docs,
            "stats_path": args.stats_path,
        }
    )

    stats_path = Path(args.stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Corpus preparation complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
