"""
Novel holdout experiment runner for RAGWitness Phase 3 Item 7.

Runs N1–N5 at levels 1, 3, 5 (15 runs total) into runs_novel/.
These attacks are never seen during IsolationForest training.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.corpus_snapshot import build_corpus_metadata
from src.embeddings import Embedder
from src.generation import StaticGenerator
from src.metrics import evaluate_run
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.retrieval import Retriever


def run_novel_experiments(
    catalog_path: str | Path = "data/catalogs/novel_holdout_attacks.json",
    levels: list[int] | None = None,
    mock_generation: bool = False,
    runs_root: str = "runs_novel",
    corpus_version: str = "v1.0_200docs_15482chunks",
) -> list[Path]:
    attacks = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    levels = levels or [1, 3, 5]
    corpus_metadata = build_corpus_metadata(corpus_version=corpus_version)

    run_dirs: list[Path] = []
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True)
    retriever = Retriever(
        chroma_path="indexes/chroma_hansard",
        collection_name="hansard_chunks",
        embedder=embedder,
    )

    for attack in attacks:
        for level in levels:
            generator = (
                StaticGenerator(answer=f"MOCK_NOVEL_ANSWER_{attack['attack_id']}_L{level}")
                if mock_generation
                else None
            )
            config = RAGPipelineConfig(
                query=attack["query"],
                observability_level=level,
                attack_id=attack["attack_id"],
                attack_type=attack.get("attack_type", "direct"),
                **corpus_metadata,
                runs_root=runs_root,
            )
            print(f"  Running {attack['attack_id']} ({attack['name']}) L{level} …", flush=True)
            run_dir = RAGPipeline(config, retriever=retriever, generator=generator).run()
            evaluate_run(run_dir)
            run_dirs.append(run_dir)
            print(f"    → {run_dir.name}")

    return run_dirs


if __name__ == "__main__":
    import sys
    mock = "--mock" in sys.argv
    dirs = run_novel_experiments(mock_generation=mock)
    print(f"\nDone. {len(dirs)} runs written to runs_novel/")
