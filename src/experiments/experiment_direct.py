from __future__ import annotations

import json
from pathlib import Path

from src.corpus_snapshot import build_corpus_metadata
from src.embeddings import Embedder
from src.generation import StaticGenerator
from src.metrics import evaluate_run
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.retrieval import Retriever


def load_direct_catalog(path: str | Path = "data/catalogs/direct_injection_variants.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_direct_experiments(
    catalog_path: str | Path = "data/catalogs/direct_injection_variants.json",
    levels: list[int] | None = None,
    mock_generation: bool = False,
    runs_root: str = "runs",
    limit: int = 0,
    corpus_version: str = "v1.0_200docs_15482chunks",
) -> list[Path]:
    attacks = load_direct_catalog(catalog_path)
    if limit > 0:
        attacks = attacks[:limit]
    levels = levels or [1, 2, 3, 4, 5]
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
                StaticGenerator(answer=f"MOCK_DIRECT_ANSWER_{attack['attack_id']}_L{level}")
                if mock_generation
                else None
            )
            config = RAGPipelineConfig(
                query=attack["query"],
                observability_level=level,
                attack_id=attack["attack_id"],
                attack_type="direct",
                **corpus_metadata,
                runs_root=runs_root,
            )
            run_dir = RAGPipeline(config, retriever=retriever, generator=generator).run()
            evaluate_run(run_dir)
            run_dirs.append(run_dir)
    return run_dirs
