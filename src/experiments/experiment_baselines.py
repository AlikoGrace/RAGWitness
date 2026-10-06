from __future__ import annotations

import json
from pathlib import Path

from src.corpus_snapshot import build_corpus_metadata
from src.embeddings import Embedder
from src.generation import StaticGenerator
from src.metrics import evaluate_run
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.retrieval import Retriever


def load_baseline_catalog(path: str | Path = "data/catalogs/baseline_queries.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _slice_baseline_catalog(
    rows: list[dict],
    start_index: int = 1,
    end_index: int = 0,
    limit: int = 0,
) -> list[dict]:
    if start_index < 1:
        raise ValueError("start_index must be 1 or greater")
    selected = rows[start_index - 1 :]
    if end_index > 0:
        if end_index < start_index:
            raise ValueError("end_index must be greater than or equal to start_index")
        selected = rows[start_index - 1 : end_index]
    if limit > 0:
        selected = selected[:limit]
    return selected


def run_baseline_experiments(
    catalog_path: str | Path = "data/catalogs/baseline_queries.json",
    levels: list[int] | None = None,
    mock_generation: bool = False,
    runs_root: str = "runs",
    limit: int = 0,
    start_index: int = 1,
    end_index: int = 0,
    corpus_version: str = "v1.0_200docs_15482chunks",
) -> list[Path]:
    levels = levels or [1, 2, 3, 4, 5]
    corpus_metadata = build_corpus_metadata(corpus_version=corpus_version)
    baselines = _slice_baseline_catalog(
        load_baseline_catalog(catalog_path),
        start_index=start_index,
        end_index=end_index,
        limit=limit,
    )

    run_dirs: list[Path] = []
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True)
    retriever = Retriever(
        chroma_path="indexes/chroma_hansard",
        collection_name="hansard_chunks",
        embedder=embedder,
    )
    for baseline in baselines:
        baseline_id = baseline["baseline_id"]
        query = baseline["query"]
        for level in levels:
            generator = (
                StaticGenerator(answer=f"MOCK_BASELINE_ANSWER_{baseline_id}_L{level}")
                if mock_generation
                else None
            )
            config = RAGPipelineConfig(
                query=query,
                observability_level=level,
                attack_id=baseline_id,
                attack_type="baseline",
                **corpus_metadata,
                runs_root=runs_root,
            )
            run_dir = RAGPipeline(config, retriever=retriever, generator=generator).run()
            evaluate_run(run_dir)
            run_dirs.append(run_dir)
    return run_dirs
