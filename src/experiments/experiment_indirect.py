from __future__ import annotations

import json
from pathlib import Path

from src.corpus_snapshot import build_corpus_metadata, sha256_tree
from src.embeddings import Embedder
from src.generation import StaticGenerator
from src.metrics import evaluate_run
from src.poisoning import prepare_poisoned_corpus
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.retrieval import Retriever


def _probe_poison_retrieval(
    retriever: Retriever,
    query: str,
    expected_chunk_id: str,
    k: int = 5,
) -> bool:
    """
    Probe whether the poison chunk appears in the top-k for this query.
    Runs once per attack before all level iterations (retrieval is deterministic).
    Stored in config.json so investigators know whether the attack penetrated retrieval.
    """
    chunks = retriever.retrieve(query, k=k)
    return any(c.chunk_id == expected_chunk_id for c in chunks)


def load_indirect_catalog(path: str | Path = "data/catalogs/indirect_injection_catalog.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_indirect_experiments(
    catalog_path: str | Path = "data/catalogs/indirect_injection_catalog.json",
    levels: list[int] | None = None,
    mock_generation: bool = False,
    runs_root: str = "runs",
    limit: int = 0,
    max_clean_chunks: int = 0,
    rebuild_indexes: bool = True,
    corpus_version: str = "v1.0_200docs_15482chunks",
) -> list[Path]:
    attacks = load_indirect_catalog(catalog_path)
    if limit > 0:
        attacks = attacks[:limit]
    levels = levels or [1, 2, 3, 4, 5]
    corpus_metadata = build_corpus_metadata(corpus_version=corpus_version)

    run_dirs: list[Path] = []
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True)
    for attack in attacks:
        poison_result = prepare_poisoned_corpus(
            attack,
            max_clean_chunks=max_clean_chunks,
            rebuild_index=rebuild_indexes,
            embedder=embedder,
        )
        retriever = Retriever(
            chroma_path=str(poison_result.chroma_path),
            collection_name=poison_result.collection,
            embedder=embedder,
        )
        poison_retrieved = _probe_poison_retrieval(
            retriever,
            attack["target_query"],
            poison_result.expected_malicious_chunk_id,
        )
        for level in levels:
            generator = (
                StaticGenerator(answer=f"MOCK_INDIRECT_ANSWER_{attack['attack_id']}_L{level}")
                if mock_generation
                else None
            )
            config = RAGPipelineConfig(
                query=attack["target_query"],
                observability_level=level,
                attack_id=attack["attack_id"],
                attack_type="indirect",
                poison_chunk_retrieved=poison_retrieved,
                expected_malicious_chunk_id=poison_result.expected_malicious_chunk_id,
                **corpus_metadata,
                chroma_path=str(poison_result.chroma_path),
                collection=poison_result.collection,
                runs_root=runs_root,
            )
            config_dict = config.__dict__.copy()
            config_dict["effective_chroma_path"] = str(poison_result.chroma_path)
            config_dict["effective_chroma_snapshot_hash"] = sha256_tree(poison_result.chroma_path)
            config = RAGPipelineConfig(**config_dict)
            run_dir = RAGPipeline(config, retriever=retriever, generator=generator).run()
            evaluate_run(run_dir)
            run_dirs.append(run_dir)
    return run_dirs
