from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from src.context_assembly import AssembledContext, assemble_context
from src.generation import GenerationConfig, Generator, GeneratorLike
from src.observability_config import ObservabilityConfig, get_observability_config
from src.retrieval import RetrievedChunk, Retriever
from src.run_manager import RunManager


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RetrieverLike(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        ...


@dataclass(frozen=True)
class RAGPipelineConfig:
    query: str
    observability_level: int
    attack_id: str = "baseline"
    attack_type: str = "baseline"
    expected_malicious_chunk_id: str | None = None
    poison_chunk_retrieved: bool | None = None  
    corpus_version: str = ""
    corpus_docs: int = 0
    corpus_chunks: int = 0
    corpus_pages_total: int = 0
    corpus_chunks_path: str = ""
    corpus_docs_path: str = ""
    corpus_stats_path: str = ""
    corpus_hash: str = ""
    corpus_docs_hash: str = ""
    corpus_stats_hash: str = ""
    chroma_snapshot_path: str = ""
    chroma_snapshot_hash: str = ""
    corpus_chroma_collection: str = ""
    effective_chroma_path: str = ""
    effective_chroma_snapshot_hash: str = ""
    chroma_path: str = "indexes/chroma_hansard"
    collection: str = "hansard_chunks"
    k: int = 5
    max_chars_per_chunk: int = 1200
    model: str = "llama3.1:8b-instruct-q4_K_M"
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int | None = None
    runs_root: str = "runs"


class RAGPipeline:
    """One RAG run with observability-aware forensic logging."""

    def __init__(
        self,
        config: RAGPipelineConfig,
        retriever: RetrieverLike | None = None,
        generator: GeneratorLike | None = None,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self.config = config
        self.observability = observability or get_observability_config(config.observability_level)
        self._retriever = retriever
        self._generator = generator

    def run(self) -> Path:
        rm = RunManager.create(
            runs_root=Path(self.config.runs_root),
            prefix=f"rag_L{self.config.observability_level}_{self.config.attack_id}",
            config={
                **asdict(self.config),
                "observability": asdict(self.observability),
            },
        )

        rm.log_event(
            "scenario.started",
            {
                "attack_id": self.config.attack_id,
                "attack_type": self.config.attack_type,
                "observability_level": self.config.observability_level,
            },
        )

        if self.observability.log_query:
            rm.log_event("query.received", {"query": self.config.query})

        chunks = self._get_retriever(rm).retrieve(self.config.query, k=self.config.k)
        self._log_retrieval(rm, chunks)

        context = assemble_context(
            self.config.query,
            chunks,
            max_chars_per_chunk=self.config.max_chars_per_chunk,
        )
        self._log_prompt(rm, context)

        generator = self._get_generator()
        result = generator.generate(context.system_prompt, context.user_prompt)
        self._log_generation(rm, result.answer, generator.config)

        integrity = rm.finalize()
        if self.observability.log_integrity_details:
            # Verify the local hash chain
            verification = RunManager.verify_hash_chain(rm.run_dir / "events.jsonl")
            rm.write_json(
                "artifacts/integrity_verification.json",
                {"integrity": integrity, "verification": verification},
            )
            # ── RFC 3161 anchor — wire into pipeline at L5 ───────────────
            # Attempt external timestamp anchoring. On network failure the run
            # still completes; the error is recorded inside integrity.json so
            # the absence of a token is explicitly documented, not silently lost.
            try:
                from src.timestamp_anchor import anchor_run_into_integrity
                anchor_run_into_integrity(rm.run_dir)
            except Exception as exc:
                # Merge error into integrity.json without crashing the run
                _merge_anchor_error(rm.run_dir, str(exc))

        return rm.run_dir

    def _get_retriever(self, rm: RunManager) -> RetrieverLike:
        if self._retriever is not None:
            return self._retriever

        from src.embeddings import Embedder

        embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", normalize=True)
        if self.observability.log_integrity_details:
            rm.log_event("embedder.loaded", embedder.fingerprint())

        return Retriever(
            chroma_path=self.config.chroma_path,
            collection_name=self.config.collection,
            embedder=embedder,
        )

    def _get_generator(self) -> GeneratorLike:
        if self._generator is not None:
            return self._generator

        return Generator(
            GenerationConfig(
                model=self.config.model,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                seed=self.config.seed,
            )
        )

    def _log_retrieval(self, rm: RunManager, chunks: list[RetrievedChunk]) -> None:
        if not self.observability.log_retrieved_ids:
            return

        topk: list[dict[str, Any]] = []
        for chunk in chunks:
            row: dict[str, Any] = {
                "rank": chunk.rank,
                "chunk_id": chunk.chunk_id,
            }
            if self.observability.log_retrieval_scores:
                row["distance"] = chunk.distance
            if self.observability.log_document_metadata:
                row["metadata"] = chunk.metadata
            if self.observability.log_retrieved_text:
                row["text"] = chunk.text
            if self.observability.log_document_hashes:
                row["text_sha256"] = sha256_text(chunk.text)
                row["document_sha256"] = chunk.document_sha256
            topk.append(row)

        rm.log_event("retrieval.completed", {"k": self.config.k, "topk": topk})

    def _log_prompt(self, rm: RunManager, context: AssembledContext) -> None:
        if not self.observability.log_full_prompt:
            return

        rm.log_event(
            "prompt.built",
            {
                "system_prompt": context.system_prompt,
                "user_prompt": context.user_prompt,
                "prompt_sha256": context.prompt_sha256,
                "evidence_blocks": context.evidence_blocks,
            },
        )

    def _log_generation(self, rm: RunManager, answer: str, config: GenerationConfig) -> None:
        payload: dict[str, Any] = {}

        if self.observability.log_response:
            payload["answer"] = answer
        if self.observability.log_response_hash:
            payload["answer_sha256"] = sha256_text(answer)
        if self.observability.log_generation_params:
            payload["model"] = config.model
            payload["temperature"] = config.temperature
            payload["top_p"] = config.top_p
            payload["seed"] = config.seed

        rm.log_event("generation.completed", payload)


def _merge_anchor_error(run_dir: Path, error: str) -> None:
    """Record a TSA anchor failure inside integrity.json without crashing."""
    import json as _json
    integrity_path = run_dir / "integrity.json"
    if integrity_path.exists():
        d = _json.loads(integrity_path.read_text())
        d["rfc3161_anchor"] = {"ok": False, "error": error}
        integrity_path.write_text(_json.dumps(d, indent=2))
