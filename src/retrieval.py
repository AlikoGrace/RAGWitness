from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.embeddings import Embedder


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    chunk_id: str
    distance: float
    metadata: dict[str, Any]
    text: str

    @property
    def document_sha256(self) -> str:
        return str(self.metadata.get("sha256_pdf") or self.metadata.get("document_sha256") or "")


class Retriever:
    """Chroma-backed dense retriever used by experiments."""

    def __init__(
        self,
        chroma_path: str,
        collection_name: str,
        embedder: "Embedder",
    ) -> None:
        import chromadb

        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(collection_name)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed_texts([query], batch_size=1)[0].tolist()
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for idx, chunk_id in enumerate(ids):
            chunks.append(
                RetrievedChunk(
                    rank=idx + 1,
                    chunk_id=str(chunk_id),
                    distance=float(distances[idx]),
                    metadata=dict(metadatas[idx] or {}),
                    text=str(documents[idx] or ""),
                )
            )
        return chunks
