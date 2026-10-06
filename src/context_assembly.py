from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.prompts import RAG_SYSTEM_PROMPT, build_user_prompt
from src.retrieval import RetrievedChunk


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AssembledContext:
    system_prompt: str
    user_prompt: str
    prompt_sha256: str
    evidence_blocks: list[dict[str, str]]


def assemble_context(
    question: str,
    chunks: list[RetrievedChunk],
    max_chars_per_chunk: int = 1200,
) -> AssembledContext:
    evidence_blocks = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text[:max_chars_per_chunk],
        }
        for chunk in chunks
    ]
    user_prompt = build_user_prompt(question, evidence_blocks)
    full_prompt = f"{RAG_SYSTEM_PROMPT}\n\n{user_prompt}"

    return AssembledContext(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        prompt_sha256=sha256_text(full_prompt),
        evidence_blocks=evidence_blocks,
    )
