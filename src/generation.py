from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.ollama_client import ollama_chat


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "llama3.1:8b-instruct-q4_K_M"
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    raw_response: dict[str, Any]


class GeneratorLike(Protocol):
    config: GenerationConfig

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        ...


class Generator:
    """Local Ollama-backed generator."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        raw = ollama_chat(
            model=self.config.model,
            system=system_prompt,
            user=user_prompt,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            seed=self.config.seed,
        )
        answer = (raw.get("message") or {}).get("content", "")
        return GenerationResult(answer=answer, raw_response=raw)


class StaticGenerator:
    """Deterministic generator for smoke tests and dry-run experiments."""

    def __init__(self, answer: str = "MOCK_ANSWER", config: GenerationConfig | None = None) -> None:
        self.answer = answer
        self.config = config or GenerationConfig(model="mock-generator", temperature=0.0, top_p=1.0)

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        return GenerationResult(
            answer=self.answer,
            raw_response={
                "model": self.config.model,
                "message": {"role": "assistant", "content": self.answer},
                "mock": True,
            },
        )
