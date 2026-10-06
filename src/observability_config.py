from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservabilityConfig:
    """Feature flags for one forensic-observability level."""

    level: int
    name: str
    log_query: bool
    log_response: bool
    log_response_hash: bool
    log_retrieved_ids: bool
    log_retrieval_scores: bool
    log_document_metadata: bool
    log_retrieved_text: bool
    log_document_hashes: bool
    log_full_prompt: bool
    log_generation_params: bool
    log_integrity_details: bool


LEVEL_NAMES = {
    1: "minimal",
    2: "basic",
    3: "standard",
    4: "enhanced",
    5: "forensic",
}


def get_observability_config(level: int) -> ObservabilityConfig:
    """Return the exact logging flags for a thesis observability level."""

    if level not in LEVEL_NAMES:
        raise ValueError("observability level must be between 1 and 5")

    return ObservabilityConfig(
        level=level,
        name=LEVEL_NAMES[level],
        log_query=True,
        log_response=True,
        log_response_hash=level >= 2,
        log_retrieved_ids=level >= 2,
        log_retrieval_scores=level >= 3,
        log_document_metadata=level >= 3,
        log_retrieved_text=level >= 4,
        log_document_hashes=level >= 4,
        log_full_prompt=level >= 5,
        log_generation_params=level >= 5,
        log_integrity_details=level >= 5,
    )


def all_observability_levels() -> list[ObservabilityConfig]:
    return [get_observability_config(level) for level in sorted(LEVEL_NAMES)]
