"""
Formal trust boundaries and forensic framework constants for RAGWitness.

Adversary model and DFRWS grounding are documented in THREAT_MODEL.md.
This module encodes the same information as importable constants so that
experiment code and reconstruction logic can reference them explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# ---------------------------------------------------------------------------
# Adversary positions
# ---------------------------------------------------------------------------

ADVERSARY_DIRECT: Final = "direct"
"""Query-side adversary: crafts malicious user queries (D1–D5)."""

ADVERSARY_INDIRECT: Final = "indirect"
"""Corpus-side adversary: inserts poisoned documents before ingestion (I1–I5)."""

ADVERSARY_MIXED: Final = "mixed"
"""Both query-side and corpus-side attack signals detected in the same run."""


# ---------------------------------------------------------------------------
# Trust boundary declarations
# ---------------------------------------------------------------------------

#: Components assumed uncompromised (Trusted Computing Base).
TRUSTED_COMPONENTS: Final[tuple[str, ...]] = (
    "run_manager",       # hash-chain log writer
    "embedder",          # local model weights
    "llm_weights",       # local Ollama model
    "ingestion_pipeline",# offline corpus preparation
)

#: Components in the attack surface (explicitly untrusted at runtime).
UNTRUSTED_COMPONENTS: Final[tuple[str, ...]] = (
    "user_query",        # direct injection surface
    "corpus_body_text",  # indirect injection surface
    "corpus_metadata",   # metadata poisoning surface (I5)
    "retrieved_chunks",  # indirect injection enters generation via retrieval
)


# ---------------------------------------------------------------------------
# The seven forensic questions, grounded in DFRWS (Palmer 2001) and
# NIST SP 800-86 (Kent et al. 2006).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForensicQuestion:
    key: str
    text: str
    dfrws_phase: str
    nist_phase: str
    min_observability_level: int


FORENSIC_QUESTIONS: Final[tuple[ForensicQuestion, ...]] = (
    ForensicQuestion(
        key="injection_present",
        text="Was a prompt injection attempt present?",
        dfrws_phase="Identification",
        nist_phase="Examination",
        min_observability_level=1,
    ),
    ForensicQuestion(
        key="attack_type",
        text="Was the attack direct (query-side) or indirect (corpus-side)?",
        dfrws_phase="Identification",
        nist_phase="Examination",
        min_observability_level=2,
    ),
    ForensicQuestion(
        key="source_attribution",
        text="Which input or retrieved document was the attack source?",
        dfrws_phase="Analysis",
        nist_phase="Analysis",
        min_observability_level=3,
    ),
    ForensicQuestion(
        key="retrieval_sequence",
        text="Which chunks were retrieved and in what order?",
        dfrws_phase="Collection",
        nist_phase="Collection",
        min_observability_level=2,
    ),
    ForensicQuestion(
        key="prompt_reconstruction",
        text="What prompt was assembled and sent to the LLM?",
        dfrws_phase="Collection",
        nist_phase="Collection",
        min_observability_level=5,
    ),
    ForensicQuestion(
        key="generation_replay",
        text="Which model and generation settings produced the response?",
        dfrws_phase="Collection",
        nist_phase="Collection",
        min_observability_level=5,
    ),
    ForensicQuestion(
        key="integrity_verification",
        text="Does the evidence hash chain verify without tampering?",
        dfrws_phase="Preservation",
        nist_phase="Reporting",
        min_observability_level=1,
    ),
)

FORENSIC_QUESTION_KEYS: Final[tuple[str, ...]] = tuple(q.key for q in FORENSIC_QUESTIONS)

# Total number of questions — used as the denominator for Evidence Completeness.
N_FORENSIC_QUESTIONS: Final[int] = len(FORENSIC_QUESTIONS)


# ---------------------------------------------------------------------------
# Hash chain scope declaration
# ---------------------------------------------------------------------------

#: What the hash chain guarantees.
HASH_CHAIN_GUARANTEE: Final = (
    "Tamper-evident after local write: any modification to events.jsonl after "
    "RunManager.finalize() completes will break chain verification."
)

#: What the hash chain does NOT guarantee.
HASH_CHAIN_NON_GUARANTEE: Final = (
    "Does not protect against OS-level or process-level compromise occurring "
    "before finalize() is called. Chain-of-custody-grade assurance requires an "
    "external RFC 3161 trusted timestamp anchor."
)


# ---------------------------------------------------------------------------
# Convenience: question lookup by key
# ---------------------------------------------------------------------------

def get_question(key: str) -> ForensicQuestion:
    for q in FORENSIC_QUESTIONS:
        if q.key == key:
            return q
    raise KeyError(f"No forensic question with key: {key!r}")
