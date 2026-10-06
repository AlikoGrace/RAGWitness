# RAGWitness Threat Model

**Version:** 1.0  
**Date:** 2026-05-12  
**Scope:** Forensic Observability Framework for RAG Systems Under Prompt Injection

This document defines the adversary model, trust boundaries, attack surface, and forensic
framework grounding for RAGWitness. It is the normative reference for all experimental
claims in the paper. Section III of the IEEE paper should be derived directly from this
document.

---

## 1. System Overview

A Retrieval-Augmented Generation (RAG) system has three principal stages:

```
[User Query] → [Retriever] → [Context Assembler] → [LLM Generator] → [Response]
                   ↑
           [Knowledge Corpus]
```

RAGWitness sits orthogonally to this pipeline. It does not modify retrieval or generation.
It instruments each stage to capture evidence under a configurable observability level
(L1–L5), writing tamper-evident, hash-chained event logs that can support post-incident
forensic reconstruction.

---

## 2. Adversary Model

### 2.1 Adversary Goals

The adversary seeks one or more of the following outcomes:

| Goal | Code | Description |
|------|------|-------------|
| Response manipulation | G1 | Cause the LLM to produce false, misleading, or harmful content |
| Evidence avoidance | G2 | Prevent the attack from being detected or attributed post-hoc |
| Attribution evasion | G3 | Obscure which input or document introduced the malicious instruction |
| Integrity subversion | G4 | Modify logged evidence to hide the attack or implicate a third party |

The RAGWitness framework is designed to frustrate G2, G3, and G4. It does not prevent G1
(that is the role of defenses such as input sanitisation or alignment training, which are
out of scope).

### 2.2 Adversary Capabilities

RAGWitness considers two distinct adversary positions:

#### Direct Adversary (query-side)

| Capability | Held |
|------------|------|
| Submit arbitrary text queries to the RAG system | Yes |
| Craft queries that embed prompt injection instructions | Yes |
| Observe the LLM response | Yes |
| Access the system prompt | No |
| Read or write event logs | No |
| Modify the retrieval index | No |
| Know the observability level in use | No |

Attack variants: D1 (simple jailbreak), D2 (role manipulation), D3 (context override),
D4 (output format manipulation), D5 (multi-step chaining).

#### Indirect Adversary (corpus-side)

| Capability | Held |
|------------|------|
| Write documents to the knowledge corpus before ingestion | Yes |
| Embed malicious instructions in document body text | Yes |
| Embed malicious instructions in document metadata/title fields | Yes |
| Distribute an attack across multiple documents | Yes |
| Use obfuscation techniques in document text | Yes |
| Modify documents after indexing | No |
| Access the ChromaDB index at query time | No |
| Read or write event logs | No |

Attack variants: I1 (simple instruction injection), I2 (obfuscated injection),
I3 (distributed multi-document injection), I4 (delayed conditional trigger),
I5 (metadata poisoning).

### 2.3 Adversary Knowledge

The adversary is assumed to have the following knowledge:

- General RAG architecture (retrieval + context assembly + generation)
- The embedding model family likely in use (sentence-transformers class)
- The general category of LLM deployed (instruction-following model)
- That some form of system prompt governs answer behavior

The adversary does NOT know:

- The exact system prompt text
- The observability level in use
- The specific ChromaDB collection structure
- Whether forensic logs are being captured
- The hash seed or chain state

This is a realistic adversary: capable and informed about the technology, but not an
insider with privileged system access.

---

## 3. Trust Boundaries

### 3.1 Trusted Components (Trusted Computing Base)

The following are assumed correct and uncompromised:

| Component | File | Trust basis |
|-----------|------|-------------|
| Log system (RunManager) | `src/run_manager.py` | Local write, append-only |
| Hash chain computation | `src/run_manager.py:log_event` | SHA-256 over canonical JSON |
| Embedding model | `src/embeddings.py` | Loaded from local weights, not user-supplied |
| LLM weights | Ollama / local model | Not exposed to corpus or query at weight level |
| Corpus ingestion pipeline | `scripts/prepare_hansard_corpus.py` | Controlled offline process |

### 3.2 Untrusted Components (Attack Surface)

| Component | Threat |
|-----------|--------|
| User query text | Direct injection (D1–D5) |
| Corpus document body text | Indirect injection (I1–I4) |
| Corpus document metadata | Metadata poisoning (I5) |
| Retrieved chunk text passed to LLM | Prompt injection via retrieval |

### 3.3 Conditional Trust: The Hash Chain Limitation

The hash chain in `events.jsonl` is tamper-evident **after the point of local write**.
It provides the following guarantee:

> If `events.jsonl` is modified after `RunManager.finalize()` completes, the hash chain
> verification will fail and the tampering will be detectable.

It does NOT provide:

> Protection against an adversary who compromises the log-writing process itself (e.g.,
> OS-level or process-level compromise before `finalize()` is called).

This is a known limitation acknowledged in Section VIII of the paper. In high-assurance
deployments, an external timestamp anchor (RFC 3161 Trusted Timestamp) can extend the
chain-of-custody guarantee to a third-party authority.

---

## 4. What RAGWitness Does and Does Not Protect Against

| Claim | Status |
|-------|--------|
| Detects whether a direct injection was present in the query | Yes (pattern + blind ML) |
| Detects whether an indirect injection was retrieved | Yes, at L3+ (metadata) and L4+ (text) |
| Attributes the attack to the specific document/chunk | Yes, at L3+ |
| Enables full prompt reconstruction | Yes, at L5 only |
| Enables deterministic generation replay | Yes, at L5 only (requires seed) |
| Verifies evidence integrity post-hoc | Yes (hash chain) |
| Prevents attacks from succeeding | No — this is a forensic framework, not a defense |
| Protects logs from OS-level tampering | No — partial (see §3.3) |
| Handles zero-day attack signatures unknown to the system | Partial (blind ML detector) |
| Guarantees attribution when the poison chunk was not retrieved | No |

The last row is particularly important: **if the poison chunk is not retrieved, no
forensic evidence of the attack exists in the pipeline logs**. This is a fundamental
property of retrieval-based attacks, not a flaw in RAGWitness. It is reported as
the I2 non-retrieval finding in the results.

---

## 5. The Seven Forensic Questions: Grounding in DFRWS

The reconstruction protocol answers seven questions. These are derived from the
DFRWS (Digital Forensics Research Workshop) 2001 investigative process model, which
defines six phases: Identification, Preservation, Collection, Examination, Analysis,
and Presentation (Palmer 2001). NIST SP 800-86 maps these to incident response as
Collection, Examination, Analysis, and Reporting.

The mapping from DFRWS phases to RAGWitness forensic questions is:

| # | Forensic Question | DFRWS Phase | NIST SP 800-86 |
|---|-------------------|-------------|----------------|
| Q1 | Was a prompt injection attempt present? | Identification | Examination |
| Q2 | Was the attack direct (query-side) or indirect (corpus-side)? | Identification | Examination |
| Q3 | Which input or retrieved document was the attack source? | Analysis | Analysis |
| Q4 | Which chunks were retrieved and in what order? | Collection | Collection |
| Q5 | What prompt was assembled and sent to the LLM? | Collection | Collection |
| Q6 | Which model and generation settings produced the response? | Collection | Collection |
| Q7 | Does the evidence hash chain verify without tampering? | Preservation | Reporting |

This grounding establishes that the 7 questions are not arbitrary. They constitute
an operationalisation of the complete DFRWS investigative cycle for a RAG pipeline
incident: the investigator must identify whether an attack occurred (Q1, Q2), collect
the pipeline artifacts that constitute evidence (Q4, Q5, Q6), attribute the attack to
its source (Q3), and verify evidence integrity (Q7).

Evidence Completeness (EC) is thus interpretable as the proportion of DFRWS investigative
phases that can be completed from the available logs — not merely a count of boolean
flags. A run with EC = 0.43 (3/7 questions answered) can complete Identification and
Preservation but cannot complete Collection or Analysis: a meaningful forensic statement
about what the investigator can and cannot conclude.

### 5.1 Why Seven Questions, Not More or Fewer

The minimum forensically necessary set for a RAG incident is:

- At least one identification question (Q1) — to establish whether an incident occurred
- At least one source-attribution question (Q3) — to establish who/what is responsible
- At least one artifact-collection question (Q4 or Q5) — to reconstruct the causal chain
- At least one integrity question (Q7) — to establish evidence admissibility

Collapsing to fewer than 7 would merge distinct investigative capabilities (e.g., knowing
a chunk was retrieved is different from knowing what it said). Expanding beyond 7 would
require distinguishing sub-questions (e.g., each retrieved chunk separately) that are
more naturally reported as data fields than as capability questions.

---

## 6. Scope Limitations

The threat model explicitly excludes:

1. **Model weight poisoning** — attacks that modify LLM behaviour through fine-tuning
   rather than prompt manipulation. RAGWitness only instruments the inference pipeline.

2. **Side-channel attacks** — timing or resource-consumption attacks that infer system
   state without injecting content.

3. **Adversaries with log write access** — system-level compromise is out of scope.
   The framework assumes the investigator controls the logging environment.

4. **Cross-user attacks** — scenarios where one user's poisoned query contaminates
   another user's retrieval. The framework evaluates single-session incidents.

5. **Real-world corpora other than Ghana parliamentary Hansards** — the experimental
   evaluation uses a single-domain corpus. Generalisation to other corpora is a stated
   limitation (§VIII) and the subject of future work.

---

## 7. References (for paper Section III)

- Palmer, G. (2001). *A Road Map for Digital Forensic Research*. DFRWS Technical Report
  DTR-T001-01. Digital Forensics Research Workshop.

- Kent, K., Chevalier, S., Grance, T., Dang, H. (2006). *Guide to Integrating Forensic
  Techniques into Incident Response*. NIST Special Publication 800-86. National Institute
  of Standards and Technology.

- Carrier, B., Spafford, E. (2003). Getting physical with the digital investigation
  process. *International Journal of Digital Evidence*, 2(2), 1–20.

- Pollitt, M. (2013). An ad hoc review of digital forensic models. In *Advances in
  Digital Forensics*, Springer.
