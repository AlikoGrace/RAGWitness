# RAGWitness

**Forensic observability for retrieval-augmented generation under prompt injection.**

RAGWitness studies what a retrieval-augmented generation (RAG) system should preserve so that an investigator can examine a suspected prompt-injection incident after it occurs. Ordinary operational traces may show a query and response while omitting the retrieved evidence, assembled prompt, generation settings, or independently verifiable integrity information needed for investigation.

The framework defines five logging levels and evaluates them against seven forensic questions grounded in the Digital Forensic Research Workshop (DFRWS) investigative model. It connects the evidence available at each level with its storage cost and supports offline examination of the preserved artifacts.

## How it works

RAGWitness records progressively richer artifacts during retrieval and generation, then reconstructs the incident from the saved evidence:

| Level | Evidence added to the preceding level |
|---|---|
| L1 | User query and generated response |
| L2 | Retrieved chunk identifiers and ranks |
| L3 | Retrieval scores and chunk metadata, including source URLs |
| L4 | Retrieved text and per-chunk content hashes |
| L5 | Full assembled prompt, generation settings, and external timestamp anchoring |

Hash-chained event logs support checking the recorded sequence. RFC 3161 external timestamp anchoring adds an independent integrity check: regenerating a locally consistent chain does not make the modified bytes match an existing external anchor. Source identification uses observed query/retrieval artifacts and a trusted clean corpus reference; evaluator labels are used separately to score predictions.

The research focuses on evidence preservation and investigative capability. Its components support investigation rather than providing a new prompt-injection prevention algorithm.

## What the evaluation shows

The main evaluation covers **90 controlled runs**: five direct-injection scenarios, five indirect-injection scenarios, and eight benign baselines, each evaluated at five logging levels. The reference corpus contains **15,482 chunks** from Ghanaian parliamentary Hansards.

- **Evidence availability has a measurable storage cost.** Level 2 supports the reported candidate-source attribution capability at **1.19×** the fixed Level 1 mean storage. Level 5 covers the full defined forensic rubric at **7.23×**.
- **Provenance and causation are different questions.** In the evaluated fixed-snapshot, additive-poisoning setting, a retrieved identifier absent from the clean reference can identify a candidate source without preserving its text. Membership alone does not prove malicious intent or causal influence.
- **Retrieval creates a visibility boundary.** Increasing retrieval-log detail cannot identify an adversarial document that never appears in the retrieved evidence. The evaluated retrieval-evasion case motivates complementary ingestion-time records.
- **External anchoring strengthens integrity verification.** The tested hash-chain and timestamp design detects the five evaluated post-hoc tamper categories, including full chain regeneration that defeats hash chaining alone.

These findings are bounded by the evaluated scenarios and logging rubric. Detailed measurement definitions, statistical qualifications, retained-token availability, and auxiliary experiments are documented in [the reproduction guide](docs/REPRODUCTION.md).

## Paper and authors

**Forensic Observability for Retrieval-Augmented Generation Systems Under Prompt Injection Attacks**  
G. Aliko, K. O. Peasah, K. Owusu-Agyeman, and L. A. Banning.

This repository accompanies the manuscript prepared for IEEE Transactions on Information Forensics and Security (TIFS) and arXiv. A public manuscript link and publication details will be added when available.

[GitHub repository](https://github.com/AlikoGrace/RAGWitness) · [Detailed reproduction guide](docs/REPRODUCTION.md) · [Software citation](CITATION.cff)

## Start here: reproduce saved results without models

From this repository root, with Python 3.10 or newer:

```bash
python scripts/reproduce_saved_results.py
```

This needs only the Python standard library. It checks:

- All 90 canonical EC, AA, and RF scores against retained event logs and configurations, and their current hash chains.
- The original keyword-match ASR and FT signals/denominators for all 30 stealthy-answer runs.
- The six sensitivity comparisons reporting a +0.50 AA difference, using four attack cases per configuration and excluding the benign baseline.
- Storage normalization against the fixed mean of the 18 retained Level 1 measurements.

It writes a CSV and detailed JSON to `reproduced/`, leaving retained evidence unchanged. Ground truth is used to evaluate reconstructed predictions, not as an attribution signal. The clean reference contains all 15,482 original chunk identifiers; it is not a full-text corpus.

For statistical recomputation and a new overview plot of the retained data:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_saved_results.py --statistics --figures
```

The plot is an independently rendered overview, not a byte-identical reproduction of the paper's figure layout. Preserved paper figure assets are in `analysis/paper_figures/`.

## What is included

| Path | Purpose |
|---|---|
| `src/` | Framework, experiment runners, scoring, statistical analysis, and repaired integrity verification |
| `runs/` | Exactly the 90 cases selected by `fresh_90run_metrics.json` |
| `runs_stealthy/` | Saved answers and artifacts for original ASR/FT checks |
| `runs_langfuse/` | Six source cases used for the explicitly labeled instrumentation replication |
| `runs_model2/`, `runs_seed/` | Retained cross-model and seed-run evidence; see the reports for their scope |
| `analysis/tables/` | Preserved historical result JSON, including auxiliary reports |
| `data/catalogs/` | Query and attack definitions |
| `data/reference/` | Exact clean membership reference and document manifest |
| `evidence/langfuse_replication/` | Fresh declared-instrumentation payloads, exports, verification, and reconciliation notes |
| `evidence/review/` | Additional review diagnostics, distinct from reported experimental scores |
| `docs/REPRODUCTION.md` | Claim-to-artifact mapping, caveats, and fresh experiment instructions |
| `tests/` | Framework and integrity regression tests |

## Integrity and comparator verification

Hash chains alone do not establish independent custody or prevent full regeneration. Run `python scripts/verify_retained_anchors.py` after installing the verification dependencies to check retained timestamps. Timestamp verification requires OpenSSL, independently trusted CA certificates, and the expected TSA policy. See `docs/REPRODUCTION.md`. Some historical Level 5 timestamp tokens are absent; absence is not a successful verification.

Recheck retained Langfuse exports without a cloud account:

```bash
python evidence/langfuse_replication/verify_exports.py --repo . --root evidence/langfuse_replication --corpus data/reference/clean_chunk_ids.jsonl
```

This shared-analyzer evaluation is a labeled fresh replay, not recovery of the lost historical Langfuse instance. It uses no ground-truth labels as analyzer inputs. Its ID-only reference has a different file hash from the full historical corpus while preserving identical membership.

## Fresh experiments

Fresh generation requires the Hansard corpus, an embedding model, ChromaDB, and Ollama. Install `requirements-experiments.txt` and follow `docs/REPRODUCTION.md`. Models, vector databases, PDFs, Docker data, credentials, caches, and unrelated development runs are excluded to keep this package small. Fresh runs are new experiments; do not expect identical model answers or silently replace the retained paper measurements.

## License and citation

Code and author-created software documentation are MIT licensed. Third-party parliamentary material, quoted excerpts, publication figure assets, and retained trace/data artifacts are outside that license; see `DATA_NOTICE.md`. Use `CITATION.cff` to cite this software. A DOI and publication venue can be added after publication; neither is invented here.
