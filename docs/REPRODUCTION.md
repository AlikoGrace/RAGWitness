# Reproduction scope and artifact map

## Retained result checks

The canonical selection is `analysis/tables/fresh_90run_metrics.json`: 18 scenarios at five levels, totaling 90 runs. Its `run_dir` fields select the supplied evidence. A similarly named historical run directory is not substituted for this selection.

| Paper result | Evidence and computation |
|---|---|
| Logging-level EC/AA/RF and storage | Canonical 90 selected runs; `scripts/reproduce_saved_results.py`; `src/forensic_reconstruction.py`; `src/metrics.py` |
| Confidence intervals and paired analyses | `statistical_report.json`, `phase3_results.json`, `src/experiments/statistical_analysis.py`, `analysis/phase3_corrections.py` |
| FT and stealthy ASR | `stealthy_attack_report.json`, 30 saved stealthy runs, original keyword scorer and FT scorer |
| +0.50 AA sensitivity delta | `sensitivity_report.json` raw rows; attack-only n=4 per embedding/k/level; baseline inclusion changes the estimand |
| Langfuse comparison | Explicit fresh 18-trace/42-observation replication under `evidence/langfuse_replication/`; 54 fidelity checks; shared analyzer and clean membership reference |
| Timestamp and chain verification | Repaired `src/timestamp_anchor.py` and `src/run_manager.py`; retained token artifacts; independent offline tests; review diagnostics |
| Cross-model and multi-seed checks | Their historical table reports and supplied `runs_model2/`, `runs_seed/` evidence |
| Auxiliary ablations and baseline comparators | Preserved reports and runner source; their complete historical run directories are omitted from this compact package |

Historical result reports are preserved, not all re-executed by the quick check. Optional new cloud replay uses `evidence/langfuse_replication/cloud_replay.py --help` and a credentials file supplied locally; no credentials are bundled. The old `langfuse_comparison_report.json` and its old runner are historical and do not represent the revised manuscript comparison. Use the labeled fresh replication.

## Measurement qualifications

- Source membership supports candidate-source identification only for the evaluated fixed-snapshot additive-poisoning setting. It does not establish malicious intent or causal influence.
- Canonical generation seeds were unset; the separate seed experiment is not the canonical 90-run protocol.
- ASR is the original permissive any-substring proxy. No semantic re-scoring was substituted for those results.
- FT counts applicable true signals; the denominator differs by attack type and available artifacts. A valid chain is an FT signal, not successful attribution.
- Storage ratios use retained byte measurements and a fixed Level 1 mean. Current directory sizes differ from the original measurement snapshot; they exclude model, corpus, and vector-index storage. Investigation time is forensic reconstruction time, not end-to-end model latency.
- The I3 evaluation identifies a designated source chunk, not exhaustive distributed-attack causal coverage. I2 illustrates lack of retrieval visibility.
- Only four discordant AA pairs underlie the adjacent-level comparison. The exact paired-binary diagnostic is supplied separately; original reported p-values are retained.
- Ten of eighteen canonical Level 5 timestamp tokens were retained and verified in the audit; eight were absent. Audit manifests created later are not evidence of historical independent custody. Certificate revocation was not checked.

## Offline tests

After installing `requirements-experiments.txt`:

```bash
python scripts/run_offline_tests.py --repo .
```

This disables live TSA requests and runs the relevant integrity, metrics, reconstruction, and logging regression checks. An independent OpenSSL installation is needed for timestamp tests. It is not necessary for the standard-library saved-result quick check.

## Fresh pipeline execution

1. Obtain the 200 parliamentary documents matching `data/reference/document_manifest.json`, including the PDF SHA-256 values. Do not assume a newly downloaded or changed document is identical. The manifest records historical paths for provenance; use your own local PDF directory.
2. Build the full corpus with the provided ingestion runner (512-word chunks, 50-word overlap):

```bash
python scripts/prepare_hansard_corpus.py --pdf-dir data/corpus/clean/pdfs --metadata-jsonl data/corpus/clean/metadata.jsonl --max-docs 200
```

Provide your document metadata file, including source URLs. Missing metadata changes provenance signals. PDF extraction/dependency differences may change chunks; compare reconstructed IDs and saved content hashes before interpreting results as matched reproduction. The lightweight `clean_chunk_ids.jsonl` must not be used as a full-text retrieval corpus.

3. Install Ollama separately and obtain the original model:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

The canonical embedding model is `all-MiniLM-L6-v2`. Model download, index building, generation, and any live TSA requests need network/resources.

4. First run an isolated smoke experiment:

```bash
python run_experiment.py --phase all --levels 1,5 --limit 1 --max-clean-chunks 50 --mock-generation --runs-root new_runs/smoke
```

Then run fresh canonical phases only when the corpus and model are ready:

```bash
python run_experiment.py --phase all --runs-root new_runs/canonical
```

`--phase all` covers direct, indirect, and baseline phases; it does not cover every auxiliary experiment. Inspect the corresponding `src/experiments/` runners before scheduling sensitivity, stealthy, ablation, or cross-model studies. Fresh generation can vary. No full model suite was rerun while assembling this release.

## Public release status

This package is prepared locally. It must be published to the manuscript's intended repository before a public-release claim is verified. The full reference corpus is not distributed here; the manuscript's existing claim that the full corpus is released needs reconciliation before submission. GitHub publication alone does not resolve that corpus assertion.
