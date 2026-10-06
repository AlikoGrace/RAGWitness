# Langfuse targeted replication — prepared, not executed

The original raw Langfuse exports have not been recovered. This study will be a new, dated instrumentation replication, not a recreation of historical traces or timings.

Six retained cases (B1, D1, D3, D5, I1, I3) supply the actual query, ranked retrieval, assembled prompt and generated answer. prepare_replay.py creates 18 payloads: six bare traces, six explicitly instrumented traces matching the manuscript's declared fields, and six enhanced traces for a stronger comparator. The enhanced configuration adds retrieved text and all retained generation parameters. The latter is additional evidence, not an approved replacement of any manuscript cell.

No LLM or embedding calls are needed. No attack class, scenario label, poison flag, expected malicious ID or outcome is attached as trace metadata. Ground truth remains in evaluator_only_mapping.json and must never be uploaded. Naturally occurring identifiers and content are retained as query-time evidence. Model seed remains null when the historical run recorded null.

## Execution and evidence

1. Choose a local or existing cloud Langfuse project. No usable Docker runtime or running local instance was found during inspection. Local setup needs an available container runtime; cloud upload needs the user's project and permission to transfer these saved artifacts.
2. Pin and record server/SDK versions, image digests where available, instrumentation field policy and current replication time. Use supported SDK/OpenTelemetry ingestion rather than rerunning the old comparison script.
3. Ingest all configurations under neutral case names, retaining generated trace/observation identifiers. Replication timestamps describe ingestion; they are not the original model execution time. Do not interpret ingestion duration as pipeline performance.
4. Fetch complete raw observations from the server after asynchronous ingestion. Retain raw response JSON, request policy, hashes and ingestion/export receipts. Check returned query, answer, retrieval order, prompt and parameters against the prepared artifacts.
5. Score the main manuscript Q1–Q7 under one declared rubric. Separate an answerable negative finding from an inability to answer. For Q3 distinguish a user-query source from a retrieved-document source. For Q6 record available settings separately from reproducibility guarantees. For Q7 distinguish local consistency from independently anchored authenticity.
6. Keep corpus access symmetric: report trace-only coverage and, separately, results when the same trusted clean reference and analyzer are supplied to each logger. Do not count labels or poison flags as attribution evidence.
7. Do not infer 'No' from absent exports or unsupported API reads. Report unresolved. Do not claim that one deliberately narrow logging configuration limits Langfuse's overall capabilities.
8. Produce per-case evidence and a proposed revised table. Obtain approval for any further changed verdicts before editing the manuscript.

Official interface reference: https://langfuse.com/docs/api-and-data-platform/features/public-api

Current state: payload preparation complete; backend ingestion, raw export validation and rescoring pending. Table VII is not fully verified yet.
