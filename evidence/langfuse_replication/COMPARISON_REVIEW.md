# Table VII replication review — approval required for further manuscript changes

On 2026-10-06 we replayed six retained runs (B1, D1, D3, D5, I1, I3) into the user-authorized EU Langfuse Cloud project RAGWitness-comparison. UI server version: 4.51.0; free Hobby plan. Transport: official OTLP/HTTP JSON, Python standard library, no SDK. No Docker, new retrieval, embedding, model generation, or TSA call was used.

18 traces / 42 observations were ingested and exported: six bare traces, six with declared retrieval/generation spans, six enhanced traces. All 54 read-back checks pass with the explicitly documented seed transformation: enhanced modelParameters.seed is exported as string "null", semantically normalized to None for comparison. Unmodified raw exports are retained. Query, answer, prompt, retrieved identifiers/order/URLs/text and model fields match their submitted artifacts. Current timestamps and zero span durations describe this replay, not historical execution performance.

## Proposed table (not yet applied)

| Question | LF bare | LF declared spans | LF enhanced spans | RW L1 | RW L5 |
|---|---|---|---|---|---|
| Q1: Injection present? | Partial | Yes | Yes | Partial | Yes |
| Q2: Direct or indirect? | Partial | Yes | Yes | Partial | Yes |
| Q3: Injection source? | Partial | Yes | Yes | Partial | Yes |
| Q4: Ranked chunks retrieved? | No | Yes | Yes | No | Yes |
| Q5: Assembled prompt? | No | Yes | Yes | No | Yes |
| Q6: Model and retained parameters? | No | Partial | Yes | No | Yes |
| Q7: Cryptographic evidence assurance? | No | No | No | Partial | Yes |

Q1–Q3 use the same retained reconstruction rules and trusted historical clean corpus as RAGWitness, with evaluator labels withheld. Bare traces support the three direct cases; the instrumented configurations support all six cases, including a negative baseline finding. The two instrumented configurations agree with the held-out source labels for all six selected cases; bare traces identify the three direct sources but cannot assess the remaining three cases. This is bounded coverage under the existing heuristic, not an independent semantic attack detector or population-wide accuracy result.

Q4 and Q5 are complete artifact availability. For Q6, declared spans preserve a model name in six cases but omit temperature/top_p, hence Partial; enhanced spans preserve all retained settings, with null seed. Yes denotes the required artifacts in all six cases; Partial denotes incomplete case coverage or incomplete required fields, as identified above. This distinction must be stated rather than mixing a strict binary metric with a model-only availability judgment. Q7 is an assurance distinction, not a fraction of answered cases: local chain consistency for RW L1 and externally anchored evidence for RW L5 under stated assumptions. Langfuse traces in these configurations contain neither a submitted native hash chain nor an independent timestamp anchor. No database-tampering test or comprehensive Langfuse security audit was performed.

RW columns retain the approved two corrections and existing assurance distinction, supported by the earlier RAGWitness audit. The current replay uses the six runs under runs_langfuse; earlier RW coverage checks used their recorded canonical counterparts. No paired performance estimate or claim of identical generated answers between those historical sets is made. The Langfuse replication concerns recording and export of fixed saved artifacts.

## Changes relative to the current manuscript

LF default (rename LF bare): Q1/Q2/Q3 No to Partial; Q5/Q6 Partial to No.

LF + spans (rename LF declared spans): Q1 Partial to Yes; Q2 No to Yes; Q3 Partial to Yes; Q5 Partial to Yes. Q4/Q6/Q7 unchanged.

Add the enhanced configuration as a clearly identified new comparator column; it records extra fields and supports Q1–Q6, without cryptographic evidence assurance in this replication. Both existing comparison columns represent explicitly specified instrumentation choices, not universal Langfuse defaults or limitations.

Revise accompanying main/supplement prose consistently: logs with sufficient instrumentation can support the same reconstruction questions when supplied the same analyzer and clean reference. The demonstrated distinction is the preservation and verification protocol, plus how much evidence is captured at each level. Claims of exclusive attack attribution or prompt reconstruction by RAGWitness are not supported by this comparison.

## Scope and remaining limits

The original Langfuse backend and exports were deleted. This is a new, dated replication on a newer cloud deployment; it does not validate historical backend behavior, timing, storage, or SDK operation. Attack labels and poison flags were not submitted as trace metadata. Query-time synthetic IDs/URLs still have recognizable text, as in the original experiment; the existing corpus membership heuristic is not based on an attack-ID substring. The earlier opaque-ID audit covers RAGWitness, not a new end-to-end opaque-ID Langfuse replay.

The prepared enhanced configuration is additional targeted evidence. No manuscript result or cell beyond the two previously approved RW L1 corrections has been changed. All original EC/AA/RF/ASR/FT statistics and figures remain untouched. RAGWitness's overall novelty argument must acknowledge the overlap in information capture.

Raw exports, submitted OTLP JSON, receipts, case mapping, hashes, scripts, and a UI screenshot are retained here. The temporary replication API key was revoked after exporting evidence, and the local credential file was removed. The cloud project and traces remain available to the user; API credentials are not included in this delivery.

Official interface references: https://langfuse.com/docs/observability/get-started and https://langfuse.com/integrations/native/opentelemetry
