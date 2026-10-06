# Historical versus replicated comparison

The original evidence is preserved unchanged in historical_report_preserved.json. No additional manuscript cell or finding has been changed in this reconciliation. The earlier proposed replacement table is a proposal, not a demonstration that the deleted historical system behaved identically to the new deployment.

## What differs

| Dimension | Retained historical report/script | New replication |
|---|---|---|
| Deployment | Report states Langfuse 3.173.0, SDK 4.6.1; original runtime unavailable | Cloud EU UI 4.51.0, direct official OTLP/HTTP JSON |
| Model execution | Original script runs retrieval and model generation | Fixed retained RAGWitness inputs and answers; zero model calls |
| Configurations | One column named langfuse_default, although script adds retrieval/generation observations | Explicit bare, declared-spans and enhanced-spans configurations |
| Question mapping | Q2 retrieved chunks; Q3 poison presence; Q4 grounding; Q5 prompt capture/tampering; Q6 full pipeline | Main-paper mapping: Q2 attack class; Q3 source; Q4 retrieval; Q5 prompt; Q6 model/settings |
| Detector comparison | Langfuse no native detector; RAGWitness reconstructor receives artifacts | Common external analyzer and clean reference supplied to compare artifact usefulness |
| Labels | Script supplies attack_id/type; historical RW answer helper checks poisoned flags | Labels withheld from trace metadata and source analyzer; evaluator mapping separate |
| Raw trace lineage | All six trace_id fields empty; report retains descriptions rather than raw observations | Trace IDs, submitted artifacts, server exports and field-fidelity checks preserved |
| Prompt | Script constructs a separate simplified prompt for its Langfuse observation while invoking RAGWitness pipeline | The actual saved prompt.built content is replayed |

The historical report's Q5 returns Partial when prompt text is present but has no integrity hash. Table VII's Q5 asks 'What prompt assembled?' rather than whether the prompt is authenticated. The replication's Yes denotes captured complete prompt content, not independent authenticity. Thus those verdicts cannot be compared as identical measurements.

Similarly, the historical script describes Langfuse's native detection capability, while the replication tests whether exported artifacts can support a separate shared analyzer. These are different legitimate questions. Neither should be presented as the other.

The historical report does provide evidence that its instrumented Langfuse comparison recorded retrieval identifiers and prompt context: its original Q2 is Yes for all six cases and its Q5 is Partial for all six. That is compatible with the capture overlap demonstrated by the replication. Missing raw exports prevent complete historical validation; they do not by themselves establish that the original observations were false.

## What remains supported

RAGWitness's verified quantitative EC/AA/RF/FT values and preserved figures are unaffected by this comparison. Its declared logging levels specify which artifacts are retained. Local hash chains provide consistency checks, and retained valid RFC 3161 tokens provide independent binding of event bytes under the specified TSA trust assumptions. A fully regenerated unanchored chain can evade local consistency checks; authentication of separate configuration/corpus artifacts requires its own protection. These are bounded, useful preservation/verification properties.

The tested Langfuse configurations captured query, retrieval and prompt information when instrumented. The enhanced configuration captured the retained model settings too. Their submitted/exported traces contained no native event chain or independent timestamp token. This is an observed protocol difference, not a comprehensive statement about every possible Langfuse integration or deployment.

## Recommended presentation awaiting approval

Replace the original empirical comparator cells with the documented replication, explicitly identify the current configurations and date, and state that Q1–Q3 coverage uses the same external analyzer and clean reference. Keep the historical report in the reproducibility archive with its original question definitions rather than silently overwriting it.

Alongside capture coverage, explain cryptographic assurance separately: available prompt content is not authenticated prompt content. Keep RW L1 consistency and RW L5 external anchoring distinct. This preserves the verified forensic point without awarding RAGWitness an information-capture advantage that the replicated artifacts do not establish.

Do not recast Table VII retrospectively as entirely 'authenticated coverage' solely to retain its old verdicts. That would require a newly defined and consistently applied rubric for every column. Do not choose an older result merely because it gives the larger apparent advantage.

The manuscript's novelty assessment still requires substantive judgment. Correct protocol verification establishes correctness; it does not by itself establish that the combination of established techniques is sufficiently novel for TIFS.

Next decision: approve the new comparison and accompanying explanation, or provide an alternative historical evidence source that establishes the two original configurations under Table VII's exact questions. No further table changes are applied until approval. Then proceed to the saved-answer ASR review, as requested one step at a time.
