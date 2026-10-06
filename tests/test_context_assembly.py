from src.context_assembly import assemble_context
from src.retrieval import RetrievedChunk


def test_assemble_context_preserves_chunk_ids_and_hashes_prompt():
    chunk = RetrievedChunk(
        rank=1,
        chunk_id="chunk-1",
        distance=0.1,
        metadata={},
        text="Education evidence text",
    )

    context = assemble_context("What about education?", [chunk], max_chars_per_chunk=12)

    assert "chunk-1" in context.user_prompt
    assert "Education ev" in context.user_prompt
    assert len(context.prompt_sha256) == 64
    assert context.evidence_blocks == [{"chunk_id": "chunk-1", "text": "Education ev"}]
