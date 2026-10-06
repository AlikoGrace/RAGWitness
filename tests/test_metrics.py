from src.metrics import attribution_accuracy, evidence_completeness, reconstruction_fidelity


def test_evidence_completeness_counts_answered_questions():
    assert evidence_completeness({"a": True, "b": False, "c": True}) == 2 / 3


def test_reconstruction_fidelity_counts_available_artifacts():
    assert reconstruction_fidelity({"query": True, "prompt": False}) == 0.5


def test_attribution_accuracy_direct_attack():
    score = attribution_accuracy(
        expected_attack_type="direct",
        reconstructed_attack_type="direct",
        expected_chunk_id=None,
        attributed_chunk_id=None,
        attributed_source="user_query",
    )
    assert score == 1.0


def test_attribution_accuracy_indirect_attack():
    score = attribution_accuracy(
        expected_attack_type="indirect",
        reconstructed_attack_type="indirect",
        expected_chunk_id="poison:I1:0",
        attributed_chunk_id="poison:I1:0",
        attributed_source="retrieved_chunk",
    )
    assert score == 1.0
