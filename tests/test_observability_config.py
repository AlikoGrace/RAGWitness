import pytest

from src.observability_config import get_observability_config


def test_observability_level_flags_progressively_increase():
    l1 = get_observability_config(1)
    assert l1.log_query
    assert l1.log_response
    assert not l1.log_retrieved_ids
    assert not l1.log_full_prompt

    l3 = get_observability_config(3)
    assert l3.log_retrieved_ids
    assert l3.log_retrieval_scores
    assert l3.log_document_metadata
    assert not l3.log_retrieved_text

    l5 = get_observability_config(5)
    assert l5.log_retrieved_text
    assert l5.log_document_hashes
    assert l5.log_full_prompt
    assert l5.log_generation_params
    assert l5.log_integrity_details


def test_invalid_observability_level_raises():
    with pytest.raises(ValueError):
        get_observability_config(0)
