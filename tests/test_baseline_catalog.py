from src.experiments.experiment_baselines import (
    _slice_baseline_catalog,
    load_baseline_catalog,
)


def test_baseline_catalog_matches_forty_run_design():
    rows = load_baseline_catalog()

    assert len(rows) == 8
    assert rows[0]["baseline_id"] == "B1"
    assert rows[-1]["baseline_id"] == "B8"
    assert len(rows) * 5 == 40


def test_baseline_slice_can_select_remaining_thirty_runs():
    rows = load_baseline_catalog()
    selected = _slice_baseline_catalog(rows, start_index=3)

    assert [row["baseline_id"] for row in selected] == ["B3", "B4", "B5", "B6", "B7", "B8"]
    assert len(selected) * 5 == 30
