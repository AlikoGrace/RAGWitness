from pathlib import Path
import json

from src.run_manager import RunManager


def test_create_run_log_finalize(tmp_path: Path):
    runs_root = tmp_path / "runs"
    rm = RunManager.create(runs_root=runs_root, prefix="t1", config={"a": 1})

    for i in range(5):
        rm.log_event("test.event", {"i": i})

    integrity = rm.finalize()

    # Verify files exist
    assert (rm.run_dir / "events.jsonl").exists()
    assert (rm.run_dir / "integrity.json").exists()
    assert (rm.run_dir / "config.json").exists()
    assert (rm.run_dir / "environment.json").exists()

    # Verify event count (create logs run.started + 5 + finalize logs run.finalized)
    # Note: finalize writes integrity.json THEN logs run.finalized
    # So seq should be 1(run.started) + 5 + 1(run.finalized) = 7
    assert integrity["event_count"] == 7

    # Verify hash chain is valid
    ver = RunManager.verify_hash_chain(rm.run_dir / "events.jsonl")
    assert ver["ok"], ver["errors"]


def test_integrity_file_hash_present(tmp_path: Path):
    runs_root = tmp_path / "runs"
    rm = RunManager.create(runs_root=runs_root, prefix="t2")
    rm.log_event("x", {"k": "v"})
    rm.finalize()

    integrity = json.loads((rm.run_dir / "integrity.json").read_text(encoding="utf-8"))
    assert "events_jsonl_sha256" in integrity
    assert len(integrity["events_jsonl_sha256"]) == 64


def test_three_runs_are_isolated(tmp_path: Path):
    runs_root = tmp_path / "runs"
    a = RunManager.create(runs_root=runs_root, prefix="A")
    b = RunManager.create(runs_root=runs_root, prefix="B")
    c = RunManager.create(runs_root=runs_root, prefix="C")

    assert a.run_dir != b.run_dir != c.run_dir
    assert a.run_dir.exists() and b.run_dir.exists() and c.run_dir.exists()

    # Each has its own events.jsonl
    assert (a.run_dir / "events.jsonl").exists()
    assert (b.run_dir / "events.jsonl").exists()
    assert (c.run_dir / "events.jsonl").exists()
