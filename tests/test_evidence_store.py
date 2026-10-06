
from pathlib import Path
import sqlite3

from src.run_manager import RunManager
from src.evidence_store import EvidenceStore


def test_ingest_jsonl_into_sqlite(tmp_path: Path):
    # Create a run
    rm = RunManager.create(runs_root=tmp_path / "runs", prefix="evstore_test")
    rm.log_event("alpha", {"x": 1})
    rm.log_event("beta", {"y": 2})
    rm.finalize()

    # Init DB + ingest
    db_path = tmp_path / "evidence" / "consolidated.db"
    store = EvidenceStore(db_path=db_path)
    store.init_db()

    run_id = store.ingest_run(rm.run_dir)
    assert run_id == rm.run_id
    assert store.count_events(run_id) >= 1


def test_query_reconstructions_table_exists(tmp_path: Path):
    db_path = tmp_path / "evidence" / "consolidated.db"
    store = EvidenceStore(db_path=db_path)
    store.init_db()

    with store.connect() as con:
        # table exists if this query doesn't error
        con.execute("SELECT COUNT(*) FROM reconstructions").fetchone()


def test_foreign_keys_enforced(tmp_path: Path):
    db_path = tmp_path / "evidence" / "consolidated.db"
    store = EvidenceStore(db_path=db_path)
    store.init_db()

    with store.connect() as con:
        # This should fail because run_id doesn't exist in runs table
        try:
            con.execute(
                "INSERT INTO events(run_id, seq, ts_utc, event, payload_json, prev_hash, event_hash) VALUES(?,?,?,?,?,?,?)",
                ("nonexistent_run", 1, "2026-01-01T00:00:00Z", "x", "{}", "0"*64, "1"*64)
            )
            con.commit()
            assert False, "Expected FOREIGN KEY constraint failure"
        except sqlite3.IntegrityError:
            pass
