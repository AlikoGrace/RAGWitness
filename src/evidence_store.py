from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  run_dir TEXT NOT NULL,
  ts_first TEXT,
  ts_last TEXT,
  environment_json TEXT,
  config_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  ts_utc TEXT NOT NULL,
  event TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  prev_hash TEXT NOT NULL,
  event_hash TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  UNIQUE(run_id, seq)
);

-- A placeholder for later phases (we'll populate once reconstruction exists)
CREATE TABLE IF NOT EXISTS reconstructions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  ts_utc TEXT NOT NULL,
  method TEXT NOT NULL,          -- e.g., "replay_v1"
  result_json TEXT NOT NULL,     -- reconstruction results
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS experiments (
  run_id TEXT PRIMARY KEY,
  attack_id TEXT,
  attack_type TEXT,
  observability_level INTEGER,
  expected_malicious_chunk_id TEXT,
  reconstructed_attack_type TEXT,
  attributed_source TEXT,
  attributed_chunk_id TEXT,
  evidence_completeness REAL,
  attribution_accuracy REAL,
  reconstruction_fidelity REAL,
  investigation_time_seconds REAL,
  storage_bytes INTEGER,
  storage_overhead REAL,
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_event ON events(event);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_experiments_level ON experiments(observability_level);
CREATE INDEX IF NOT EXISTS idx_experiments_attack_type ON experiments(attack_type);
"""

@dataclass
class EvidenceStore:
    """
    Derived analytics store for cross-run analysis.
    Primary evidence remains runs/*/events.jsonl (immutable).
    This DB is reconstructed from those logs.
    """
    db_path: Path

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA_SQL)

    def ingest_run(self, run_dir: str | Path) -> str:
        """
        Ingest one run folder into SQLite.
        Expects:
          - events.jsonl
          - optional config.json, environment.json
        Returns run_id.
        """
        run_dir = Path(run_dir)
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            raise FileNotFoundError(f"Missing events.jsonl: {events_path}")

        # Load optional metadata
        config_json = self._read_optional_json(run_dir / "config.json")
        env_json = self._read_optional_json(run_dir / "environment.json")

        # Stream events
        first_ts = None
        last_ts = None
        run_id = None
        rows: list[Tuple[Any, ...]] = []

        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                run_id = rec.get("run_id") or run_id
                ts = rec["ts_utc"]
                first_ts = ts if first_ts is None else first_ts
                last_ts = ts
                rows.append((
                    rec["run_id"],
                    rec["seq"],
                    rec["ts_utc"],
                    rec["event"],
                    json.dumps(rec.get("payload", {}), ensure_ascii=False),
                    rec["prev_hash"],
                    rec["event_hash"],
                ))

        if run_id is None:
            raise ValueError("Could not determine run_id from events.jsonl")

        # Insert into DB
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO runs(run_id, run_dir, ts_first, ts_last, environment_json, config_json)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                  run_dir=excluded.run_dir,
                  ts_first=excluded.ts_first,
                  ts_last=excluded.ts_last,
                  environment_json=excluded.environment_json,
                  config_json=excluded.config_json
                """,
                (run_id, str(run_dir), first_ts, last_ts,
                 json.dumps(env_json, ensure_ascii=False) if env_json is not None else None,
                 json.dumps(config_json, ensure_ascii=False) if config_json is not None else None)
            )

            con.executemany(
                """
                INSERT OR REPLACE INTO events(run_id, seq, ts_utc, event, payload_json, prev_hash, event_hash)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                rows
            )

        return run_id

    def count_events(self, run_id: Optional[str] = None) -> int:
        with self.connect() as con:
            if run_id:
                return con.execute("SELECT COUNT(*) FROM events WHERE run_id=?", (run_id,)).fetchone()[0]
            return con.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def list_runs(self) -> list[str]:
        with self.connect() as con:
            return [r[0] for r in con.execute("SELECT run_id FROM runs ORDER BY ts_first").fetchall()]

    def fetch_events(self, run_id: str, event: Optional[str] = None, limit: int = 50) -> list[Dict[str, Any]]:
        q = "SELECT run_id, seq, ts_utc, event, payload_json, prev_hash, event_hash FROM events WHERE run_id=?"
        params: list[Any] = [run_id]
        if event:
            q += " AND event=?"
            params.append(event)
        q += " ORDER BY seq ASC LIMIT ?"
        params.append(limit)

        out = []
        with self.connect() as con:
            for row in con.execute(q, tuple(params)).fetchall():
                out.append({
                    "run_id": row[0],
                    "seq": row[1],
                    "ts_utc": row[2],
                    "event": row[3],
                    "payload": json.loads(row[4]),
                    "prev_hash": row[5],
                    "event_hash": row[6],
                })
        return out

    def ingest_reconstruction(self, run_dir: str | Path, method: str = "reconstruct_v1") -> None:
        run_dir = Path(run_dir)
        result_path = run_dir / "reconstruction.json"
        if not result_path.exists():
            return
        result = json.loads(result_path.read_text(encoding="utf-8"))
        run_id = result.get("run_id")
        if not run_id:
            return

        ts_utc = None
        with self.connect() as con:
            row = con.execute("SELECT ts_last FROM runs WHERE run_id=?", (run_id,)).fetchone()
            ts_utc = row[0] if row else ""
            con.execute(
                """
                INSERT INTO reconstructions(run_id, ts_utc, method, result_json)
                VALUES(?, ?, ?, ?)
                """,
                (run_id, ts_utc, method, json.dumps(result, ensure_ascii=False)),
            )

    def ingest_metrics(self, run_dir: str | Path) -> None:
        run_dir = Path(run_dir)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            return
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_id = metrics.get("run_id")
        if not run_id:
            return

        with self.connect() as con:
            con.execute(
                """
                INSERT INTO experiments(
                  run_id, attack_id, attack_type, observability_level,
                  expected_malicious_chunk_id, reconstructed_attack_type,
                  attributed_source, attributed_chunk_id, evidence_completeness,
                  attribution_accuracy, reconstruction_fidelity,
                  investigation_time_seconds, storage_bytes, storage_overhead
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                  attack_id=excluded.attack_id,
                  attack_type=excluded.attack_type,
                  observability_level=excluded.observability_level,
                  expected_malicious_chunk_id=excluded.expected_malicious_chunk_id,
                  reconstructed_attack_type=excluded.reconstructed_attack_type,
                  attributed_source=excluded.attributed_source,
                  attributed_chunk_id=excluded.attributed_chunk_id,
                  evidence_completeness=excluded.evidence_completeness,
                  attribution_accuracy=excluded.attribution_accuracy,
                  reconstruction_fidelity=excluded.reconstruction_fidelity,
                  investigation_time_seconds=excluded.investigation_time_seconds,
                  storage_bytes=excluded.storage_bytes,
                  storage_overhead=excluded.storage_overhead
                """,
                (
                    run_id,
                    metrics.get("attack_id"),
                    metrics.get("attack_type"),
                    metrics.get("observability_level"),
                    metrics.get("expected_malicious_chunk_id"),
                    metrics.get("reconstructed_attack_type"),
                    metrics.get("attributed_source"),
                    metrics.get("attributed_chunk_id"),
                    metrics.get("evidence_completeness"),
                    metrics.get("attribution_accuracy"),
                    metrics.get("reconstruction_fidelity"),
                    metrics.get("investigation_time_seconds"),
                    metrics.get("storage_bytes"),
                    metrics.get("storage_overhead"),
                ),
            )

    def summarize_by_level(self) -> list[Dict[str, Any]]:
        query = """
            SELECT
              observability_level,
              COUNT(*) AS runs,
              AVG(evidence_completeness),
              AVG(attribution_accuracy),
              AVG(reconstruction_fidelity),
              AVG(storage_overhead)
            FROM experiments
            GROUP BY observability_level
            ORDER BY observability_level
        """
        with self.connect() as con:
            return [
                {
                    "observability_level": row[0],
                    "runs": row[1],
                    "evidence_completeness": row[2],
                    "attribution_accuracy": row[3],
                    "reconstruction_fidelity": row[4],
                    "storage_overhead": row[5],
                }
                for row in con.execute(query).fetchall()
            ]

    def summarize_by_attack_type(self) -> list[Dict[str, Any]]:
        query = """
            SELECT
              attack_type,
              COUNT(*) AS runs,
              AVG(evidence_completeness),
              AVG(attribution_accuracy),
              AVG(reconstruction_fidelity)
            FROM experiments
            GROUP BY attack_type
            ORDER BY attack_type
        """
        with self.connect() as con:
            return [
                {
                    "attack_type": row[0],
                    "runs": row[1],
                    "evidence_completeness": row[2],
                    "attribution_accuracy": row[3],
                    "reconstruction_fidelity": row[4],
                }
                for row in con.execute(query).fetchall()
            ]

    def _read_optional_json(self, p: Path) -> Optional[Dict[str, Any]]:
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
