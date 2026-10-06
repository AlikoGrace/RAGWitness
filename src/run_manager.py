
from __future__ import annotations

import json
import os
import platform
import secrets
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> bytes:
    # Stable encoding so hashes are reproducible
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


@dataclass
class RunManager:
    """
    RunManager = "case file" manager for a single experimental run.

    Creates an isolated run directory under runs/<run_id>/ and writes:
      - config.json
      - environment.json (lightweight; you can extend later)
      - events.jsonl (append-only)
      - integrity.json (final SHA256 + hash chain metadata)

    Event integrity:
      - Each event includes prev_hash and event_hash (hash chaining).
      - Any modification breaks the chain and can be detected later.
    """

    run_id: str
    run_dir: Path
    seq: int = 0
    prev_hash: str = "0" * 64  # genesis hash

    @staticmethod
    def new_run_id(prefix: str = "run") -> str:
        # timestamp + short random suffix is human-friendly and unique
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = secrets.token_hex(3)  # 6 hex chars
        return f"{ts}_{prefix}_{suffix}"

    @classmethod
    def create(
        cls,
        runs_root: str | Path = "runs",
        prefix: str = "run",
        config: Optional[Dict[str, Any]] = None,
        extra_environment: Optional[Dict[str, Any]] = None,
    ) -> "RunManager":
        runs_root = Path(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)

        run_id = cls.new_run_id(prefix=prefix)
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        # Standard folders
        artifacts = run_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)

        rm = cls(run_id=run_id, run_dir=run_dir)

        # Write config + environment upfront
        rm.write_json("config.json", config or {})
        rm.write_json("environment.json", rm._build_environment(extra_environment or {}))

        # Initialize events log
        (run_dir / "events.jsonl").touch(exist_ok=True)

        # First event
        rm.log_event("run.started", {"run_id": run_id})
        return rm

    def _build_environment(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        env = {
            "ts_utc": _utc_now_iso(),
            "platform": platform.platform(),
            "python_executable": os.environ.get("VIRTUAL_ENV", "") + "/bin/python3" if os.environ.get("VIRTUAL_ENV") else "",
            "cwd": str(Path.cwd()),
        }
        env.update(extra)
        return env

    def path(self, filename: str) -> Path:
        return self.run_dir / filename

    def write_json(self, filename: str, obj: Any) -> None:
        p = self.path(filename)
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def log_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Append one event to events.jsonl with an integrity hash chain.
        """
        self.seq += 1
        record: Dict[str, Any] = {
            "ts_utc": _utc_now_iso(),
            "run_id": self.run_id,
            "seq": self.seq,
            "event": event,
            "payload": payload or {},
            "prev_hash": self.prev_hash,
        }
        # Compute hash over canonical JSON (without event_hash)
        record_bytes = _canonical_json(record)
        event_hash = _sha256_hex(record_bytes)
        record["event_hash"] = event_hash

        # Append line
        with self.path("events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Update chain
        self.prev_hash = event_hash
        return record

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the run:
          - append a final event (run.finalized)
          - compute SHA256 over entire events.jsonl (file hash)
          - store integrity.json containing final_event_hash + file hash + event count

        This ensures integrity.json corresponds to the final on-disk events.jsonl.
        """
        # 1) Append final event first (this increments seq and updates prev_hash)
        self.log_event("run.finalized", {"status": "ok"})

        # 2) Now compute hash of the final events.jsonl
        events_path = self.path("events.jsonl")
        data = events_path.read_bytes()
        file_hash = _sha256_hex(data)

        integrity = {
            "ts_utc": _utc_now_iso(),
            "run_id": self.run_id,
            "event_count": self.seq,
            "final_event_hash": self.prev_hash,
            "events_jsonl_sha256": file_hash,
            "hash_chain": "prev_hash/event_hash per event (canonical json)",
        }
        self.write_json("integrity.json", integrity)
        return integrity

    @staticmethod
    def verify_hash_chain(
        events_jsonl_path: str | Path,
        *,
        require_finalized: bool = True,
        expected_event_count: int | None = None,
        expected_final_hash: str | None = None,
    ) -> Dict[str, Any]:
        """Check event integrity and completion without trusting local metadata.

        Optional expected values must come from an independently protected
        record. Completion checks reject empty logs and ordinary truncation;
        they cannot detect a fully regenerated chain by themselves.
        """
        errors: list[str] = []
        prev: Any = "0" * 64
        count = 0
        run_id: Any = None
        last_event: Any = None
        try:
            with Path(events_jsonl_path).open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        errors.append(f"line={line_number}: event is not an object")
                        break
                    count += 1
                    if type(rec.get("seq")) is not int or rec["seq"] != count:
                        errors.append(f"line={line_number}: noncontiguous sequence")
                    if count == 1:
                        run_id = rec.get("run_id")
                        if not isinstance(run_id, str) or not run_id:
                            errors.append("Missing run identifier")
                        if rec.get("event") != "run.started":
                            errors.append("Missing initial run.started event")
                    elif rec.get("run_id") != run_id:
                        errors.append(f"line={line_number}: run identifier changed")
                    if last_event == "run.finalized":
                        errors.append(f"line={line_number}: event after finalization")
                    if rec.get("prev_hash") != prev:
                        errors.append(f"seq={rec.get('seq')}: prev_hash mismatch")
                    record_for_hash = dict(rec)
                    record_for_hash.pop("event_hash", None)
                    if rec.get("event_hash") != _sha256_hex(_canonical_json(record_for_hash)):
                        errors.append(f"seq={rec.get('seq')}: event_hash mismatch")
                    prev = rec.get("event_hash")
                    last_event = rec.get("event")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"Unreadable event log: {exc}")
        if count == 0:
            errors.append("Empty event log")
        if require_finalized and last_event != "run.finalized":
            errors.append("Missing final run.finalized event")
        if expected_event_count is not None and count != expected_event_count:
            errors.append("Protected expected event count mismatch")
        if expected_final_hash is not None and prev != expected_final_hash:
            errors.append("Protected expected final hash mismatch")
        return {
            "ok": not errors, "event_count": count,
            "errors": errors, "final_event_hash": prev,
        }
