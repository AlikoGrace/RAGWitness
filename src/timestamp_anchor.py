"""
RFC 3161 Timestamp Anchor for RAGWitness.

Extends the local SHA-256 hash chain with an external RFC 3161 timestamp
token, extending unkeyed local consistency checks with an external signed imprint.

Without an RFC 3161 anchor:
  - The hash chain proves ordering of events (each event hashes the previous)
  - BUT: an attacker with file system access could regenerate the whole chain
  - Limitation: "tamper-evident after local write, not chain-of-custody grade"
    (explicitly stated in THREAT_MODEL.md §4)

With an authentic retained RFC 3161 anchor:
  - The final chain hash is submitted to a trusted Timestamp Authority (TSA)
  - TSA signs it with their private key and returns a timestamp token
  - The token is cryptographically bound to both the hash AND the wall-clock time
  - Even with full file system access, an attacker cannot backdate the timestamp
  - Trust, imprint and signature verification plus protected original-anchor
    identity are required; a replaceable local token is not sufficient.

TSA used: DigiCert (http://timestamp.digicert.com) — free, widely trusted

The timestamp token (.tsr file) and its verification result are stored in
  runs/<run_id>/artifacts/rfc3161_timestamp.tsr
  runs/<run_id>/artifacts/rfc3161_verification.json
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import ssl
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    import rfc3161ng
    _RFC3161_AVAILABLE = True
except ImportError:
    _RFC3161_AVAILABLE = False

# Free public TSA endpoints (in order of preference)
_TSA_URLS = [
    "http://timestamp.digicert.com",
    "http://timestamp.sectigo.com",
    "http://tsa.starfieldtech.com",
]

_TIMEOUT_SECONDS = 15


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_timestamp(data: bytes, tsa_url: str = _TSA_URLS[0]) -> bytes:
    """
    Request an RFC 3161 timestamp token for `data` from the TSA.
    Returns the DER-encoded TimeStampToken bytes.
    """
    if not _RFC3161_AVAILABLE:
        raise RuntimeError(
            "rfc3161ng is not installed. Run: pip install rfc3161ng"
        )
    # RemoteTimestamper handles request encoding and HTTP POST
    stamper = rfc3161ng.RemoteTimestamper(
        tsa_url,
        hashname="sha256",
        include_tsa_certificate=True,
        timeout=_TIMEOUT_SECONDS,
    )
    # return_tsr=False → returns DER-encoded TimeStampToken bytes
    tst_bytes = stamper(data=data, return_tsr=False)
    return tst_bytes


def verify_timestamp_token(
    tst_bytes: bytes,
    data: bytes,
    *,
    ca_file: str | Path | None = None,
    policy_oid: str | None = None,
    openssl_executable: str = "openssl",
) -> dict[str, Any]:
    """Verify signature, imprint and signer path against independent trust.

    Embedded certificates supply only untrusted chain material. The trust
    bundle is explicitly supplied, selected with RAGWITNESS_TSA_CA_FILE, or
    taken from Python/OpenSSL's system trust configuration. Policy defaults
    to the DigiCert endpoint used by this pipeline and can be explicitly
    configured for another TSA. Verification uses current certificate
    validity and does not establish historical revocation or archive custody.
    """
    trust_path = ca_file or os.environ.get("RAGWITNESS_TSA_CA_FILE")
    if trust_path is None:
        trust_path = ssl.get_default_verify_paths().cafile
    policy = policy_oid or os.environ.get(
        "RAGWITNESS_TSA_POLICY_OID", "2.16.840.1.114412.7.1"
    )
    base = {
        "ok": False,
        "ca_verified": False,
        "signature_verified": False,
        "message_imprint_matched": False,
        "policy_verified": False,
        "hash_algorithm": "sha256",
        "policy_oid": policy,
        "token_sha256": hashlib.sha256(tst_bytes).hexdigest(),
        "data_sha256": hashlib.sha256(data).hexdigest(),
        "revocation_checked": False,
    }
    if not trust_path or not Path(trust_path).is_file():
        return {**base, "error": "Independent CA trust bundle is unavailable"}
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", policy):
        return {**base, "error": "Invalid expected TSA policy OID"}
    trust_path = Path(trust_path).resolve()
    try:
        base.update(
            trust_store=str(trust_path),
            trust_store_sha256=hashlib.sha256(trust_path.read_bytes()).hexdigest(),
        )
        with tempfile.TemporaryDirectory(prefix="ragwitness-ts-verify-") as tmp:
            root = Path(tmp)
            token = root / "token.der"
            original = root / "data.bin"
            chain = root / "untrusted.pem"
            token.write_bytes(tst_bytes)
            original.write_bytes(data)
            env = {**os.environ, "LC_ALL": "C"}

            def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [openssl_executable, *arguments], capture_output=True,
                    text=True, timeout=15, check=False, env=env,
                )

            extraction = run([
                "pkcs7", "-inform", "DER", "-in", str(token),
                "-print_certs", "-out", str(chain),
            ])
            if extraction.returncode:
                return {**base, "error": "Invalid timestamp token", "details": extraction.stderr[-2000:]}
            verification = run([
                "ts", "-verify", "-token_in", "-in", str(token),
                "-data", str(original), "-untrusted", str(chain),
                "-CAfile", str(trust_path),
                "-x509_strict", "-tspolicy", policy,
            ])
            if verification.returncode:
                return {**base, "error": "Timestamp verification failed", "details": verification.stderr[-2000:]}
            info = run(["ts", "-reply", "-token_in", "-in", str(token), "-text"])
            if info.returncode:
                return {**base, "error": "Timestamp decoding failed", "details": info.stderr[-2000:]}
            actual_policy = re.search(r"^Policy OID: (.+)$", info.stdout, re.MULTILINE)
            if not actual_policy or actual_policy.group(1).strip() != policy:
                return {**base, "error": "Unexpected timestamp policy OID"}
            algorithm = re.search(r"^Hash Algorithm: (.+)$", info.stdout, re.MULTILINE)
            if not algorithm or algorithm.group(1).strip() != "sha256":
                return {**base, "error": "Unexpected timestamp imprint algorithm"}
            time_match = re.search(r"^Time stamp: (.+)$", info.stdout, re.MULTILINE)
            if not time_match:
                return {**base, "error": "Timestamp time is absent"}
            gen_time = datetime.strptime(time_match.group(1).strip(), "%b %d %H:%M:%S %Y GMT").replace(tzinfo=timezone.utc)
            return {
                **base, "ok": True, "ca_verified": True,
                "signature_verified": True, "message_imprint_matched": True,
                "policy_verified": True, "token_structurally_valid": True,
                "gen_time": gen_time.isoformat(),
                "certificate_validation": "current-time timestamp-signing purpose, strict X.509",
            }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {**base, "error": f"Timestamp verifier unavailable or failed: {exc}"}


def anchor_run(
    run_dir: str | Path,
    tsa_url: str = _TSA_URLS[0],
) -> dict[str, Any]:
    """
    Anchor the final hash of a run's events.jsonl with an RFC 3161 timestamp.

    Writes:
      artifacts/rfc3161_timestamp.tsr   — raw DER token
      artifacts/rfc3161_verification.json — verification result

    Returns the verification dict.
    """
    run_dir = Path(run_dir)
    events_path = run_dir / "events.jsonl"

    if not events_path.exists():
        return {"ok": False, "error": "events.jsonl not found"}

    # Hash the entire events.jsonl file (deterministic, byte-for-byte)
    file_bytes = events_path.read_bytes()
    chain_hash = hashlib.sha256(file_bytes).hexdigest()

    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    try:
        tsr_bytes = request_timestamp(file_bytes, tsa_url=tsa_url)
    except Exception as exc:
        result = {
            "ok": False,
            "error": f"TSA request failed: {exc}",
            "chain_hash_sha256": chain_hash,
            "tsa_url": tsa_url,
        }
        (artifacts_dir / "rfc3161_verification.json").write_text(
            json.dumps(result, indent=2)
        )
        return result

    # Save raw token
    tsr_path = artifacts_dir / "rfc3161_timestamp.tsr"
    tsr_path.write_bytes(tsr_bytes)

    # Verify
    verification = verify_timestamp_token(tsr_bytes, file_bytes)
    result = {
        **verification,
        "chain_hash_sha256": chain_hash,
        "tsa_url": tsa_url,
        "tsr_path": str(tsr_path),
        "tsr_b64": base64.b64encode(tsr_bytes).decode(),
    }

    (artifacts_dir / "rfc3161_verification.json").write_text(
        json.dumps(result, indent=2)
    )
    return result


def anchor_run_into_integrity(
    run_dir: str | Path,
    tsa_url: str = _TSA_URLS[0],
) -> dict[str, Any]:
    """
    RFC 3161 anchor wired into the pipeline finalize step (Phase 3 Item 11).

    Merges the token directly into integrity.json under 'rfc3161_anchor' —
    no separate .tsr file. Called automatically by rag_pipeline.py at L5.

    integrity.json after anchoring gains:
      "rfc3161_anchor": {
        "ok": true,
        "tsa_url": "...",
        "gen_time": "2026-...",
        "chain_hash_sha256": "...",
        "tsr_b64": "<base64 DER token>",
        "token_structurally_valid": true,
        "ca_verified": true
      }
    """
    run_dir = Path(run_dir)
    events_path    = run_dir / "events.jsonl"
    integrity_path = run_dir / "integrity.json"

    if not events_path.exists():
        return {"ok": False, "error": "events.jsonl not found"}

    file_bytes  = events_path.read_bytes()
    chain_hash  = hashlib.sha256(file_bytes).hexdigest()

    try:
        tsr_bytes = request_timestamp(file_bytes, tsa_url=tsa_url)
    except Exception as exc:
        anchor = {
            "ok": False,
            "error": f"TSA request failed: {exc}",
            "chain_hash_sha256": chain_hash,
            "tsa_url": tsa_url,
        }
        _merge_anchor(integrity_path, anchor)
        return anchor

    verification = verify_timestamp_token(tsr_bytes, file_bytes)
    anchor = {
        **verification,
        "chain_hash_sha256": chain_hash,
        "tsa_url": tsa_url,
        "tsr_b64": base64.b64encode(tsr_bytes).decode(),
    }
    _merge_anchor(integrity_path, anchor)
    return anchor


def _merge_anchor(integrity_path: Path, anchor: dict[str, Any]) -> None:
    """Write anchor result into integrity.json in-place."""
    existing = json.loads(integrity_path.read_text()) if integrity_path.exists() else {}
    existing["rfc3161_anchor"] = anchor
    integrity_path.write_text(json.dumps(existing, indent=2))


def anchor_run_batch(
    runs_root: str | Path = "runs",
    level_filter: int | None = 5,
    tsa_url: str = _TSA_URLS[0],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """
    Anchor a batch of runs at the specified level (default L5 — forensic).
    `limit` caps the number of TSA calls to avoid rate-limiting (default 3).
    """
    results = []
    count = 0
    for metrics_path in sorted(Path(runs_root).glob("*/metrics.json")):
        if count >= limit:
            break
        run_dir = metrics_path.parent
        cfg = json.loads((run_dir / "config.json").read_text())
        if level_filter is not None and int(cfg.get("observability_level", 0)) != level_filter:
            continue
        print(f"  Anchoring {run_dir.name} …", end=" ", flush=True)
        result = anchor_run(run_dir, tsa_url=tsa_url)
        result["run_id"] = cfg.get("run_id", run_dir.name)
        result["attack_id"] = cfg.get("attack_id")
        results.append(result)
        status = "OK" if result.get("ok") else f"FAIL: {result.get('error','')}"
        print(status)
        count += 1
    return results
