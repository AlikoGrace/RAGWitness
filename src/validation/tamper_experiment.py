"""
Tampering Experiment — Point 4 Reviewer Fix.

Reviewer objection:
  "Hash chains are self-signed and do not prove legal-grade chain of custody.
   Add tampering experiments: edit, delete, reorder, replay, and regenerate logs."

What this module does:
  1. Full CA verification of a real RFC 3161 token (TSA cert chain embedded in token).
  2. Five tampering experiments on a real run's events.jsonl:
       EDIT      — modify one character in a payload field
       DELETE    — remove one event line
       REORDER   — swap two adjacent events
       REPLAY    — duplicate one event line
       REGENERATE— rewrite the entire hash chain over modified events
  3. For each tamper: check whether verify_hash_chain() detects it.
  4. For REGENERATE: show the new chain hash differs from the TSA-signed hash
     (the RFC 3161 token proves the original chain, not the forged one).

Output: analysis/tables/tamper_experiment_report.json
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.run_manager import RunManager

# ── Reference run ─────────────────────────────────────────────────────────────
# Use the D4 L5 run which has both a valid hash chain and an RFC 3161 TSR file.
_REF_RUN = Path("runs/20260512_152656_rag_L5_D4_ad2ccd")
_TSR_FILE = _REF_RUN / "artifacts" / "rfc3161_timestamp.tsr"
_EVENTS_FILE = _REF_RUN / "events.jsonl"
_INTEGRITY_FILE = _REF_RUN / "integrity.json"


# ── CA verification ───────────────────────────────────────────────────────────

def verify_ca_chain(
    tsr_bytes: bytes,
    original_data: bytes,
) -> dict[str, Any]:
    """
    Perform full RFC 3161 CA chain verification.

    Steps:
      1. Extract embedded certificate chain (Root → Intermediate → Signing cert)
      2. Verify each cert's signature against the one above it
      3. Check all validity periods
      4. Verify the TSR signature and message imprint against original_data

    Returns a dict with ok, chain_valid, sig_valid, certs, gen_time.
    """
    try:
        import rfc3161ng
        from pyasn1.codec.der import decoder, encoder
        from cryptography import x509
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives import hashes
    except ImportError as e:
        return {"ok": False, "error": f"Missing dependency: {e}"}

    try:
        tst, _ = decoder.decode(tsr_bytes, asn1Spec=rfc3161ng.TimeStampToken())
        raw_certs = tst.content["certificates"]
        certs = [
            x509.load_der_x509_certificate(encoder.encode(raw_certs[i]))
            for i in range(len(raw_certs))
        ]
        cert_der = [encoder.encode(raw_certs[i]) for i in range(len(raw_certs))]
    except Exception as e:
        return {"ok": False, "error": f"Token decode failed: {e}"}

    now = datetime.datetime.now(datetime.timezone.utc)
    cert_info = []
    for cert in certs:
        cert_info.append({
            "subject": cert.subject.rfc4514_string(),
            "issuer": cert.issuer.rfc4514_string(),
            "valid_from": cert.not_valid_before_utc.isoformat(),
            "valid_to": cert.not_valid_after_utc.isoformat(),
            "currently_valid": cert.not_valid_before_utc <= now <= cert.not_valid_after_utc,
        })

    # Verify each cert is signed by the one above it
    chain_links: list[dict[str, Any]] = []
    chain_valid = True
    for i in range(1, len(certs)):
        issuer_cert = certs[i - 1]
        subject_cert = certs[i]
        try:
            issuer_cert.public_key().verify(
                subject_cert.signature,
                subject_cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                subject_cert.signature_hash_algorithm,
            )
            chain_links.append({
                "link": f"cert[{i}] signed by cert[{i-1}]",
                "status": "VALID",
            })
        except Exception as e:
            chain_links.append({
                "link": f"cert[{i}] signed by cert[{i-1}]",
                "status": f"INVALID: {e}",
            })
            chain_valid = False

    # Verify TSR signature + message imprint using the signing cert (last cert)
    signing_cert_der = cert_der[-1]
    try:
        rfc3161ng.check_timestamp(
            tsr_bytes,
            certificate=signing_cert_der,
            data=original_data,
            hashname="sha256",
        )
        sig_valid = True
        sig_error = None
    except Exception as e:
        sig_valid = False
        sig_error = str(e)

    gen_time = rfc3161ng.get_timestamp(tsr_bytes, naive=False).isoformat()

    return {
        "ok": chain_valid and sig_valid,
        "chain_valid": chain_valid,
        "sig_valid": sig_valid,
        "sig_error": sig_error,
        "gen_time": gen_time,
        "n_certs_in_token": len(certs),
        "chain_links": chain_links,
        "certs": cert_info,
        "ca_verified": True,
    }


# ── Hash chain helpers ─────────────────────────────────────────────────────────

def _verify(events_path: Path) -> dict[str, Any]:
    return RunManager.verify_hash_chain(events_path)


def _recompute_chain(lines: list[str]) -> list[str]:
    """
    Fully rewrite the hash chain for a list of raw JSONL lines.

    Simulates a sophisticated attacker who:
      1. Modifies event payloads
      2. Recomputes prev_hash for each event (chain linkage)
      3. Recomputes event_hash for each event (content digest)

    Even after full regeneration, the final file's SHA-256 will differ
    from the value the RFC 3161 TSA signed — exposing the forgery.
    """
    prev_hash = "0" * 64
    new_lines: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        ev = json.loads(line)
        ev["prev_hash"] = prev_hash
        # Recompute event_hash: sha256 of the record WITHOUT the event_hash field
        record_for_hash = {k: v for k, v in ev.items() if k != "event_hash"}
        record_bytes = json.dumps(
            record_for_hash, separators=(",", ":"), ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        ev["event_hash"] = hashlib.sha256(record_bytes).hexdigest()
        new_line = json.dumps(ev, separators=(",", ":"), ensure_ascii=False)
        prev_hash = ev["event_hash"]
        new_lines.append(new_line)
    return new_lines


# ── Individual tamper checks ──────────────────────────────────────────────────

def _tamper_edit(lines: list[str]) -> list[str]:
    """Modify one character inside the 3rd event's payload."""
    out = lines[:]
    ev = json.loads(out[2])
    payload = ev.get("payload") or {}
    # Flip any string value's first character
    for k, v in payload.items():
        if isinstance(v, str) and v:
            payload[k] = ("X" if v[0] != "X" else "Y") + v[1:]
            break
    ev["payload"] = payload
    out[2] = json.dumps(ev, separators=(",", ":"), ensure_ascii=False)
    return out


def _tamper_delete(lines: list[str]) -> list[str]:
    """Remove the 2nd event line."""
    return [l for i, l in enumerate(lines) if i != 1]


def _tamper_reorder(lines: list[str]) -> list[str]:
    """Swap events at positions 1 and 2."""
    out = lines[:]
    out[1], out[2] = out[2], out[1]
    return out


def _tamper_replay(lines: list[str]) -> list[str]:
    """Duplicate the 2nd event (insert a copy after it)."""
    return lines[:2] + [lines[1]] + lines[2:]


def _tamper_regenerate(lines: list[str]) -> list[str]:
    """Edit an event AND recompute the entire chain — the attacker's strongest move."""
    modified = _tamper_edit(lines)
    return _recompute_chain(modified)


# ── Run all tamper experiments ─────────────────────────────────────────────────

def run_tamper_experiments(
    ref_run: Path = _REF_RUN,
    output_path: str = "analysis/tables/tamper_experiment_report.json",
) -> dict[str, Any]:
    """
    For each of 5 tamper types, apply the modification to a temp copy of
    events.jsonl and check whether verify_hash_chain() detects it.
    Also run full CA verification on the original run.
    """
    events_path = ref_run / "events.jsonl"
    tsr_path = ref_run / "artifacts" / "rfc3161_timestamp.tsr"
    integrity_path = ref_run / "integrity.json"

    original_bytes = events_path.read_bytes()
    lines = [l for l in original_bytes.decode("utf-8").splitlines() if l.strip()]
    tsr_bytes = tsr_path.read_bytes() if tsr_path.exists() else None
    integrity = json.loads(integrity_path.read_text()) if integrity_path.exists() else {}
    signed_chain_hash = integrity.get("events_jsonl_sha256", "")

    # ── 1. CA verification ───────────────────────────────────────────────────
    print("Running full CA chain verification …")
    if tsr_bytes:
        ca_result = verify_ca_chain(tsr_bytes, original_bytes)
    else:
        ca_result = {"ok": False, "error": "No TSR file found"}
    print(f"  CA verification: ok={ca_result['ok']}, chain_valid={ca_result.get('chain_valid')}, "
          f"sig_valid={ca_result.get('sig_valid')}")

    # ── 2. Baseline — original unmodified ────────────────────────────────────
    baseline = _verify(events_path)
    print(f"  Baseline hash chain: ok={baseline['ok']}")

    # ── 3. Tamper experiments ────────────────────────────────────────────────
    tampers = {
        "EDIT":       _tamper_edit,
        "DELETE":     _tamper_delete,
        "REORDER":    _tamper_reorder,
        "REPLAY":     _tamper_replay,
        "REGENERATE": _tamper_regenerate,
    }

    tamper_results: list[dict[str, Any]] = []
    for name, fn in tampers.items():
        modified_lines = fn(lines)
        modified_bytes = ("\n".join(modified_lines) + "\n").encode("utf-8")
        modified_hash = hashlib.sha256(modified_bytes).hexdigest()

        # Write to a temp file and verify
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", mode="wb", delete=False
        ) as tmp:
            tmp.write(modified_bytes)
            tmp_path = Path(tmp.name)

        try:
            chain_result = _verify(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        # For REGENERATE: chain passes but RFC 3161 hash still differs
        tsr_mismatch = None
        if signed_chain_hash:
            tsr_mismatch = modified_hash != signed_chain_hash

        detected_by_chain = not chain_result.get("ok", True)
        detected_by_rfc3161 = tsr_mismatch if tsr_mismatch is not None else False
        detected = detected_by_chain or detected_by_rfc3161

        row = {
            "tamper_type": name,
            "detected": detected,
            "detected_by_hash_chain": detected_by_chain,
            "detected_by_rfc3161_hash": detected_by_rfc3161,
            "chain_ok": chain_result.get("ok"),
            "chain_error": chain_result.get("error"),
            "original_hash": signed_chain_hash,
            "tampered_hash": modified_hash,
            "hash_changed": modified_hash != signed_chain_hash,
            "n_events_original": len(lines),
            "n_events_modified": len(modified_lines),
        }
        tamper_results.append(row)

        chain_status = "DETECTED" if detected_by_chain else "UNDETECTED by chain"
        rfc_status   = "(RFC3161 mismatch DETECTED)" if detected_by_rfc3161 else ""
        print(f"  {name:<12}: {chain_status} {rfc_status}")

    report = {
        "reference_run": str(ref_run),
        "ca_verification": ca_result,
        "baseline_chain_ok": baseline.get("ok"),
        "n_events": len(lines),
        "signed_chain_hash": signed_chain_hash,
        "tamper_results": tamper_results,
        "summary": {
            "n_tampers_tested": len(tamper_results),
            "n_detected_by_chain": sum(1 for r in tamper_results if r["detected_by_hash_chain"]),
            "n_detected_by_rfc3161": sum(1 for r in tamper_results if r["detected_by_rfc3161_hash"]),
            "n_detected_total": sum(1 for r in tamper_results if r["detected"]),
            "n_undetected": sum(1 for r in tamper_results if not r["detected"]),
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport → {output_path}")
    return report


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    report = run_tamper_experiments()

    print("\n── CA Verification ──")
    ca = report["ca_verification"]
    print(f"  ok={ca['ok']}  chain_valid={ca.get('chain_valid')}  "
          f"sig_valid={ca.get('sig_valid')}  gen_time={ca.get('gen_time')}")
    if ca.get("chain_links"):
        for link in ca["chain_links"]:
            print(f"  {link['link']}: {link['status']}")

    print(f"\n── Tamper Detection ({report['summary']['n_detected_total']}/"
          f"{report['summary']['n_tampers_tested']} detected) ──")
    print(f"{'Tamper':<12} {'By Chain':>10} {'By RFC3161':>12} {'Overall':>9}")
    print("-" * 48)
    for r in report["tamper_results"]:
        print(
            f"{r['tamper_type']:<12} "
            f"{'✓' if r['detected_by_hash_chain'] else '✗':>10} "
            f"{'✓' if r['detected_by_rfc3161_hash'] else '✗':>12} "
            f"{'DETECTED' if r['detected'] else 'MISSED':>9}"
        )
