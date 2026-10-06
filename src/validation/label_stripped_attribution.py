"""
Label-Stripped Attribution Experiment.

Addresses the reviewer objection (Point 3):
  "A real attacker would not label poison chunks with poisoned=True or attack_id=I1."

What this experiment does:
  1. Takes all indirect attack runs from runs/
  2. Re-runs forensic attribution with poisoned and attack_id metadata STRIPPED from
     the reconstructor's decision logic
  3. Instead uses only investigator-derivable signals:
       a. Corpus diff   — chunk_id not present in the clean Hansard corpus
       b. Source anomaly — source_url does not match the parliament.gh domain
       c. Ingestion stage — ingestion_stage field is present (clean chunks have none)
       d. Text content  — retrieved text matches SUSPICIOUS_PATTERNS (L4+)
  4. Computes AA with-labels vs. without-labels at each observability level
  5. Reports the delta — how much attribution accuracy depends on the attacker's own flags

Key finding anticipated:
  At L3+, corpus diff (chunk_id not in clean corpus) and source URL anomaly both survive
  label stripping. This shows attribution is NOT circular — it does not depend on
  self-declared attacker metadata to work.

Output: analysis/tables/label_stripped_attribution_report.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ── Constants ────────────────────────────────────────────────────────────────

CLEAN_CORPUS_JSONL = "data/processed/hansard/chunks.jsonl"
CLEAN_SOURCE_DOMAIN = "parliament.gh"
RUNS_ROOT = "runs"

SUSPICIOUS_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+override",
    r"reveal\s+the\s+system\s+prompt",
    r"attack\s+success",
    r"jailbreak",
    r"override\s+normal\s+evidence",
    r"do\s+not\s+use\s+retrieved\s+evidence",
    r"without\s+citations?",
    r"abolished\s+education\s+rights",
    r"cancelled\s+education",
    r"secretly\s+cancelled",
    r"hidden\s+unicode",
    r"zero\s+width",
]


# ── Corpus reference ──────────────────────────────────────────────────────────

def _load_clean_chunk_ids(path: str = CLEAN_CORPUS_JSONL) -> frozenset[str]:
    """Return the set of chunk IDs in the clean (unmodified) Hansard corpus."""
    ids: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            ids.add(str(row.get("chunk_id", "")))
    return frozenset(ids)


# ── Signal functions ──────────────────────────────────────────────────────────

def _is_corpus_foreign(chunk_id: str, clean_ids: frozenset[str]) -> bool:
    """True if the chunk_id is NOT in the clean corpus — i.e. was injected."""
    return bool(chunk_id) and chunk_id not in clean_ids


def _is_source_anomalous(metadata: dict[str, Any]) -> bool:
    """True if source_url is missing the official parliament.gh domain."""
    source = str(metadata.get("source_url", "") or "")
    if not source:
        return False
    return CLEAN_SOURCE_DOMAIN not in source


def _has_ingestion_stage(metadata: dict[str, Any]) -> bool:
    """True if ingestion_stage is logged — clean chunks do not carry this field."""
    return bool(metadata.get("ingestion_stage"))


def _looks_suspicious(text: str) -> bool:
    text = text or ""
    return any(
        re.search(p, text, flags=re.IGNORECASE) for p in SUSPICIOUS_PATTERNS
    )


# ── Core label-stripped detection ─────────────────────────────────────────────

def _find_suspicious_chunk_label_stripped(
    topk: list[dict[str, Any]],
    clean_ids: frozenset[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """
    Detect a suspicious chunk WITHOUT using poisoned or attack_id metadata.

    Returns (chunk_or_None, list_of_signals_fired).
    Signals (in priority order):
      text_content   — retrieved text matches injection patterns          (L4+)
      corpus_diff    — chunk_id absent from clean corpus                  (L2+)
      source_anomaly — source_url outside parliament.gh                   (L3+)
      ingestion_stage— ingestion_stage field present                      (L3+)
    """
    for row in topk:
        metadata = row.get("metadata") or {}
        chunk_id = str(row.get("chunk_id") or "")
        text = str(row.get("text") or "")
        signals: list[str] = []

        if _looks_suspicious(text):
            signals.append("text_content")
        if _is_corpus_foreign(chunk_id, clean_ids):
            signals.append("corpus_diff")
        if _is_source_anomalous(metadata):
            signals.append("source_anomaly")
        if _has_ingestion_stage(metadata):
            signals.append("ingestion_stage")

        if signals:
            return row, signals

    return None, []


# ── Original (labelled) detection — for comparison ───────────────────────────

def _find_suspicious_chunk_labelled(topk: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Original detection that reads poisoned and attack_id metadata flags."""
    for row in topk:
        metadata = row.get("metadata") or {}
        if metadata.get("poisoned") is True or str(metadata.get("poisoned", "")).lower() == "true":
            return row
        if metadata.get("attack_id"):
            return row
        text = str(row.get("text") or "")
        meta_text = json.dumps(metadata, ensure_ascii=False)
        if _looks_suspicious(text) or _looks_suspicious(meta_text):
            return row
    return None


# ── Per-run analysis ──────────────────────────────────────────────────────────

def _load_events(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    ep = run_dir / "events.jsonl"
    for line in ep.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _get_topk(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for ev in reversed(events):
        if ev.get("event") == "retrieval.completed":
            return ev.get("payload", {}).get("topk") or []
    return []


def analyse_run(
    run_dir: Path,
    clean_ids: frozenset[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute labelled and label-stripped attribution for one indirect run.

    Attribution accuracy (AA):
      1.0  — correct chunk attributed
      0.0  — wrong or no chunk attributed
    """
    expected_chunk_id = cfg.get("expected_malicious_chunk_id", "")
    level = cfg.get("observability_level")
    attack_id = cfg.get("attack_id")
    poison_retrieved = cfg.get("poison_chunk_retrieved")

    events = _load_events(run_dir)
    topk = _get_topk(events)

    # ── Labelled ────────────────────────────────────────────────────────────
    labelled_chunk = _find_suspicious_chunk_labelled(topk)
    labelled_chunk_id = labelled_chunk.get("chunk_id") if labelled_chunk else None
    labelled_aa = 1.0 if labelled_chunk_id and labelled_chunk_id == expected_chunk_id else 0.0

    # ── Label-stripped ──────────────────────────────────────────────────────
    stripped_chunk, signals = _find_suspicious_chunk_label_stripped(topk, clean_ids)
    stripped_chunk_id = stripped_chunk.get("chunk_id") if stripped_chunk else None
    stripped_aa = 1.0 if stripped_chunk_id and stripped_chunk_id == expected_chunk_id else 0.0

    return {
        "run_id": run_dir.name,
        "attack_id": attack_id,
        "observability_level": level,
        "poison_chunk_retrieved": poison_retrieved,
        "expected_chunk_id": expected_chunk_id,
        "labelled_attributed_chunk_id": labelled_chunk_id,
        "stripped_attributed_chunk_id": stripped_chunk_id,
        "labelled_aa": labelled_aa,
        "stripped_aa": stripped_aa,
        "delta_aa": stripped_aa - labelled_aa,
        "signals_fired": signals,
        "topk_chunk_ids": [r.get("chunk_id") for r in topk],
        "topk_n": len(topk),
    }


# ── Main experiment ───────────────────────────────────────────────────────────

def run_label_stripped_experiment(
    runs_root: str = RUNS_ROOT,
    clean_corpus_path: str = CLEAN_CORPUS_JSONL,
    output_path: str = "analysis/tables/label_stripped_attribution_report.json",
) -> dict[str, Any]:
    """
    Run label-stripped attribution over all indirect runs in runs_root.
    Returns (and writes) the full report dict.
    """
    print(f"Loading clean corpus chunk IDs from {clean_corpus_path} …")
    clean_ids = _load_clean_chunk_ids(clean_corpus_path)
    print(f"  {len(clean_ids):,} clean chunk IDs loaded.")

    rows: list[dict[str, Any]] = []
    for cp in sorted(Path(runs_root).glob("*/config.json")):
        cfg = json.loads(cp.read_text(encoding="utf-8"))
        if cfg.get("attack_type") != "indirect":
            continue
        row = analyse_run(cp.parent, clean_ids, cfg)
        rows.append(row)
        status = (
            f"lvl={row['observability_level']} "
            f"retrieved={row['poison_chunk_retrieved']} "
            f"labelled_aa={row['labelled_aa']:.1f} "
            f"stripped_aa={row['stripped_aa']:.1f} "
            f"signals={row['signals_fired']}"
        )
        print(f"  {row['attack_id']:4s} {row['run_id'][-6:]}  {status}")

    if not rows:
        print("No indirect runs found.")
        return {"rows": [], "summary": {}}

    # ── Per-level summary ────────────────────────────────────────────────────
    levels = sorted(set(r["observability_level"] for r in rows))
    summary: dict[str, Any] = {}
    for lvl in levels:
        subset = [r for r in rows if r["observability_level"] == lvl]
        # Only count runs where the poison chunk was actually retrieved
        retrieved = [r for r in subset if r["poison_chunk_retrieved"] is True]
        n = len(subset)
        n_ret = len(retrieved)
        summary[f"L{lvl}"] = {
            "n_total": n,
            "n_poison_retrieved": n_ret,
            "labelled_aa_mean": round(
                sum(r["labelled_aa"] for r in subset) / n, 4
            ),
            "stripped_aa_mean": round(
                sum(r["stripped_aa"] for r in subset) / n, 4
            ),
            "labelled_aa_retrieved_only": round(
                sum(r["labelled_aa"] for r in retrieved) / n_ret, 4
            ) if n_ret else None,
            "stripped_aa_retrieved_only": round(
                sum(r["stripped_aa"] for r in retrieved) / n_ret, 4
            ) if n_ret else None,
            "delta_aa_mean": round(
                sum(r["delta_aa"] for r in subset) / n, 4
            ),
            "signals_distribution": _count_signals(subset),
        }

    # ── Signal coverage summary ──────────────────────────────────────────────
    all_signals: dict[str, int] = {}
    for r in rows:
        for s in r["signals_fired"]:
            all_signals[s] = all_signals.get(s, 0) + 1

    report = {
        "n_indirect_runs": len(rows),
        "clean_corpus_ids": len(clean_ids),
        "summary_by_level": summary,
        "signal_coverage_total": all_signals,
        "rows": rows,
        "interpretation": (
            "stripped_aa_mean = AA achieved WITHOUT reading poisoned/attack_id flags. "
            "delta_aa_mean = 0 means label stripping causes no accuracy loss. "
            "Positive signals_fired at L3+ confirm attribution survives without attacker-provided labels."
        ),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written → {output_path}")
    return report


def _count_signals(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        for s in r["signals_fired"]:
            counts[s] = counts.get(s, 0) + 1
    return counts


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    report = run_label_stripped_experiment()

    print("\n── Label-Stripped Attribution Results ──\n")
    print(f"{'Level':<6} {'N':>4} {'Ret':>4} {'AA_label':>9} {'AA_strip':>9} {'ΔAA':>7}  Signals")
    print("-" * 70)
    for lvl_key, s in report["summary_by_level"].items():
        sigs = ", ".join(
            f"{k}={v}" for k, v in sorted(s["signals_distribution"].items())
        )
        print(
            f"{lvl_key:<6} {s['n_total']:>4} {s['n_poison_retrieved'] or 0:>4} "
            f"{s['labelled_aa_mean']:>9.4f} {s['stripped_aa_mean']:>9.4f} "
            f"{s['delta_aa_mean']:>+7.4f}  {sigs}"
        )

    print("\nSignal coverage (across all indirect runs):")
    for sig, cnt in sorted(report["signal_coverage_total"].items()):
        print(f"  {sig:<20} {cnt:>3} / {report['n_indirect_runs']} runs")
