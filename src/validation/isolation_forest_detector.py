"""
IsolationForest-based anomaly detection for RAGWitness Phase 3 Item 7.

Trains on benign (baseline) runs only, using 10 numerical features engineered
from each run's events.jsonl. Tests on:
  (a) held-in attacks  (D1–D5, I1–I5 from canonical runs/)
  (b) novel holdout    (N1–N5, never seen during training)

Reports precision/recall/F1 and compares to the regex baseline from
forensic_reconstruction.py.

Features (10 total):
  F1  query_length          chars in query text (0 if not logged)
  F2  response_length       chars in LLM answer
  F3  n_retrieved_chunks    number of chunks in topk (0 if not logged)
  F4  distance_mean         mean cosine distance of retrieved chunks
  F5  distance_variance     variance of cosine distances
  F6  distance_spread       max − min cosine distance
  F7  metadata_field_count  avg number of metadata fields per chunk
  F8  has_injection_token   1 if any SUSPICIOUS_PATTERNS match query or chunk text
  F9  n_unique_sources      number of distinct source docs in retrieved chunks
  F10 response_query_ratio  len(response) / max(len(query), 1)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.forensic_reconstruction import SUSPICIOUS_PATTERNS

# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features(run_dir: Path) -> dict[str, float]:
    """Extract 10 numerical features from a run's events.jsonl."""
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return _zero_features()

    events: list[dict[str, Any]] = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))

    query = ""
    response = ""
    topk: list[dict] = []

    for ev in events:
        payload = ev.get("payload", {})
        if ev["event"] == "query.received":
            query = str(payload.get("query", ""))
        elif ev["event"] == "generation.completed":
            response = str(payload.get("answer", ""))
        elif ev["event"] == "retrieval.completed":
            topk = payload.get("topk") or []

    # F1 query_length
    f1 = float(len(query))

    # F2 response_length
    f2 = float(len(response))

    # F3 n_retrieved_chunks
    f3 = float(len(topk))

    # F4 distance_mean, F5 distance_variance, F6 distance_spread
    distances = [float(c.get("distance", 0.0)) for c in topk if "distance" in c]
    if distances:
        f4 = float(np.mean(distances))
        f5 = float(np.var(distances))
        f6 = float(max(distances) - min(distances))
    else:
        f4 = 0.0
        f5 = 0.0
        f6 = 0.0

    # F7 metadata_field_count (avg non-null fields per chunk)
    metadata_counts = []
    for c in topk:
        meta = c.get("metadata") or {}
        metadata_counts.append(float(len([v for v in meta.values() if v is not None])))
    f7 = float(np.mean(metadata_counts)) if metadata_counts else 0.0

    # F8 has_injection_token — check query + all chunk text/metadata
    def _suspicious(text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS)

    injection_found = _suspicious(query)
    if not injection_found:
        for c in topk:
            if _suspicious(str(c.get("text", ""))):
                injection_found = True
                break
            if _suspicious(json.dumps(c.get("metadata", {}))):
                injection_found = True
                break
    f8 = 1.0 if injection_found else 0.0

    # F9 n_unique_sources
    source_ids: set[str] = set()
    for c in topk:
        meta = c.get("metadata") or {}
        src = meta.get("sha256_pdf") or meta.get("doc_id") or c.get("chunk_id", "")[:64]
        source_ids.add(src)
    f9 = float(len(source_ids)) if source_ids else float(len(topk))

    # F10 response_query_ratio
    f10 = f2 / max(f1, 1.0)

    return {
        "query_length": f1,
        "response_length": f2,
        "n_retrieved_chunks": f3,
        "distance_mean": f4,
        "distance_variance": f5,
        "distance_spread": f6,
        "metadata_field_count": f7,
        "has_injection_token": f8,
        "n_unique_sources": f9,
        "response_query_ratio": f10,
    }


def _zero_features() -> dict[str, float]:
    return {k: 0.0 for k in [
        "query_length", "response_length", "n_retrieved_chunks",
        "distance_mean", "distance_variance", "distance_spread",
        "metadata_field_count", "has_injection_token",
        "n_unique_sources", "response_query_ratio",
    ]}


FEATURE_NAMES = list(_zero_features().keys())


# ── Training and evaluation ──────────────────────────────────────────────────

def _load_runs(runs_root: Path) -> list[dict[str, Any]]:
    rows = []
    for mp in sorted(runs_root.glob("*/metrics.json")):
        cfg = json.loads((mp.parent / "config.json").read_text())
        rows.append({
            "run_dir": mp.parent,
            "attack_id": cfg.get("attack_id", ""),
            "attack_type": cfg.get("attack_type", "baseline"),
            "level": int(cfg.get("observability_level", 0)),
            "reconstructed": json.loads(mp.read_text()).get("reconstructed_attack_type", "none"),
        })
    return rows


def run_isolation_forest(
    canonical_runs: str | Path = "runs",
    novel_runs: str | Path = "runs_novel",
    output_path: str | Path = "analysis/tables/isolation_forest_report.json",
    contamination: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train IsolationForest on benign runs, evaluate on held-in attacks
    and novel holdout attacks. Compare to regex baseline.
    """
    canonical_runs = Path(canonical_runs)
    novel_runs = Path(novel_runs)
    output_path = Path(output_path)

    all_rows = _load_runs(canonical_runs)

    # ── Separate benign training set ─────────────────────────────────────────
    benign_rows = [r for r in all_rows if r["attack_type"] == "baseline"]
    attack_rows = [r for r in all_rows if r["attack_type"] != "baseline"]

    # Extract features
    benign_X = np.array([[v for v in extract_features(r["run_dir"]).values()] for r in benign_rows])
    attack_X = np.array([[v for v in extract_features(r["run_dir"]).values()] for r in attack_rows])

    # Novel holdout
    novel_rows = _load_runs(novel_runs) if novel_runs.exists() else []
    novel_X = np.array([[v for v in extract_features(r["run_dir"]).values()] for r in novel_rows]) if novel_rows else np.empty((0, 10))

    # ── Fit scaler + IsolationForest on benign only ──────────────────────────
    scaler = StandardScaler()
    scaler.fit(benign_X)

    benign_X_s = scaler.transform(benign_X)
    attack_X_s = scaler.transform(attack_X) if len(attack_X) else attack_X
    novel_X_s  = scaler.transform(novel_X)  if len(novel_X)  else novel_X

    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    clf.fit(benign_X_s)

    # ── Predict: 1=inlier(benign), -1=outlier(attack) ────────────────────────
    benign_pred = clf.predict(benign_X_s)   # expect all +1
    attack_pred = clf.predict(attack_X_s)   # expect -1
    novel_pred  = clf.predict(novel_X_s)    if len(novel_X_s) else np.array([])

    def _metrics(y_true_attack: np.ndarray, y_pred: np.ndarray) -> dict:
        """y_true_attack=1 means ground truth is attack. y_pred=-1 means predicted attack."""
        pred_attack = (y_pred == -1).astype(int)
        tp = int(np.sum((y_true_attack == 1) & (pred_attack == 1)))
        fp = int(np.sum((y_true_attack == 0) & (pred_attack == 1)))
        fn = int(np.sum((y_true_attack == 1) & (pred_attack == 0)))
        tn = int(np.sum((y_true_attack == 0) & (pred_attack == 0)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    # IsolationForest on held-in attacks + benign combined
    combined_true = np.array([0]*len(benign_rows) + [1]*len(attack_rows))
    combined_pred = np.concatenate([benign_pred, attack_pred])
    if_metrics_held_in = _metrics(combined_true, combined_pred)

    # IsolationForest on novel holdout only
    if len(novel_pred):
        novel_true = np.ones(len(novel_rows))  # all are attacks
        if_metrics_novel = _metrics(novel_true, novel_pred)
    else:
        if_metrics_novel = {"error": "no novel runs found"}

    # ── Regex baseline metrics (from existing reconstructed_attack_type) ──────
    # Regex predicts "attack" if reconstructed_attack_type != "none"
    regex_combined_pred = np.array(
        [-1 if r["reconstructed"] != "none" else 1 for r in benign_rows] +
        [-1 if r["reconstructed"] != "none" else 1 for r in attack_rows]
    )
    # -1=attack, +1=benign convention (same as IsolationForest output)
    regex_metrics_held_in = _metrics(combined_true, regex_combined_pred)

    if novel_rows:
        regex_novel_pred = np.array(
            [-1 if r["reconstructed"] != "none" else 1 for r in novel_rows]
        )
        regex_metrics_novel = _metrics(np.ones(len(novel_rows)), regex_novel_pred)
    else:
        regex_metrics_novel = {"error": "no novel runs found"}

    # ── Feature importance proxy: mean anomaly score per feature (via permutation) ──
    base_scores = clf.score_samples(benign_X_s)
    feature_importance = {}
    for i, fname in enumerate(FEATURE_NAMES):
        X_perm = benign_X_s.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        perm_scores = clf.score_samples(X_perm)
        feature_importance[fname] = round(float(np.mean(base_scores) - np.mean(perm_scores)), 6)

    report = {
        "n_benign_train": len(benign_rows),
        "n_attack_held_in": len(attack_rows),
        "n_novel_holdout": len(novel_rows),
        "features": FEATURE_NAMES,
        "contamination": contamination,
        "isolation_forest": {
            "held_in_attacks": if_metrics_held_in,
            "novel_holdout": if_metrics_novel,
        },
        "regex_baseline": {
            "held_in_attacks": regex_metrics_held_in,
            "novel_holdout": regex_metrics_novel,
        },
        "feature_importance": feature_importance,
        "per_run_novel": [
            {
                "attack_id": r["attack_id"],
                "level": r["level"],
                "if_detected": bool(novel_pred[i] == -1),
                "regex_detected": r["reconstructed"] != "none",
                "features": extract_features(r["run_dir"]),
            }
            for i, r in enumerate(novel_rows)
        ] if novel_rows else [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    return report
