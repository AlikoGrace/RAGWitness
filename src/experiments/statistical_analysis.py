"""
Statistical analysis for RAGWitness observability experiments.

Produces:
  - Descriptive stats (mean, SD, 95% CI) per metric per level
  - Paired t-tests + Cohen's d between adjacent levels (L1-L2, L2-L3, ...)
  - Pearson correlations between metrics
  - Storage overhead normalised relative to L1

Design note: the same N attack scenarios are run at every observability level,
making this a repeated-measures / paired design.  Paired t-tests are therefore
appropriate and more powerful than independent-samples tests for this data.
"""
from __future__ import annotations

import json
from collections import defaultdict
from json import JSONDecodeError
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Metrics analysed
# ---------------------------------------------------------------------------

ANALYSED_METRICS: tuple[str, ...] = (
    "evidence_completeness",
    "attribution_accuracy",
    "reconstruction_fidelity",
    "investigation_time_seconds",
    "storage_overhead_normalised",  # added by normalise_storage()
)

LEVEL_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 3), (3, 4), (4, 5)
)

COHEN_D_THRESHOLDS = {"negligible": 0.2, "small": 0.5, "medium": 0.8}


def _interpret_d(d: float) -> str:
    d = abs(d)
    if d < COHEN_D_THRESHOLDS["negligible"]:
        return "negligible"
    if d < COHEN_D_THRESHOLDS["small"]:
        return "small"
    if d < COHEN_D_THRESHOLDS["medium"]:
        return "medium"
    return "large"


def _interpret_p(p: float) -> str:
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return "p<0.01"
    if p < 0.05:
        return "p<0.05"
    return f"p={p:.3f} (ns)"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(runs_root: str | Path = "runs") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in Path(runs_root).glob("*/metrics.json"):
        raw = metrics_path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except JSONDecodeError:
            continue
    return rows


def load_metrics_from_file(path: str | Path) -> list[dict[str, Any]]:
    """Load from a pre-aggregated metrics JSON file (list of records)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalise_storage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add storage_overhead_normalised = storage_bytes / mean(L1 storage_bytes).
    Reports how much larger each run's evidence footprint is relative to L1.
    """
    l1_bytes = [
        float(r["storage_bytes"])
        for r in rows
        if r.get("observability_level") == 1 and r.get("storage_bytes")
    ]
    if not l1_bytes:
        for r in rows:
            r["storage_overhead_normalised"] = None
        return rows
    l1_mean = mean(l1_bytes)
    for r in rows:
        sb = r.get("storage_bytes")
        r["storage_overhead_normalised"] = float(sb) / l1_mean if sb else None
    return rows


# ---------------------------------------------------------------------------
# Descriptive statistics per level
# ---------------------------------------------------------------------------

def _ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    sd = float(np.std(values, ddof=1))
    if sd < 1e-10:
        # Constant — CI collapses to the point estimate.
        m = float(np.mean(values))
        return (m, m)
    m = np.mean(values)
    se = stats.sem(values)
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
    return (float(lo), float(hi))


def compute_level_stats(
    rows: list[dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    """Return descriptive stats for one metric across all observability levels."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        level = r.get("observability_level")
        val = r.get(metric)
        if level is not None and val is not None:
            grouped[int(level)].append(float(val))

    result = []
    for level in sorted(grouped):
        vals = grouped[level]
        n = len(vals)
        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        lo, hi = _ci95(vals)
        result.append(
            {
                "level": level,
                "n": n,
                "mean": round(m, 4),
                "sd": round(sd, 4),
                "ci95_lo": round(lo, 4),
                "ci95_hi": round(hi, 4),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Paired t-tests + Cohen's d between adjacent levels
# ---------------------------------------------------------------------------

def _get_paired_values(
    rows: list[dict[str, Any]],
    level_a: int,
    level_b: int,
    metric: str,
) -> tuple[list[float], list[float]]:
    """
    Return (values_a, values_b) paired by attack_id, sorted consistently.
    Drops pairs where either value is missing.
    """
    a_map: dict[str, float] = {}
    b_map: dict[str, float] = {}
    for r in rows:
        lvl = r.get("observability_level")
        val = r.get(metric)
        aid = r.get("attack_id", r.get("run_id", ""))
        if val is None:
            continue
        if lvl == level_a:
            a_map[aid] = float(val)
        elif lvl == level_b:
            b_map[aid] = float(val)

    common = sorted(set(a_map) & set(b_map))
    return [a_map[k] for k in common], [b_map[k] for k in common]


def _cohens_d_paired(a: list[float], b: list[float]) -> float:
    """Paired Cohen's d = mean(diff) / std(diff, ddof=1)."""
    diffs = [bv - av for av, bv in zip(a, b)]
    sd = float(np.std(diffs, ddof=1))
    return float(np.mean(diffs)) / sd if sd > 0 else 0.0


def compute_adjacent_level_tests(
    rows: list[dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    """
    Paired t-test + Cohen's d for each adjacent level pair on one metric.
    Returns a list of comparison records.
    """
    results = []
    for level_a, level_b in LEVEL_PAIRS:
        vals_a, vals_b = _get_paired_values(rows, level_a, level_b, metric)
        n = len(vals_a)
        if n < 2:
            results.append(
                {
                    "comparison": f"L{level_a}_vs_L{level_b}",
                    "n_pairs": n,
                    "mean_a": None,
                    "mean_b": None,
                    "mean_diff": None,
                    "t_stat": None,
                    "p_value": None,
                    "significant": None,
                    "cohens_d": None,
                    "effect_size": None,
                    "note": "insufficient paired observations",
                }
            )
            continue

        diffs = [bv - av for av, bv in zip(vals_a, vals_b)]
        diff_sd = float(np.std(diffs, ddof=1))
        mean_diff = float(np.mean(vals_b)) - float(np.mean(vals_a))

        if diff_sd < 1e-10:
            # All differences are identical — the improvement is constant across
            # every scenario at this level transition. This is a deterministic
            # effect: no variance remains to test. Report the constant shift.
            results.append(
                {
                    "comparison": f"L{level_a}_vs_L{level_b}",
                    "n_pairs": n,
                    "mean_a": round(float(np.mean(vals_a)), 4),
                    "mean_b": round(float(np.mean(vals_b)), 4),
                    "mean_diff": round(mean_diff, 4),
                    "t_stat": None,
                    "p_value": None,
                    "p_label": "constant_effect",
                    "significant": True,
                    "cohens_d": None,
                    "effect_size": "constant",
                    "note": (
                        f"Every scenario improved by exactly {mean_diff:+.4f}. "
                        "Zero variance in differences — parametric test not applicable. "
                        "The effect is deterministic at this level transition."
                    ),
                }
            )
            continue

        t_stat, p_value = stats.ttest_rel(vals_a, vals_b)
        d = _cohens_d_paired(vals_a, vals_b)

        results.append(
            {
                "comparison": f"L{level_a}_vs_L{level_b}",
                "n_pairs": n,
                "mean_a": round(float(np.mean(vals_a)), 4),
                "mean_b": round(float(np.mean(vals_b)), 4),
                "mean_diff": round(mean_diff, 4),
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_value), 6),
                "p_label": _interpret_p(float(p_value)),
                "significant": bool(float(p_value) < 0.05),
                "cohens_d": round(d, 4),
                "effect_size": _interpret_d(d),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def compute_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Pearson correlations between key metric pairs across all runs.
    Pairs: (EC, AA), (EC, RF), (storage_overhead_normalised, RF).
    """
    pairs = [
        ("evidence_completeness", "attribution_accuracy"),
        ("evidence_completeness", "reconstruction_fidelity"),
        ("storage_overhead_normalised", "reconstruction_fidelity"),
        ("storage_overhead_normalised", "evidence_completeness"),
    ]
    results = []
    for x_key, y_key in pairs:
        xs, ys = [], []
        for r in rows:
            xv = r.get(x_key)
            yv = r.get(y_key)
            if xv is not None and yv is not None:
                xs.append(float(xv))
                ys.append(float(yv))
        if len(xs) < 3:
            results.append({"pair": f"{x_key} × {y_key}", "n": len(xs), "r": None, "p": None})
            continue
        r_val, p_val = stats.pearsonr(xs, ys)
        results.append(
            {
                "pair": f"{x_key} × {y_key}",
                "n": len(xs),
                "r": round(float(r_val), 4),
                "p_value": round(float(p_val), 6),
                "p_label": _interpret_p(float(p_val)),
                "significant": bool(float(p_val) < 0.05),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Full statistical report builder
# ---------------------------------------------------------------------------

def build_statistical_report(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the complete statistical report dict from a list of metric records.
    Normalises storage first, then computes all statistics.
    """
    rows = normalise_storage(list(rows))  # add storage_overhead_normalised

    descriptive: dict[str, list[dict]] = {}
    pairwise: dict[str, list[dict]] = {}

    for metric in ANALYSED_METRICS:
        descriptive[metric] = compute_level_stats(rows, metric)
        pairwise[metric] = compute_adjacent_level_tests(rows, metric)

    # L1 vs L5 summary (most paper-relevant comparison)
    l1_vs_l5: dict[str, Any] = {}
    for metric in ("evidence_completeness", "attribution_accuracy", "reconstruction_fidelity"):
        vals_1, vals_5 = _get_paired_values(rows, 1, 5, metric)
        if len(vals_1) < 2:
            continue
        diffs = [b - a for a, b in zip(vals_1, vals_5)]
        diff_sd = float(np.std(diffs, ddof=1))
        mean_diff = float(np.mean(vals_5)) - float(np.mean(vals_1))
        if diff_sd < 1e-10:
            l1_vs_l5[metric] = {
                "mean_L1": round(float(np.mean(vals_1)), 4),
                "mean_L5": round(float(np.mean(vals_5)), 4),
                "mean_diff": round(mean_diff, 4),
                "t_stat": None,
                "p_value": None,
                "p_label": "constant_effect",
                "cohens_d": None,
                "effect_size": "constant",
                "note": (
                    f"Every scenario improved by exactly {mean_diff:+.4f}. "
                    "Deterministic effect — zero variance in differences."
                ),
            }
        else:
            t_stat, p_value = stats.ttest_rel(vals_1, vals_5)
            d = _cohens_d_paired(vals_1, vals_5)
            l1_vs_l5[metric] = {
                "mean_L1": round(float(np.mean(vals_1)), 4),
                "mean_L5": round(float(np.mean(vals_5)), 4),
                "mean_diff": round(mean_diff, 4),
                "t_stat": round(float(t_stat), 4),
                "p_value": round(float(p_value), 6),
                "p_label": _interpret_p(float(p_value)),
                "cohens_d": round(d, 4),
                "effect_size": _interpret_d(d),
            }

    correlations = compute_correlations(rows)

    # Summary metrics by attack type (preserved from original)
    by_type: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        at = r.get("attack_type", "unknown")
        for m in ("evidence_completeness", "attribution_accuracy", "reconstruction_fidelity"):
            v = r.get(m)
            if v is not None:
                by_type[at][m].append(float(v))
    attack_type_summary = {
        at: {m: round(mean(vals), 4) for m, vals in metrics.items()}
        for at, metrics in by_type.items()
    }

    return {
        "n_runs": len(rows),
        "levels_present": sorted({r.get("observability_level") for r in rows if r.get("observability_level")}),
        "note": (
            "Paired t-tests: each attack scenario run at every level, so L_a and L_b "
            "observations are paired by attack_id. Cohen's d is the paired variant "
            "(mean_diff / sd_diff). Storage overhead normalised to L1 mean."
        ),
        "descriptive": descriptive,
        "pairwise_tests": pairwise,
        "l1_vs_l5_summary": l1_vs_l5,
        "correlations": correlations,
        "by_attack_type": attack_type_summary,
    }


# ---------------------------------------------------------------------------
# Legacy summarise helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def summarize_metrics(
    rows: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key)].append(row)

    summary: list[dict[str, Any]] = []
    for value, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        summary.append(
            {
                key: value,
                "runs": len(group),
                "evidence_completeness": mean(
                    float(r.get("evidence_completeness", 0.0)) for r in group
                ),
                "attribution_accuracy": mean(
                    float(r.get("attribution_accuracy", 0.0)) for r in group
                ),
                "reconstruction_fidelity": mean(
                    float(r.get("reconstruction_fidelity", 0.0)) for r in group
                ),
                "storage_overhead": mean(
                    float(r.get("storage_overhead", 0.0)) for r in group
                ),
            }
        )
    return summary


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_summary_tables(
    runs_root: str | Path = "runs",
    out_dir: str | Path = "analysis/tables",
    metrics_file: str | Path | None = None,
) -> dict[str, Path]:
    """
    Write all statistical output files.

    If metrics_file is provided, load from that pre-aggregated file instead of
    scanning runs_root.  This allows running analysis on a saved snapshot.
    """
    if metrics_file:
        rows = load_metrics_from_file(metrics_file)
    else:
        rows = load_metrics(runs_root)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Legacy summary tables (mean only — kept for backward compat)
    by_level_path = out_dir / "summary_by_observability_level.json"
    by_type_path = out_dir / "summary_by_attack_type.json"
    by_level_path.write_text(
        json.dumps(summarize_metrics(rows, "observability_level"), indent=2),
        encoding="utf-8",
    )
    by_type_path.write_text(
        json.dumps(summarize_metrics(rows, "attack_type"), indent=2),
        encoding="utf-8",
    )

    # Full statistical report
    report = build_statistical_report(rows)
    report_path = out_dir / "statistical_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "by_level": by_level_path,
        "by_attack_type": by_type_path,
        "statistical_report": report_path,
    }
