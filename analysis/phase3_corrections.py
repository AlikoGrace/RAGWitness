"""
Phase 3 analysis: EC by run type (attack vs benign) + Wilson CIs on AA/recall/FPR.
Writes results to analysis/tables/phase3_results.json.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

TABLES = Path(__file__).parent / "tables"
METRICS_FILE = TABLES / "fresh_90run_metrics.json"
DETECTION_FILE = TABLES / "detection_performance_report.json"
OUT_FILE = TABLES / "phase3_results.json"


# ── Wilson score interval ─────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score 95% CI for k successes in n trials.
    Returns (lower, upper), clamped to [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def fmt(v: float) -> str:
    return f"{v:.3f}"


def main() -> None:
    metrics = json.loads(METRICS_FILE.read_text())
    detection = json.loads(DETECTION_FILE.read_text())

    levels = [1, 2, 3, 4, 5]

    # ── EC split ──────────────────────────────────────────────────────────────
    ec_attack: dict[int, dict] = {}
    ec_benign: dict[int, dict] = {}

    for level in levels:
        attack_runs = [r for r in metrics
                       if r["observability_level"] == level
                       and r["attack_type"] != "baseline"]
        benign_runs = [r for r in metrics
                       if r["observability_level"] == level
                       and r["attack_type"] == "baseline"]

        def stats(runs: list[dict]) -> dict:
            ecs = [r["evidence_completeness"] for r in runs]
            n = len(ecs)
            mean = sum(ecs) / n if n else 0.0
            sd = math.sqrt(sum((e - mean) ** 2 for e in ecs) / n) if n > 1 else 0.0
            # EC is deterministic per level (no variance at several levels) so
            # we report the actual range rather than a CI.
            return {
                "n": n,
                "mean": round(mean, 4),
                "sd": round(sd, 4),
                "min": round(min(ecs), 4) if ecs else None,
                "max": round(max(ecs), 4) if ecs else None,
                "values": sorted(set(round(e, 4) for e in ecs)),
            }

        ec_attack[level] = stats(attack_runs)
        ec_benign[level] = stats(benign_runs)

    # ── Wilson CIs on AA (binary per attack run) ──────────────────────────────
    aa_by_level: dict[int, dict] = {}
    for level in levels:
        attack_runs = [r for r in metrics
                       if r["observability_level"] == level
                       and r["attack_type"] != "baseline"]
        n = len(attack_runs)
        k = sum(1 for r in attack_runs if r["attribution_accuracy"] == 1.0)
        lo, hi = wilson_ci(k, n)
        aa_by_level[level] = {
            "n": n,
            "correct": k,
            "proportion": round(k / n, 4) if n else None,
            "wilson_ci_95": [round(lo, 4), round(hi, 4)],
        }

    # ── AA split by attack type ───────────────────────────────────────────────
    aa_direct: dict[int, dict] = {}
    aa_indirect: dict[int, dict] = {}
    for level in levels:
        for attack_type, bucket in [("direct", aa_direct), ("indirect", aa_indirect)]:
            runs = [r for r in metrics
                    if r["observability_level"] == level
                    and r["attack_type"] == attack_type]
            n = len(runs)
            k = sum(1 for r in runs if r["attribution_accuracy"] == 1.0)
            lo, hi = wilson_ci(k, n)
            bucket[level] = {
                "n": n,
                "correct": k,
                "proportion": round(k / n, 4) if n else None,
                "wilson_ci_95": [round(lo, 4), round(hi, 4)],
            }

    # ── Wilson CIs on recall and FPR from detection report ───────────────────
    recall_by_level: dict[int, dict] = {}
    fpr_by_level: dict[int, dict] = {}

    for level in levels:
        det = detection["by_level"][str(level)]
        tp, fn = det["tp"], det["fn"]
        fp, tn = det["fp"], det["tn"]

        n_attack = tp + fn
        n_benign = fp + tn

        r_lo, r_hi = wilson_ci(tp, n_attack)
        f_lo, f_hi = wilson_ci(fp, n_benign)

        recall_by_level[level] = {
            "tp": tp, "fn": fn, "n": n_attack,
            "recall": round(tp / n_attack, 4) if n_attack else None,
            "wilson_ci_95": [round(r_lo, 4), round(r_hi, 4)],
        }
        fpr_by_level[level] = {
            "fp": fp, "tn": tn, "n": n_benign,
            "fpr": round(fp / n_benign, 4) if n_benign else None,
            "wilson_ci_95": [round(f_lo, 4), round(f_hi, 4)],
        }

    # ── Assemble and write ────────────────────────────────────────────────────
    result = {
        "ec_attack": {str(l): ec_attack[l] for l in levels},
        "ec_benign": {str(l): ec_benign[l] for l in levels},
        "aa_combined": {str(l): aa_by_level[l] for l in levels},
        "aa_direct": {str(l): aa_direct[l] for l in levels},
        "aa_indirect": {str(l): aa_indirect[l] for l in levels},
        "recall": {str(l): recall_by_level[l] for l in levels},
        "fpr": {str(l): fpr_by_level[l] for l in levels},
    }

    OUT_FILE.write_text(json.dumps(result, indent=2))

    # ── Print summary ─────────────────────────────────────────────────────────
    print("=== EC — ATTACK runs ===")
    for l in levels:
        r = ec_attack[l]
        print(f"  L{l}: n={r['n']} mean={r['mean']:.4f} sd={r['sd']:.4f} "
              f"values={r['values']}")

    print("\n=== EC — BENIGN runs ===")
    for l in levels:
        r = ec_benign[l]
        print(f"  L{l}: n={r['n']} mean={r['mean']:.4f} sd={r['sd']:.4f} "
              f"values={r['values']}")

    print("\n=== AA — Combined (attack runs only) with Wilson 95% CI ===")
    for l in levels:
        r = aa_by_level[l]
        lo, hi = r["wilson_ci_95"]
        print(f"  L{l}: {r['correct']}/{r['n']} = {r['proportion']:.3f} "
              f"  CI=[{lo:.3f}, {hi:.3f}]")

    print("\n=== AA — Direct ===")
    for l in levels:
        r = aa_direct[l]
        lo, hi = r["wilson_ci_95"]
        print(f"  L{l}: {r['correct']}/{r['n']} = {r['proportion']:.3f} "
              f"  CI=[{lo:.3f}, {hi:.3f}]")

    print("\n=== AA — Indirect ===")
    for l in levels:
        r = aa_indirect[l]
        lo, hi = r["wilson_ci_95"]
        print(f"  L{l}: {r['correct']}/{r['n']} = {r['proportion']:.3f} "
              f"  CI=[{lo:.3f}, {hi:.3f}]")

    print("\n=== Recall with Wilson 95% CI ===")
    for l in levels:
        r = recall_by_level[l]
        lo, hi = r["wilson_ci_95"]
        print(f"  L{l}: TP={r['tp']} FN={r['fn']} recall={r['recall']:.3f} "
              f"  CI=[{lo:.3f}, {hi:.3f}]")

    print("\n=== FPR with Wilson 95% CI ===")
    for l in levels:
        r = fpr_by_level[l]
        lo, hi = r["wilson_ci_95"]
        print(f"  L{l}: FP={r['fp']} TN={r['tn']} FPR={r['fpr']:.3f} "
              f"  CI=[{lo:.3f}, {hi:.3f}]")

    print(f"\nWritten to {OUT_FILE}")


if __name__ == "__main__":
    main()
