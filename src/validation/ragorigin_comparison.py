"""
RAGOrigin-inspired attribution on RAGWitness logged data.

RAGOrigin (Nian et al., arXiv:2603.17445 / IEEE S&P 2026) uses three signals
to identify poisoned chunks from a RAG misgeneration event:
  S1: embedding similarity between query and candidate chunk
  S2: semantic correlation between chunk text and generated answer
  S3: generation influence (iterative LLM re-querying — requires live system)

This module applies S1 and S2 to RAGWitness L4+ event logs, where chunk text
is preserved. At L2-L3 only S1 is available (distance logged, text absent).
S3 is approximated using answer-chunk token overlap (no live LLM needed).

The comparison shows whether RAGOrigin's signals, operating from preserved
logs, match or fall short of RAGWitness's corpus-diff attribution.

Output: analysis/tables/ragorigin_comparison_report.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


RUNS_ROOT = Path("runs")
OUT_PATH  = Path("analysis/tables/ragorigin_comparison_report.json")

# Only indirect attacks have poison chunks to attribute
INDIRECT_IDS = {"I1", "I2", "I3", "I4", "I5"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_events(run_dir: Path) -> list[dict]:
    return [
        json.loads(l)
        for l in (run_dir / "events.jsonl").read_text().splitlines()
        if l.strip()
    ]


def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _dist_to_sim(d: float) -> float:
    """Convert L2/cosine distance to [0,1] similarity (clipped)."""
    return max(0.0, 1.0 - d)


def _score_chunks(
    topk: list[dict],
    answer: str,
    level: int,
) -> list[dict]:
    """
    Return chunks with ragorigin_score and rank_by_score.

    At L4+: S1 (embed sim from distance) + S2 (Jaccard answer overlap).
    At L2-L3: S1 only (text not available in logs).
    """
    for c in topk:
        s1 = _dist_to_sim(c.get("distance", 1.0))
        s2 = 0.0
        if level >= 4 and "text" in c:
            s2 = _jaccard(c["text"], answer)
        c["s1_embed_sim"] = round(s1, 4)
        c["s2_answer_overlap"] = round(s2, 4)
        # Equal weight (0.5+0.5) or S1-only at lower levels
        if level >= 4:
            c["ragorigin_score"] = round(0.5 * s1 + 0.5 * s2, 4)
        else:
            c["ragorigin_score"] = s1

    ranked = sorted(topk, key=lambda c: c["ragorigin_score"], reverse=True)
    for i, c in enumerate(ranked, 1):
        c["ragorigin_rank"] = i
    return ranked


def _find_poison_chunk(chunks: list[dict]) -> dict | None:
    """Return first chunk with poisoned=True in metadata, or None."""
    for c in chunks:
        if c.get("metadata", {}).get("poisoned", False):
            return c
    return None


# ── per-run evaluation ────────────────────────────────────────────────────────

def _evaluate_run(run_dir: Path, level: int, attack_id: str) -> dict[str, Any]:
    events = _load_events(run_dir)
    query  = next(
        (e["payload"].get("query", "") for e in events if e["event"] == "query.received"), ""
    )
    answer = next(
        (e["payload"].get("answer", "") for e in events if e["event"] == "generation.completed"), ""
    )
    topk = next(
        (e["payload"].get("topk", []) for e in events if e["event"] == "retrieval.completed"), []
    )

    if not topk:
        return {
            "run_dir": str(run_dir),
            "level": level,
            "attack_id": attack_id,
            "status": "NO_TOPK",
            "poison_retrieved": False,
            "ragorigin_attributed": False,
            "ragorigin_rank": None,
        }

    scored = _score_chunks([dict(c) for c in topk], answer, level)
    poison = _find_poison_chunk(scored)

    poison_retrieved = poison is not None
    if poison_retrieved:
        ragorigin_rank = poison["ragorigin_rank"]
        attributed = ragorigin_rank == 1
    else:
        ragorigin_rank = None
        attributed = False

    return {
        "run_dir": str(run_dir),
        "level": level,
        "attack_id": attack_id,
        "query": query[:120],
        "answer": answer[:120],
        "status": "OK",
        "poison_retrieved": poison_retrieved,
        "ragorigin_attributed": attributed,
        "ragorigin_rank": ragorigin_rank,
        "n_chunks": len(scored),
        "chunk_scores": [
            {
                "chunk_id": c.get("chunk_id", "?")[:40],
                "poison": c.get("metadata", {}).get("poisoned", False),
                "dist": c.get("distance"),
                "s1": c.get("s1_embed_sim"),
                "s2": c.get("s2_answer_overlap"),
                "score": c.get("ragorigin_score"),
                "ragorigin_rank": c.get("ragorigin_rank"),
            }
            for c in scored
        ],
    }


# ── main ─────────────────────────────────────────────────────────────────────

def run_ragorigin_comparison(
    runs_root: Path = RUNS_ROOT,
    out_path: Path = OUT_PATH,
) -> dict[str, Any]:
    per_run: list[dict] = []

    for mp in sorted(runs_root.glob("*/metrics.json")):
        cfg = json.loads((mp.parent / "config.json").read_text())
        lvl  = int(cfg.get("observability_level", 0))
        aid  = cfg.get("attack_id", "")
        atype = cfg.get("attack_type", "")

        if atype != "indirect":
            continue

        result = _evaluate_run(mp.parent, lvl, aid)
        per_run.append(result)
        status = "✓" if result["ragorigin_attributed"] else ("NO_CHUNK" if not result["poison_retrieved"] else "✗")
        print(f"  {aid} L{lvl}: {status}  rank={result['ragorigin_rank']}")

    # Aggregate per level
    by_level: dict[int, dict] = {}
    for r in per_run:
        lvl = r["level"]
        if lvl not in by_level:
            by_level[lvl] = {
                "level": lvl,
                "n_runs": 0,
                "poison_retrieved": 0,
                "attributed": 0,
                "not_retrieved": 0,
            }
        by_level[lvl]["n_runs"] += 1
        if r["poison_retrieved"]:
            by_level[lvl]["poison_retrieved"] += 1
            if r["ragorigin_attributed"]:
                by_level[lvl]["attributed"] += 1
        else:
            by_level[lvl]["not_retrieved"] += 1

    # Attribution accuracy = attributed / n_runs (same denominator as RAGWitness AA)
    summary: list[dict] = []
    for lvl in sorted(by_level):
        row = by_level[lvl]
        aa = row["attributed"] / row["n_runs"] if row["n_runs"] else 0.0
        row["ragorigin_aa"] = round(aa, 3)
        summary.append(row)

    report = {
        "description": (
            "RAGOrigin-inspired attribution (S1=embed_sim, S2=answer_overlap) "
            "applied to RAGWitness event logs. "
            "S3 (generation influence via live LLM) is omitted — logs only."
        ),
        "levels_with_text": [4, 5],
        "levels_signal_1_only": [1, 2, 3],
        "per_level_summary": summary,
        "per_run": per_run,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    report = run_ragorigin_comparison()
    print("\n=== RAGOrigin Attribution Accuracy (indirect attacks only) ===")
    print(f"{'Level':<8} {'Runs':<6} {'Retrieved':<11} {'Attributed':<12} {'AA':<8} {'Signals'}")
    print("-" * 60)
    for row in report["per_level_summary"]:
        lvl = row["level"]
        sigs = "S1+S2" if lvl >= 4 else "S1 only"
        print(
            f"L{lvl:<7} {row['n_runs']:<6} {row['poison_retrieved']:<11} "
            f"{row['attributed']:<12} {row['ragorigin_aa']:<8.3f} {sigs}"
        )
    print(f"\nSaved → {OUT_PATH}")
