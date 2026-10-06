import argparse

from src.experiments.experiment_baselines import run_baseline_experiments
from src.experiments.experiment_content_hash import run_content_hash_experiment
from src.experiments.experiment_direct import run_direct_experiments
from src.experiments.experiment_indirect import run_indirect_experiments
from src.experiments.experiment_sensitivity import run_sensitivity_experiments
from src.experiments.statistical_analysis import write_summary_tables


def _parse_levels(raw: str) -> list[int]:
    if raw == "all":
        return [1, 2, 3, 4, 5]
    return [int(part) for part in raw.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run forensic RAG experiments")
    parser.add_argument(
        "--phase",
        choices=["direct", "indirect", "baselines", "all", "summaries",
                 "sensitivity", "content-hash"],
        required=True,
    )
    parser.add_argument(
        "--levels",
        default="all",
        help="Comma-separated observability levels, for example 1,3,5. Default: all",
    )
    parser.add_argument(
        "--mock-generation",
        action="store_true",
        help="Use deterministic mock responses instead of calling local Ollama.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit attacks/queries per phase for smoke tests.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Directory where experiment run folders are written. Default: runs",
    )
    parser.add_argument(
        "--corpus-version",
        default="v1.0_200docs_15482chunks",
        help="Corpus version string stamped into every run config.",
    )
    parser.add_argument(
        "--max-clean-chunks",
        type=int,
        default=0,
        help="Indirect experiments only: limit clean chunks before poisoning. 0 means all.",
    )
    parser.add_argument(
        "--skip-index-rebuild",
        action="store_true",
        help="Indirect experiments only: reuse existing poisoned Chroma indexes.",
    )
    parser.add_argument(
        "--baseline-start-index",
        type=int,
        default=1,
        help="Baseline experiments only: start from this 1-based baseline query index. Use 3 to run the remaining B3-B8 comparisons.",
    )
    parser.add_argument(
        "--baseline-end-index",
        type=int,
        default=0,
        help="Baseline experiments only: stop at this 1-based baseline query index. 0 means run through the end.",
    )

    args = parser.parse_args()
    levels = _parse_levels(args.levels)

    run_dirs = []
    if args.phase in {"direct", "all"}:
        run_dirs.extend(
            run_direct_experiments(
                levels=levels,
                mock_generation=args.mock_generation,
                limit=args.limit,
                runs_root=args.runs_root,
                corpus_version=args.corpus_version,
            )
        )
    if args.phase in {"indirect", "all"}:
        run_dirs.extend(
            run_indirect_experiments(
                levels=levels,
                mock_generation=args.mock_generation,
                limit=args.limit,
                max_clean_chunks=args.max_clean_chunks,
                rebuild_indexes=not args.skip_index_rebuild,
                runs_root=args.runs_root,
                corpus_version=args.corpus_version,
            )
        )
    if args.phase in {"baselines", "all"}:
        run_dirs.extend(
            run_baseline_experiments(
                levels=levels,
                mock_generation=args.mock_generation,
                limit=args.limit,
                start_index=args.baseline_start_index,
                end_index=args.baseline_end_index,
                runs_root=args.runs_root,
                corpus_version=args.corpus_version,
            )
        )
    if args.phase == "sensitivity":
        run_sensitivity_experiments(
            mock_generation=args.mock_generation,
            skip_index_rebuild=args.skip_index_rebuild,
            corpus_version=args.corpus_version,
        )
    if args.phase == "content-hash":
        run_content_hash_experiment(
            mock_generation=args.mock_generation,
            corpus_version=args.corpus_version,
        )
    if args.phase in {"summaries", "all"}:
        outputs = write_summary_tables(runs_root=args.runs_root)
        print("Wrote summaries:")
        for name, path in outputs.items():
            print(f"  {name}: {path}")

    if run_dirs:
        print("Completed runs:")
        for run_dir in run_dirs:
            print(f"  {run_dir}")


if __name__ == "__main__":
    main()
