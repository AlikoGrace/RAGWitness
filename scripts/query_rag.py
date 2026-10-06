import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation import StaticGenerator
from src.metrics import evaluate_run
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one RAG query with forensic observability logging.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default="llama3.1:8b-instruct-q4_K_M")
    ap.add_argument("--chroma_path", default="indexes/chroma_hansard")
    ap.add_argument("--collection", default="hansard_chunks")
    ap.add_argument("--max_chars_per_chunk", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--observability", type=int, default=5, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--attack_id", default="manual")
    ap.add_argument("--attack_type", default="manual")
    ap.add_argument("--mock-generation", action="store_true")
    args = ap.parse_args()

    generator = StaticGenerator(answer="MOCK_MANUAL_ANSWER") if args.mock_generation else None
    config = RAGPipelineConfig(
        query=args.query,
        observability_level=args.observability,
        attack_id=args.attack_id,
        attack_type=args.attack_type,
        chroma_path=args.chroma_path,
        collection=args.collection,
        k=args.k,
        max_chars_per_chunk=args.max_chars_per_chunk,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    run_dir = RAGPipeline(config, generator=generator).run()
    metrics = evaluate_run(run_dir)
    print("run_dir:", run_dir)
    print("metrics:", json.dumps(metrics, indent=2))
    answer = _read_answer(run_dir)
    if answer:
        print("\nANSWER\n")
        print(answer)


def _read_answer(run_dir: Path) -> str:
    events_path = run_dir / "events.jsonl"
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            if event.get("event") == "generation.completed":
                return str((event.get("payload") or {}).get("answer") or "")
    return ""


if __name__ == "__main__":
    main()
