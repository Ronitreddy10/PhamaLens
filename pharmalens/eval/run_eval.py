"""Run PharmaEval retrieval and faithfulness evaluation."""

import argparse
import json
import re
import time
from pathlib import Path

from groq import APIStatusError, RateLimitError

from pharmalens.agent import kb_agent
from pharmalens.agent.kb_agent import format_context, query
from pharmalens.eval.faithfulness_scorer import DEFAULT_JUDGE_MODEL, DEFAULT_JUDGE_PROVIDER, judge_answer, heuristic_score
from pharmalens.eval.retrieval_evaluator import print_report, run_retrieval_eval
from pharmalens.eval.retrieval_evaluator import load_benchmark


RESULTS_DIR = Path(__file__).with_name("results")
MAX_GROQ_RETRIES = 3


def _score_value(score: dict, key: str) -> float:
    try:
        value = score.get(key, 0.0)
        if isinstance(value, dict):
            value = value.get("score", 0.0)
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _aggregate(items: list[dict]) -> dict:
    if not items:
        return {"n": 0, "faithfulness": 0.0, "citation_accuracy": 0.0, "modality_preservation": 0.0}
    return {
        "n": len(items),
        "faithfulness": sum(item["faithfulness"] for item in items) / len(items),
        "citation_accuracy": sum(item["citation_accuracy"] for item in items) / len(items),
        "modality_preservation": sum(item["modality_preservation"] for item in items) / len(items),
    }


def _retry_seconds(exc: Exception) -> float:
    message = str(exc)
    hour_match = re.search(r"try again in (\d+)h(\d+)m([\d.]+)s", message, re.I)
    if hour_match:
        return int(hour_match.group(1)) * 3600 + int(hour_match.group(2)) * 60 + float(hour_match.group(3)) + 3
    minute_match = re.search(r"try again in (\d+)m([\d.]+)s", message, re.I)
    if minute_match:
        return int(minute_match.group(1)) * 60 + float(minute_match.group(2)) + 3
    second_match = re.search(r"try again in ([\d.]+)s", message, re.I)
    if second_match:
        return float(second_match.group(1)) + 3
    return 20.0


def _with_groq_retries(label: str, func):
    for attempt in range(MAX_GROQ_RETRIES + 1):
        try:
            return func()
        except (RateLimitError, APIStatusError) as exc:
            if attempt >= MAX_GROQ_RETRIES or getattr(exc, "status_code", None) not in (413, 429):
                raise
            delay = _retry_seconds(exc)
            print(f"[eval] {label} rate-limited; retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)


def run_faithfulness_eval(
    answer_provider: str | None = None,
    answer_model: str | None = None,
    answer_max_context: int | None = None,
    judge_provider: str = DEFAULT_JUDGE_PROVIDER,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    question_type: str | None = None,
) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    progress_path = RESULTS_DIR / "faithfulness_progress.json"
    details = []
    effective_max_context = answer_max_context
    if effective_max_context is None and answer_provider == "ollama":
        effective_max_context = 8
    benchmark = [
        item for item in load_benchmark()
        if question_type is None or item.get("type") == question_type
    ]
    for item in benchmark:
        response = _with_groq_retries(
            item["id"],
            lambda: query(
                item["question"],
                llm_provider=answer_provider,
                llm_model=answer_model,
                max_context_chunks=effective_max_context,
            ),
        )
        context = format_context(response.sources)
        try:
            score = _with_groq_retries(
                f"{item['id']} judge",
                lambda: judge_answer(
                    item["question"],
                    response.answer,
                    context,
                    provider=judge_provider,
                    model=judge_model,
                ),
            )
            scorer = f"{judge_provider}_judge"
        except Exception as exc:
            score = heuristic_score(response.answer, context)
            scorer = f"heuristic_fallback: {exc}"
        details.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "gold_answer": item.get("gold_answer"),
            "answer": response.answer,
            "source_count": len(response.sources),
            "scorer": scorer,
            "raw_score": score,
            "faithfulness": _score_value(score, "faithfulness"),
            "citation_accuracy": _score_value(score, "citation_accuracy"),
            "modality_preservation": _score_value(score, "modality_preservation"),
        })
        progress_path.write_text(json.dumps({"details": details}, indent=2))
        print(f"[faithfulness] {item['id']} done", flush=True)

    by_type: dict[str, list[dict]] = {}
    for detail in details:
        by_type.setdefault(detail["type"], []).append(detail)

    return {
        "overall": _aggregate(details),
        "by_type": {key: _aggregate(value) for key, value in by_type.items()},
        "details": details,
    }


def print_faithfulness_report(results: dict) -> None:
    overall = results["overall"]
    print("\nPharmaEval Faithfulness Report")
    print(
        "OVERALL "
        f"Faithfulness: {overall['faithfulness']:.3f}  "
        f"Citations: {overall['citation_accuracy']:.3f}  "
        f"Modality: {overall['modality_preservation']:.3f}  "
        f"n={overall['n']}"
    )
    for qtype, metrics in results["by_type"].items():
        print(
            f"{qtype:<22} "
            f"Faithfulness: {metrics['faithfulness']:.3f}  "
            f"Citations: {metrics['citation_accuracy']:.3f}  "
            f"Modality: {metrics['modality_preservation']:.3f}  "
            f"n={metrics['n']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="Run retrieval plus faithfulness evaluation.")
    parser.add_argument("--answer-provider", default=None, choices=["groq", "ollama"], help="Optional answer-generation provider for full eval.")
    parser.add_argument("--answer-model", default=None, help="Optional model to use for answer generation during full eval.")
    parser.add_argument("--answer-max-context", type=int, default=None, help="Optional max context chunks passed to answer generation.")
    parser.add_argument("--judge-provider", default=DEFAULT_JUDGE_PROVIDER, choices=["ollama", "groq", "heuristic"], help="Faithfulness judge provider.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Model to use as the faithfulness judge.")
    parser.add_argument("--type", default=None, help="Optional question type filter, e.g. numerical_table.")
    parser.add_argument("--eval-model", default=None, help="Deprecated alias for --answer-model.")
    args = parser.parse_args()
    retrieval_results = run_retrieval_eval(k=args.k, question_types=[args.type] if args.type else None)
    print_report(retrieval_results)
    if args.full:
        faithfulness_results = run_faithfulness_eval(
            answer_provider=args.answer_provider,
            answer_model=args.answer_model or args.eval_model,
            answer_max_context=args.answer_max_context,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            question_type=args.type,
        )
        print_faithfulness_report(faithfulness_results)
        RESULTS_DIR.mkdir(exist_ok=True)
        output_path = RESULTS_DIR / "faithfulness_last.json"
        output_path.write_text(json.dumps(faithfulness_results, indent=2))
        print(f"\nSaved faithfulness details to {output_path}")


if __name__ == "__main__":
    main()
