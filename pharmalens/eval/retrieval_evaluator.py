"""Retrieval metrics for the local PharmaEval benchmark."""

from pathlib import Path

import yaml

from pharmalens.agent.kb_agent import detect_query_filters, out_of_scope_reason
from pharmalens.retrieval.disambiguator import detect_product_intent
from pharmalens.retrieval.hybrid import hybrid_search

DEFAULT_BENCHMARK = Path(__file__).with_name("pharmaeval.yaml")


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK) -> list[dict]:
    return yaml.safe_load(Path(path).read_text())["questions"]


def _matches(chunk, gold: dict) -> bool:
    filename = chunk.metadata.get("filename", "")
    if gold.get("file") and gold["file"] != filename:
        return False
    if "section" in gold:
        section = str(gold["section"])
        return section in str(chunk.metadata.get("section_number", "")) or section in str(chunk.metadata.get("section_title", ""))
    if "page" in gold:
        return int(chunk.metadata.get("page_number", -1)) == int(gold["page"])
    return True


def _contains_gold_terms(chunk, terms: list[str]) -> bool:
    if not terms:
        return False
    haystack = (chunk.text + " " + " ".join(str(value) for value in chunk.metadata.values())).lower()
    return any(term.lower() in haystack for term in terms)


def run_retrieval_eval(k: int = 5, question_types: list[str] | None = None) -> dict:
    questions = load_benchmark()
    if question_types:
        questions = [q for q in questions if q["type"] in question_types]

    details = []
    for question in questions:
        gold = question.get("gold_chunks", [])
        gold_terms = question.get("gold_terms", [])
        filters = detect_query_filters(question["question"])
        intent = detect_product_intent(question["question"])
        if intent.get("status") == "clear" and intent.get("formulation"):
            filters.setdefault("formulation", intent["formulation"])
        if out_of_scope_reason(question["question"]):
            retrieved = []
        else:
            retrieved = hybrid_search(question["question"], top_k=k, metadata_filters=filters)
        if not gold:
            if question.get("expected_no_sources", True):
                recall = 1.0 if not retrieved else 0.0
                precision = 1.0 if not retrieved else 0.0
                mrr = 1.0 if not retrieved else 0.0
            else:
                recall = 1.0
                precision = 1.0
                mrr = 1.0
        else:
            matched_gold = sum(1 for item in gold if any(_matches(chunk, item) or _contains_gold_terms(chunk, gold_terms) for chunk in retrieved[:k]))
            recall = matched_gold / len(gold)
            precision = sum(1 for chunk in retrieved[:k] if any(_matches(chunk, item) for item in gold) or _contains_gold_terms(chunk, gold_terms)) / max(k, 1)
            mrr = 0.0
            for rank, chunk in enumerate(retrieved, 1):
                if any(_matches(chunk, item) for item in gold) or _contains_gold_terms(chunk, gold_terms):
                    mrr = 1.0 / rank
                    break
        details.append({"id": question["id"], "type": question["type"], "recall": recall, "precision": precision, "mrr": mrr})

    by_type: dict[str, list[dict]] = {}
    for item in details:
        by_type.setdefault(item["type"], []).append(item)

    def aggregate(items: list[dict]) -> dict:
        return {
            "n": len(items),
            "recall_at_k": sum(item["recall"] for item in items) / len(items),
            "precision_at_k": sum(item["precision"] for item in items) / len(items),
            "mrr": sum(item["mrr"] for item in items) / len(items),
        }

    return {"k": k, "overall": aggregate(details), "by_type": {key: aggregate(value) for key, value in by_type.items()}, "details": details}


def print_report(results: dict) -> None:
    k = results["k"]
    overall = results["overall"]
    print(f"\nPharmaEval Retrieval Report")
    print(f"OVERALL R@{k}: {overall['recall_at_k']:.3f}  P@{k}: {overall['precision_at_k']:.3f}  MRR: {overall['mrr']:.3f}  n={overall['n']}")
    for qtype, metrics in results["by_type"].items():
        print(f"{qtype:<22} R@{k}: {metrics['recall_at_k']:.3f}  P@{k}: {metrics['precision_at_k']:.3f}  MRR: {metrics['mrr']:.3f}  n={metrics['n']}")
