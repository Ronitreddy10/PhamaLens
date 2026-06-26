"""Ablation study runner for PharmaLens retrieval."""

from pharmalens.eval.retrieval_evaluator import run_retrieval_eval

CONFIGURATIONS = [
    {
        "name": "Baseline: Vector-only (no BM25)",
        "settings": {"bm25_weight": 0.0, "vector_weight": 1.0, "reranker_enabled": False, "synonym_expansion": False},
    },
    {
        "name": "Baseline: BM25-only (no vector)",
        "settings": {"bm25_weight": 1.0, "vector_weight": 0.0, "reranker_enabled": False, "synonym_expansion": False},
    },
    {
        "name": "Hybrid (BM25 + Vector, no reranker)",
        "settings": {"bm25_weight": 0.4, "vector_weight": 0.6, "reranker_enabled": False, "synonym_expansion": False},
    },
    {
        "name": "Hybrid + Synonym Expansion",
        "settings": {"bm25_weight": 0.4, "vector_weight": 0.6, "reranker_enabled": False, "synonym_expansion": True},
    },
    {
        "name": "Full PharmaLens (Hybrid + Synonym + Reranker)",
        "settings": {"bm25_weight": 0.4, "vector_weight": 0.6, "reranker_enabled": True, "synonym_expansion": True},
    },
]


def patch_settings(config: dict) -> None:
    import pharmalens.ingest.synonym_expander as syn_mod
    import pharmalens.retrieval.hybrid as hybrid_mod
    import pharmalens.retrieval.reranker as reranker_mod

    hybrid_mod.BM25_WEIGHT = config["bm25_weight"]
    hybrid_mod.VECTOR_WEIGHT = config["vector_weight"]
    hybrid_mod.SETTINGS["bm25_weight"] = config["bm25_weight"]
    hybrid_mod.SETTINGS["vector_weight"] = config["vector_weight"]
    reranker_mod.RERANKER_ENABLED = config["reranker_enabled"]
    syn_mod._ENABLED = config["synonym_expansion"]


def run_ablation(k: int = 5) -> list[dict]:
    print(f"\n{'=' * 75}")
    print(f"PharmaLens Ablation Study — Recall@{k}, Precision@{k}, MRR")
    print(f"{'=' * 75}")
    print(f"{'Configuration':<48} {'R@5':>6} {'P@5':>6} {'MRR':>6} {'n':>4}")
    print("-" * 75)

    results = []
    for cfg in CONFIGURATIONS:
        patch_settings(cfg["settings"])
        metrics = run_retrieval_eval(k=k)
        ov = metrics["overall"]
        row = {
            "name": cfg["name"],
            "recall": ov["recall_at_k"],
            "precision": ov["precision_at_k"],
            "mrr": ov["mrr"],
            "n": ov["n"],
        }
        results.append(row)
        print(f"{row['name']:<48} {row['recall']:>6.3f} {row['precision']:>6.3f} {row['mrr']:>6.3f} {row['n']:>4}")

    print("=" * 75)
    return results


if __name__ == "__main__":
    run_ablation(k=5)
