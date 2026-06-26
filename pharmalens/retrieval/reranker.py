"""Optional cross-encoder reranker for final retrieval precision."""

from typing import Any
from pathlib import Path

import yaml

from pharmalens.models import RetrievedChunk
from pharmalens.paths import CONFIG_DIR

with (CONFIG_DIR / "settings.yaml").open() as handle:
    SETTINGS = yaml.safe_load(handle).get("retrieval", {})

RERANKER_MODEL = SETTINGS.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ENABLED = bool(SETTINGS.get("reranker_enabled", True))

_model: Any = None
_load_attempted = False


def _get_model():
    """Load the model lazily. If unavailable/offline, fall back instantly afterward."""
    global _model, _load_attempted
    if _model is not None or _load_attempted:
        return _model
    _load_attempted = True
    cache_name = "models--" + RERANKER_MODEL.replace("/", "--")
    cache_roots = [
        Path.home() / ".cache" / "huggingface" / "hub" / cache_name,
        Path("/private/tmp") / cache_name,
    ]
    if not any(path.exists() for path in cache_roots):
        print(f"[reranker] {RERANKER_MODEL} is not cached locally. Falling back to hybrid scores.")
        return None
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RERANKER_MODEL, model_kwargs={"local_files_only": True}, tokenizer_kwargs={"local_files_only": True})
    except TypeError as exc:
        print(f"[reranker] Local-only CrossEncoder args unsupported: {exc}. Falling back to hybrid scores.")
    except Exception as exc:
        print(f"[reranker] Could not load model: {exc}. Falling back to hybrid scores.")
    return _model


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
    if not RERANKER_ENABLED or not candidates:
        return sorted(candidates, key=lambda chunk: chunk.score, reverse=True)[:top_k]

    model = _get_model()
    if model is None:
        return sorted(candidates, key=lambda chunk: chunk.score, reverse=True)[:top_k]

    pairs = [(query, chunk.text[:512]) for chunk in candidates]
    try:
        scores = model.predict(pairs)
    except Exception as exc:
        print(f"[reranker] Predict failed: {exc}. Falling back to hybrid scores.")
        return sorted(candidates, key=lambda chunk: chunk.score, reverse=True)[:top_k]

    for chunk, score in zip(candidates, scores):
        chunk.score = float(score)
    return sorted(candidates, key=lambda chunk: chunk.score, reverse=True)[:top_k]
