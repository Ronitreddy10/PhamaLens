"""Chroma persistence and configurable embedding provider."""

import os
from typing import Any

import yaml

from pharmalens.models import ChunkMetadata
from pharmalens.paths import CONFIG_DIR, resolve_data_path

with (CONFIG_DIR / "settings.yaml").open() as handle:
    SETTINGS = yaml.safe_load(handle)
VS = SETTINGS["vector_store"]
_client: Any = None
_collection_with_embedding: Any = None
_collection_without_embedding: Any = None


def _build_embedding_function():
    provider, model = VS["embedding_provider"], VS["embedding_model"]
    if provider == "openai":
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the configured embedding provider")
        return OpenAIEmbeddingFunction(api_key=api_key, model_name=model)
    if provider == "sentence_transformers":
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        return SentenceTransformerEmbeddingFunction(model_name=model)
    raise ValueError(f"Unknown embedding provider: {provider}")


def get_collection(with_embedding_function: bool = True):
    global _client, _collection_with_embedding, _collection_without_embedding
    if _client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        persist_dir = resolve_data_path(VS["persist_dir"])
        persist_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False))
    if with_embedding_function and _collection_with_embedding is None:
        _collection_with_embedding = _client.get_or_create_collection(
            name=VS["collection_name"], embedding_function=_build_embedding_function(), metadata={"hnsw:space": "cosine"}
        )
    if not with_embedding_function and _collection_without_embedding is None:
        _collection_without_embedding = _client.get_collection(name=VS["collection_name"])
    return _collection_with_embedding if with_embedding_function else _collection_without_embedding


def is_already_indexed(doc_id: str) -> bool:
    try:
        return bool(get_collection(with_embedding_function=False).get(where={"doc_id": {"$eq": doc_id}}, limit=1)["ids"])
    except Exception:
        return False


def upsert_chunks(chunks: list[ChunkMetadata]) -> int:
    if not chunks:
        return 0
    collection = get_collection()
    for start in range(0, len(chunks), 500):
        batch = chunks[start:start+500]
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch], documents=[chunk.text for chunk in batch],
            metadatas=[{
                "doc_id": c.doc_id, "filename": c.filename, "doc_type": c.doc_type,
                "regulatory_body": c.regulatory_body, "product_name": c.product_name, "api": c.api,
                "formulation": c.formulation, "route": c.route, "version_date": c.version_date or "",
                "section_title": c.section_title, "section_number": c.section_number,
                "page_number": c.page_number, "chunk_index": c.chunk_index, "token_count": c.token_count,
                "chunk_kind": c.chunk_kind, "dominant_modality": c.dominant_modality,
            } for c in batch],
        )
    return len(chunks)


def delete_document(doc_id: str) -> None:
    get_collection().delete(where={"doc_id": {"$eq": doc_id}})
