"""Section-level change detection across regulatory document versions."""

import numpy as np
import yaml

from pharmalens.ingest.embedder import get_collection
from pharmalens.models import DocMetadata
from pharmalens.paths import CONFIG_DIR

with (CONFIG_DIR / "settings.yaml").open() as handle:
    THRESHOLD = yaml.safe_load(handle)["delta"]["similarity_threshold"]


def find_previous_version(new_doc: DocMetadata) -> dict | None:
    if not new_doc.version_date:
        return None
    where = {"$and": [{"doc_type": {"$eq": new_doc.doc_type}}, {"api": {"$eq": new_doc.api}},
                      {"formulation": {"$eq": new_doc.formulation}}]}
    results = get_collection().get(where=where, include=["metadatas"], limit=1000)
    candidates = {meta["doc_id"]: meta for meta in results["metadatas"]
                  if meta.get("doc_id") != new_doc.doc_id and meta.get("version_date")
                  and meta["version_date"] < new_doc.version_date}
    return max(candidates.values(), key=lambda item: item["version_date"], default=None)


def detect_changes(new_doc: DocMetadata) -> list[str]:
    prior = find_previous_version(new_doc)
    if not prior:
        return []
    collection = get_collection()
    new = collection.get(where={"doc_id": {"$eq": new_doc.doc_id}}, include=["metadatas", "embeddings"])
    old = collection.get(where={"doc_id": {"$eq": prior["doc_id"]}}, include=["metadatas", "embeddings"])
    if new.get("embeddings") is None or old.get("embeddings") is None:
        return [f"[RegDelta] New version {new_doc.version_date}; prior version {prior['version_date']}. Comparison unavailable."]

    def sections(data: dict) -> dict:
        result = {}
        for meta, embedding in zip(data["metadatas"], data["embeddings"]):
            key = meta.get("section_number") or f"chunk-{meta['chunk_index']}"
            result.setdefault(key, (meta.get("section_title", ""), np.asarray(embedding)))
        return result

    current, previous = sections(new), sections(old)
    alerts = [f"[RegDelta] NEW §{section} '{current[section][0]}'" for section in current.keys() - previous.keys()]
    for section in current.keys() & previous.keys():
        a, b = current[section][1], previous[section][1]
        similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
        if similarity < THRESHOLD:
            alerts.append(f"[RegDelta] CHANGED §{section} '{current[section][0]}' ({(1-similarity)*100:.1f}% divergence)")
    alerts.extend(f"[RegDelta] REMOVED §{section} '{previous[section][0]}'" for section in previous.keys() - current.keys())
    return alerts or [f"[RegDelta] Minor or no changes between {prior['version_date']} and {new_doc.version_date}."]
