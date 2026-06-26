"""Create compact structured numeric-fact chunks from existing indexed evidence."""

from __future__ import annotations

import hashlib
import re

from pharmalens.ingest.embedder import get_collection

NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:\.\d+)?%?(?:\s*(?:mg|kg|mmol/L|hours?|days?|weeks?|months?|years?|%|m|meters?))?", re.I)
TRIAL_RE = re.compile(r"\b(?:SUSTAIN|FLOW|STRIDE)\s*(?:FORTE|\d+)?\b", re.I)
NUMERIC_KEYWORDS = (
    "sustain", "hba", "1c", "auc", "cmax", "fpg", "body weight", "baseline", "change from baseline",
    "week", "patients", "population", "intent-to-treat", "itt", "confidence", "95% ci", "p<",
    "paracetamol", "atorvastatin", "warfarin", "inr", "salcaprozate", "exposure", "bioequivalence",
    "fasting", "fed", "hours", "days", "72", "table", "figure", "hazard ratio", "mace",
)
ENDPOINT_TERMS = (
    "HbA1c", "HbA", "body weight", "FPG", "AUC", "Cmax", "INR", "MACE", "hazard ratio",
    "patients achieving HbA1c <7%", "Intent-to-Treat Population", "bioequivalence", "pharmacokinetic",
)


def _has_numeric_signal(text: str) -> bool:
    lower = text.lower()
    return bool(NUMBER_RE.search(text)) and any(term in lower for term in NUMERIC_KEYWORDS)


def _values(text: str, limit: int = 80) -> list[str]:
    seen: list[str] = []
    for match in NUMBER_RE.finditer(text):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        if value and value not in seen:
            seen.append(value)
        if len(seen) >= limit:
            break
    return seen


def _trials(text: str) -> list[str]:
    seen: list[str] = []
    for match in TRIAL_RE.finditer(text):
        trial = re.sub(r"\s+", " ", match.group(0)).upper().strip()
        if trial and trial not in seen:
            seen.append(trial)
    return seen


def _endpoints(text: str) -> list[str]:
    lower = text.lower()
    return [term for term in ENDPOINT_TERMS if term.lower() in lower]


def _compact_evidence(text: str, max_chars: int = 1800) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    anchors = [m.start() for m in re.finditer(r"\b(?:SUSTAIN|HbA|AUC|Cmax|Table|week|Change from baseline|Intent-to-Treat|paracetamol|atorvastatin|salcaprozate)\b", text, re.I)]
    if not anchors:
        return text[:max_chars].rstrip()
    start = max(min(anchors) - 300, 0)
    return text[start:start + max_chars].strip()


def _fact_id(source_id: str, evidence: str) -> str:
    digest = hashlib.sha256(f"{source_id}:{evidence}".encode("utf-8")).hexdigest()[:16]
    return f"numeric_fact_{digest}"


def _fact_text(document: str, metadata: dict) -> str:
    evidence = _compact_evidence(document)
    return "\n".join([
        "STRUCTURED_NUMERIC_FACT",
        f"source_file: {metadata.get('filename', '')}",
        f"page: {metadata.get('page_number', '')}",
        f"section: {metadata.get('section_number', '')} {metadata.get('section_title', '')}".strip(),
        f"trial_names: {'; '.join(_trials(evidence)) or 'not explicitly detected'}",
        f"endpoint_terms: {'; '.join(_endpoints(evidence)) or 'not explicitly detected'}",
        f"values: {'; '.join(_values(evidence))}",
        f"exact_evidence: {evidence}",
    ])


def backfill_numeric_facts() -> dict:
    collection = get_collection(with_embedding_function=False)
    try:
        collection.delete(where={"chunk_kind": {"$eq": "numeric_fact"}})
    except Exception:
        pass
    result = collection.get(include=["documents", "metadatas"], limit=5000)
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    for source_id, document, metadata in zip(result.get("ids", []), result.get("documents", []), result.get("metadatas", [])):
        if metadata.get("chunk_kind") == "numeric_fact" or not _has_numeric_signal(document):
            continue
        fact_text = _fact_text(document, metadata)
        ids.append(_fact_id(source_id, fact_text))
        docs.append(fact_text)
        metas.append({
            **metadata,
            "chunk_kind": "numeric_fact",
            "source_chunk_id": source_id,
            "token_count": len(fact_text.split()),
        })
    if ids:
        get_collection(with_embedding_function=True).upsert(ids=ids, documents=docs, metadatas=metas)
    return {"numeric_facts_upserted": len(ids)}


def main() -> None:
    print(backfill_numeric_facts())


if __name__ == "__main__":
    main()
