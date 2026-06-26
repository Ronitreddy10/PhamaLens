"""Grounded QA over hybrid-retrieved regulatory excerpts."""

import os
import re

import httpx
import yaml

from pharmalens.models import QueryResponse, RetrievedChunk
from pharmalens.ingest.embedder import get_collection
from pharmalens.paths import CONFIG_DIR
from pharmalens.retrieval.clinical_trial_store import structured_trial_answer
from pharmalens.retrieval.disambiguator import detect_product_intent
from pharmalens.retrieval.hybrid import hybrid_search

with (CONFIG_DIR / "settings.yaml").open() as handle:
    LLM_CONFIG = yaml.safe_load(handle)["llm"]

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
CROSS_REF_RE = re.compile(r"\bsee\s+(?:section\s+)?(\d+(?:\.\d+)+)\b|\[CROSS_REFERENCES:\s*([^\]]+)\]", re.I)
SECTION_REF_RE = re.compile(r"\bsection\s+(\d+(?:\.\d+)+)\b|(\d+(?:\.\d+)+)", re.I)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def detect_query_filters(query: str) -> dict:
    lower = query.lower()
    filters = {}
    mentions_fda = any(re.search(pattern, lower) for pattern in (r"\bfda\b", r"\bamerican\b", r"\bunited states\b"))
    mentions_ema = any(re.search(pattern, lower) for pattern in (r"\bema\b", r"\beuropean\b", r"\beurope\b"))
    if mentions_fda and not mentions_ema:
        filters["regulatory_body"] = "FDA"
    elif mentions_ema and not mentions_fda:
        filters["regulatory_body"] = "EMA"
    mentioned_products = [name for name in ("ozempic", "wegovy", "rybelsus") if name in lower]
    is_comparison = any(term in lower for term in ("compare", "comparison", "difference", "differences", "between", "versus", " vs "))
    if "ozempic" in lower and not (is_comparison and len(mentioned_products) > 1):
        filters["product_name"] = "Ozempic"
        filters.setdefault("doc_type", "EPAR")
    elif "wegovy" in lower and not (is_comparison and len(mentioned_products) > 1):
        filters["product_name"] = "Wegovy"
        filters.setdefault("doc_type", "EPAR")
    elif "rybelsus" in lower and not (is_comparison and len(mentioned_products) > 1):
        filters["product_name"] = "Rybelsus"
    if any(term in lower for term in ("tablet", "oral", "rybelsus")):
        filters["formulation"] = "OralTablet"
    elif any(term in lower for term in ("injection", "subcutaneous", "ozempic", "wegovy")):
        filters["formulation"] = "SubcutaneousSolution"
    if any(term in lower for term in ("bioequivalence", "be study", "psg", "generic")):
        filters["doc_type"] = "PSG"
    elif any(term in lower for term in ("smpc", "epar", "posology")):
        filters["doc_type"] = "EPAR"
    if any(term in lower for term in ("atorvastatin", "paracetamol", "warfarin", "inr", "auc", "cmax", "interaction")):
        filters["doc_type"] = "EPAR"
    if "type 1 diabetes" in lower or "injection site" in lower or "injection sites" in lower:
        filters["doc_type"] = "EPAR"
    return filters


def is_delta_query(query: str) -> bool:
    return any(re.search(pattern, query, re.I) for pattern in (
        r"what.{0,20}change", r"differ.{0,20}between", r"update.{0,20}from", r"revised", r"latest.{0,20}vs"
    ))


def out_of_scope_reason(query: str) -> str | None:
    lower = query.lower()
    if any(term in lower for term in ("price", "cost", "coupon", "insurance", "india", "pharmacy price")):
        return "Pricing, reimbursement, and country-market price information is not contained in the indexed regulatory documents."
    if "tirzepatide" in lower:
        return "The indexed corpus is about semaglutide products; tirzepatide is not covered in these documents."
    if any(term in lower for term in ("planned trial", "future trial", "pipeline", "ongoing trial")):
        return "Future or planned trial information is not contained in the indexed regulatory documents."
    return None


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        citation = f"[{meta.get('filename', 'Unknown')}, Section {meta.get('section_number', '')} {meta.get('section_title', '')}, p.{meta.get('page_number', '?')}]"
        modality = meta.get("dominant_modality") or "NEUTRAL"
        kind = meta.get("chunk_kind") or "text"
        blocks.append(f"--- Excerpt {index} {citation} | kind={kind} | regulatory_modality={modality} ---\n{chunk.text}")
    return "\n\n".join(blocks)


def _cross_references_in_text(text: str) -> list[str]:
    refs: set[str] = set()
    for match in CROSS_REF_RE.finditer(text):
        direct_ref, bracketed_refs = match.groups()
        if direct_ref:
            refs.add(direct_ref.rstrip("."))
        if bracketed_refs:
            for ref_match in SECTION_REF_RE.finditer(bracketed_refs):
                ref = (ref_match.group(1) or ref_match.group(2) or "").rstrip(".")
                if ref:
                    refs.add(ref)
    return sorted(refs)


def _fetch_section_chunks(doc_id: str, section_ref: str, max_chunks: int = 3) -> list[RetrievedChunk]:
    try:
        collection = get_collection(with_embedding_function=False)
        result = collection.get(
            where={"doc_id": {"$eq": doc_id}},
            include=["documents", "metadatas"],
            limit=2000,
        )
    except Exception as exc:
        print(f"[kb_agent] Cross-reference lookup failed for section {section_ref}: {exc}")
        return []
    matches = []
    for chunk_id, document, metadata in zip(result.get("ids", []), result.get("documents", []), result.get("metadatas", [])):
        section_number = str(metadata.get("section_number", "")).rstrip(".")
        if section_number == section_ref or section_number.startswith(f"{section_ref}."):
            matches.append((int(metadata.get("chunk_index", 0)), chunk_id, document, metadata))
    matches.sort(key=lambda item: item[0])
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=document,
            score=0.001,
            metadata={**metadata, "cross_reference_resolved": True},
        )
        for _, chunk_id, document, metadata in matches[:max_chunks]
    ]


def _append_cross_reference_chunks(chunks: list[RetrievedChunk], max_total_extra: int = 10) -> list[RetrievedChunk]:
    enriched = list(chunks)
    seen_ids = {chunk.chunk_id for chunk in enriched}
    added = 0
    for chunk in chunks:
        doc_id = str(chunk.metadata.get("doc_id", ""))
        if not doc_id:
            continue
        for section_ref in _cross_references_in_text(chunk.text):
            if added >= max_total_extra:
                return enriched
            for ref_chunk in _fetch_section_chunks(doc_id, section_ref):
                if added >= max_total_extra:
                    return enriched
                if ref_chunk.chunk_id in seen_ids:
                    continue
                ref_chunk.score = max(chunk.score * 0.75, ref_chunk.score)
                ref_chunk.metadata["cross_reference_from"] = chunk.metadata.get("section_number", "")
                enriched.append(ref_chunk)
                seen_ids.add(ref_chunk.chunk_id)
                added += 1
    return enriched


def _retrieval_query(user_query: str, conversation_history: list[dict] | None = None) -> str:
    """Resolve short follow-ups to the previous substantive user question."""
    compact = user_query.strip().lower()
    follow_up_patterns = (
        "give more", "more", "explain more", "expand", "go deeper", "details",
        "tell me more", "elaborate", "what else", "continue",
    )
    if compact in follow_up_patterns or len(_tokens_for_followup(compact)) <= 2:
        for message in reversed(conversation_history or []):
            if message.get("role") == "user" and len(message.get("content", "").split()) > 3:
                return f"{message['content']} {user_query}"
    return user_query


def _tokens_for_followup(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text)


def _retrieve_with_fallbacks(user_query: str, metadata_filters: dict | None = None) -> list[RetrievedChunk]:
    """Try inferred filters first, then broaden so demo queries don't falsely miss."""
    explicit_filters = metadata_filters or {}
    inferred_filters = detect_query_filters(user_query)
    intent = detect_product_intent(user_query)
    if intent.get("status") == "clear" and intent.get("formulation"):
        inferred_filters.setdefault("formulation", intent["formulation"])
    filter_attempts = [
        {**inferred_filters, **explicit_filters},
        explicit_filters,
        {},
    ]
    seen: set[tuple[tuple[str, str], ...]] = set()
    for filters in filter_attempts:
        key = tuple(sorted((str(k), str(v)) for k, v in filters.items()))
        if key in seen:
            continue
        seen.add(key)
        chunks = hybrid_search(user_query, metadata_filters=filters)
        if chunks:
            return chunks
    return []


def _extractive_answer(chunks: list[RetrievedChunk], reason: str | None = None) -> str:
    """Grounded fallback when the Groq key is not loaded in the API process."""
    intro = reason or "I found relevant source excerpts, but synthesized answering is not available right now."
    lines = [intro, "", "Top excerpts:"]
    for chunk in chunks[:3]:
        meta = chunk.metadata
        citation = f"[{meta.get('filename', 'Unknown')}, Section {meta.get('section_number', '')} {meta.get('section_title', '')}, p.{meta.get('page_number', '?')}]"
        snippet = re.sub(r"\s+", " ", chunk.text).strip()
        lines.append(f"- {snippet[:550]}{'...' if len(snippet) > 550 else ''} {citation}")
    return "\n".join(lines)


def _build_messages(user_query: str, chunks: list[RetrievedChunk], conversation_history: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": LLM_CONFIG["system_prompt"]}]
    messages.extend(conversation_history or [])
    messages.append({"role": "user", "content": (
        f"CONTEXT FROM KNOWLEDGE BASE:\n\n{format_context(chunks)}\n\n"
        f"ENGINEER'S QUESTION: {user_query}\n\n"
        "Produce a polished regulatory intelligence answer. Be specific and useful. "
        "If the user asked for more detail, expand the prior answer rather than restarting. "
        "When regulatory_modality is present, preserve it carefully: may/acceptable is not required, should is recommended, must/required is mandatory. "
        "For numerical, table, figure, endpoint, percentage, confidence interval, p-value, pharmacokinetic, or trial-result questions, answer only when the exact value appears verbatim in the context. Quote or restate only that exact visible value and cite it. If an exact number is not visible, say exactly: 'The precise value is not in the retrieved context.' Never infer, interpolate, calculate, approximate, round, or use prior knowledge to fill numerical values. "
        "Use headings or bullets when they improve clarity. "
        "Answer only from the context and cite every claim as [filename, Section section, p.N]."
    )})
    return messages


def _call_groq(messages: list[dict], model: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""
    from groq import Groq
    response = Groq(api_key=api_key).chat.completions.create(
        model=model, max_completion_tokens=_env_int("PHARMALENS_MAX_COMPLETION_TOKENS", LLM_CONFIG["max_tokens"]),
        messages=messages, temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _call_ollama(messages: list[dict], model: str) -> str:
    response = httpx.post(
        f"{DEFAULT_OLLAMA_URL.rstrip('/')}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
            "messages": messages,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "") or ""


def query(user_query: str, conversation_history: list[dict] | None = None,
          metadata_filters: dict | None = None, stream: bool = False,
          llm_provider: str | None = None, llm_model: str | None = None,
          max_context_chunks: int | None = None) -> QueryResponse:
    del stream
    search_query = _retrieval_query(user_query, conversation_history)
    if reason := out_of_scope_reason(search_query):
        return QueryResponse(answer=f"I could not answer that from this knowledge base. {reason}", sources=[])
    if structured_response := structured_trial_answer(search_query):
        return structured_response
    intent = detect_product_intent(search_query)
    if intent.get("status") == "ambiguous" and not metadata_filters:
        return QueryResponse(answer=intent["message"], sources=[])
    chunks = _retrieve_with_fallbacks(search_query, metadata_filters)
    if not chunks:
        return QueryResponse(answer="I could not find relevant information in the knowledge base for your query.", sources=[])
    chunks = _append_cross_reference_chunks(chunks)
    if max_context_chunks is None:
        max_context_chunks = _env_int("PHARMALENS_MAX_CONTEXT_CHUNKS", 4)
    if max_context_chunks is not None:
        chunks = chunks[:max_context_chunks]
    provider = (llm_provider or LLM_CONFIG.get("provider", "groq")).lower()
    model = llm_model or os.getenv("PHARMALENS_LLM_MODEL") or os.getenv("GROQ_MODEL") or LLM_CONFIG["model"]
    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        return QueryResponse(
            answer=_extractive_answer(
                chunks,
                "I found relevant source excerpts. Add `GROQ_API_KEY` to the backend environment and restart the API for a synthesized Groq answer.",
            ),
            sources=chunks,
        )
    messages = _build_messages(user_query, chunks, conversation_history)
    try:
        if provider == "ollama":
            answer = _call_ollama(messages, model)
        else:
            answer = _call_groq(messages, model)
    except Exception as exc:
        detail = str(exc)
        if "rate_limit" in detail.lower() or "429" in detail:
            answer = _extractive_answer(
                chunks,
                "Groq is temporarily rate-limited for this backend, so I am showing grounded source excerpts instead of a synthesized answer.",
            )
        else:
            raise
    alerts = ["RegDelta: version differences may be relevant; see cited versions."] if is_delta_query(user_query) else []
    return QueryResponse(answer=answer, sources=chunks, delta_alerts=alerts)
