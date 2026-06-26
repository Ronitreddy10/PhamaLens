"""PharmChunk: structure-aware pharmaceutical document chunking."""

import re

import tiktoken
import yaml

from pharmalens.ingest.figure_extractor import extract_figure_chunks
from pharmalens.ingest.modality_tagger import tag_chunk
from pharmalens.ingest.parser import extract_pages
from pharmalens.ingest.synonym_expander import expand_for_indexing
from pharmalens.ingest.table_extractor import extract_tables_from_pdf
from pharmalens.models import ChunkMetadata, DocMetadata
from pharmalens.paths import CONFIG_DIR

with (CONFIG_DIR / "settings.yaml").open() as handle:
    SETTINGS = yaml.safe_load(handle)
with (CONFIG_DIR / "doc_types.yaml").open() as handle:
    DOC_TYPES = yaml.safe_load(handle)["doc_types"]

TOKENIZER = tiktoken.get_encoding("cl100k_base")
CHUNK_SIZE = SETTINGS["ingest"]["chunk_size_tokens"]
CHUNK_OVERLAP = SETTINGS["ingest"]["chunk_overlap_tokens"]
EPAR_SECTION_RE = re.compile(r"^([1-9]\d?(?:\.\d+)+\.?)\s+([A-Z][A-Za-z].+)$", re.M)
LEGAL_SECTION_RE = re.compile(r"^([A-Z])\.\s+(.+)$", re.M)
CROSS_REF_RE = re.compile(r"\bsee\s+(?:section\s+)?(\d+(?:\.\d+)+)\b", re.I)


def token_count(text: str) -> int:
    return len(TOKENIZER.encode(text))


def _chunk(text: str, title: str, number: str, page: int, meta: DocMetadata, offset: int) -> list[ChunkMetadata]:
    text = _prepare_chunk_text(text)
    tokens = TOKENIZER.encode(text)
    result: list[ChunkMetadata] = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        body = TOKENIZER.decode(tokens[start:end]).strip()
        if body:
            modality = tag_chunk(body)["dominant_modality"]
            result.append(ChunkMetadata(
                doc_id=meta.doc_id, filename=meta.filename, doc_type=meta.doc_type,
                regulatory_body=meta.regulatory_body, product_name=meta.product_name, api=meta.api,
                formulation=meta.formulation, route=meta.route, version_date=meta.version_date,
                section_title=title, section_number=number, page_number=page,
                chunk_index=offset + len(result), token_count=end-start, text=body, dominant_modality=modality,
            ))
        if end == len(tokens):
            break
        start = end - CHUNK_OVERLAP
    return result


def _prepare_chunk_text(text: str) -> str:
    prepared = expand_for_indexing(text)
    refs = sorted(set(CROSS_REF_RE.findall(prepared)))
    if refs:
        prepared += "\n[CROSS_REFERENCES: " + "; ".join(f"section {ref}" for ref in refs) + "]"
    return prepared


def _sections_on_pages(pages: list[tuple[int, str]], pattern: re.Pattern[str]) -> list[tuple[int, str, str, str]]:
    sections: list[tuple[int, str, str, str]] = []
    current: tuple[int, str, str, list[str]] | None = None
    for page, text in pages:
        matches = list(pattern.finditer(text))
        cursor = 0
        for match in matches:
            if current and match.start() > cursor:
                current[3].append(text[cursor:match.start()])
            if current:
                sections.append((current[0], current[1], current[2], "\n".join(current[3]).strip()))
            number, title = match.group(1).rstrip("."), match.group(2).strip()
            current = (page, number, title, [match.group(0)])
            cursor = match.end()
        if current:
            current[3].append(text[cursor:])
    if current:
        sections.append((current[0], current[1], current[2], "\n".join(current[3]).strip()))
    return [section for section in sections if section[3]]


def parse_epar_smpc(pages: list[tuple[int, str]], meta: DocMetadata) -> list[ChunkMetadata]:
    sections = _sections_on_pages(pages, EPAR_SECTION_RE)
    return _materialize(sections, meta) if sections else parse_generic(pages, meta)


def parse_legal_sections(pages: list[tuple[int, str]], meta: DocMetadata) -> list[ChunkMetadata]:
    sections = _sections_on_pages(pages, LEGAL_SECTION_RE)
    return _materialize(sections, meta) if sections else parse_generic(pages, meta)


def _materialize(sections: list[tuple[int, str, str, str]], meta: DocMetadata) -> list[ChunkMetadata]:
    chunks: list[ChunkMetadata] = []
    for page, number, title, text in sections:
        chunks.extend(_chunk(text, title, number, page, meta, len(chunks)))
    return chunks


def parse_psg(pages: list[tuple[int, str]], meta: DocMetadata) -> list[ChunkMetadata]:
    blocks: list[tuple[int, str, str, str]] = []
    split_re = re.compile(r"^(Option\s+[IVX]+:?|Recommended Studies:?|Document History:?)", re.I | re.M)
    for page, text in pages:
        parts = [part.strip() for part in split_re.split(text) if part.strip()]
        for index in range(0, len(parts), 2):
            heading = parts[index] if split_re.match(parts[index]) else f"Page {page}"
            body = parts[index + 1] if split_re.match(parts[index]) and index + 1 < len(parts) else parts[index]
            blocks.append((page, str(len(blocks)+1), heading[:120], f"{heading}\n{body}"))
    return _materialize(blocks, meta)


def parse_generic(pages: list[tuple[int, str]], meta: DocMetadata) -> list[ChunkMetadata]:
    chunks: list[ChunkMetadata] = []
    for page, text in pages:
        if text.strip():
            chunks.extend(_chunk(text, f"Page {page}", "", page, meta, len(chunks)))
    return chunks


PARSERS = {"epar_smpc": parse_epar_smpc, "psg": parse_psg, "legal_sections": parse_legal_sections,
           "fda_label": parse_generic, "csr": parse_generic, "generic": parse_generic}


def chunk_document(filepath: str, doc_meta: DocMetadata) -> list[ChunkMetadata]:
    try:
        pages = extract_pages(filepath)
    except Exception as exc:
        print(f"[chunker] Error reading {filepath}: {exc}")
        return []
    parser_name = DOC_TYPES.get(doc_meta.doc_type, {}).get("section_parser", "generic")
    chunks = PARSERS.get(parser_name, parse_generic)(pages, doc_meta)
    chunks.extend(extract_tables_from_pdf(filepath, doc_meta))
    chunks.extend(extract_figure_chunks(filepath, doc_meta))
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
        if chunk.dominant_modality == "NEUTRAL":
            chunk.dominant_modality = tag_chunk(chunk.text)["dominant_modality"]
    print(f"[chunker] {doc_meta.filename}: {len(chunks)} chunks using '{parser_name}'")
    return chunks
