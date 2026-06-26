"""Structured table extraction for PDF tables."""

import json

import pdfplumber

from pharmalens.ingest.synonym_expander import expand_for_indexing
from pharmalens.models import ChunkMetadata, DocMetadata
from pharmalens.paths import DATA_DIR

TABLE_STORE_DIR = DATA_DIR / "tables"
TABLE_STORE_DIR.mkdir(parents=True, exist_ok=True)


def _clean_cell(value) -> str:
    return str(value or "").strip()


def extract_tables_from_pdf(filepath: str, doc_meta: DocMetadata) -> list[ChunkMetadata]:
    chunks: list[ChunkMetadata] = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                chunks.extend(extract_tables_from_page(page, page_num, doc_meta))
    except Exception as exc:
        print(f"[table_extractor] Error extracting tables from {doc_meta.filename}: {exc}")
    return chunks


def extract_tables_from_page(page, page_num: int, doc_meta: DocMetadata) -> list[ChunkMetadata]:
    chunks: list[ChunkMetadata] = []
    for table_idx, table in enumerate(page.extract_tables() or []):
        if not table or len(table) < 2:
            continue

        header = [_clean_cell(cell) for cell in table[0]]
        if not any(header):
            continue
        rows = table[1:]
        table_id = f"{doc_meta.doc_id[:8]}_p{page_num}_t{table_idx}"
        structured_rows = [
            {header[i]: _clean_cell(cell) for i, cell in enumerate(row) if i < len(header) and header[i]}
            for row in rows
        ]
        (TABLE_STORE_DIR / f"{table_id}.json").write_text(json.dumps({
            "table_id": table_id,
            "doc_id": doc_meta.doc_id,
            "filename": doc_meta.filename,
            "page_number": page_num,
            "headers": header,
            "rows": structured_rows,
        }, indent=2))

        nl_lines = [f"Table {table_id} on page {page_num} of {doc_meta.filename}.", f"Columns: {', '.join(h for h in header if h)}."]
        for row in structured_rows[:20]:
            row_text = " | ".join(f"{key}: {value}" for key, value in row.items() if value)
            if row_text:
                nl_lines.append(row_text)
        nl_text = expand_for_indexing("\n".join(nl_lines))
        chunks.append(ChunkMetadata(
            doc_id=doc_meta.doc_id, filename=doc_meta.filename, doc_type=doc_meta.doc_type,
            regulatory_body=doc_meta.regulatory_body, product_name=doc_meta.product_name, api=doc_meta.api,
            formulation=doc_meta.formulation, route=doc_meta.route, version_date=doc_meta.version_date,
            section_title=f"Table {table_idx}", section_number=f"table_{table_idx}", page_number=page_num,
            chunk_index=len(chunks), token_count=len(nl_text.split()), text=nl_text, chunk_kind="table",
        ))

        for row_idx, row in enumerate(structured_rows):
            row_label = next((value for value in row.values() if value), f"row_{row_idx}")
            row_text = expand_for_indexing(
                f"[{doc_meta.filename}, p.{page_num}, table {table_idx}, row {row_label}]\n"
                + " | ".join(f"{key}: {value}" for key, value in row.items() if value)
            )
            chunks.append(ChunkMetadata(
                doc_id=doc_meta.doc_id, filename=doc_meta.filename, doc_type=doc_meta.doc_type,
                regulatory_body=doc_meta.regulatory_body, product_name=doc_meta.product_name, api=doc_meta.api,
                formulation=doc_meta.formulation, route=doc_meta.route, version_date=doc_meta.version_date,
                section_title=f"Table row: {row_label}", section_number=f"table_{table_idx}_row_{row_idx}",
                page_number=page_num, chunk_index=len(chunks), token_count=len(row_text.split()),
                text=row_text, chunk_kind="table_row",
            ))
    return chunks


def lookup_table(table_id: str) -> dict | None:
    path = TABLE_STORE_DIR / f"{table_id}.json"
    return json.loads(path.read_text()) if path.exists() else None
