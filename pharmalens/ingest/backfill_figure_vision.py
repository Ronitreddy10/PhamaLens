"""Backfill Groq vision descriptions into existing figure chunks."""

import re
from pathlib import Path

from pharmalens.ingest.embedder import get_collection
from pharmalens.ingest.figure_extractor import describe_figure_page
from pharmalens.paths import DATA_DIR


def _document_path(filename: str) -> Path:
    return DATA_DIR / "documents" / filename


def backfill_figure_vision(limit_pages: int | None = None, force: bool = False) -> dict:
    collection = get_collection(with_embedding_function=False)
    result = collection.get(where={"chunk_kind": {"$eq": "figure"}}, include=["documents", "metadatas"], limit=2000)
    ids = result.get("ids", [])
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    descriptions: dict[tuple[str, int], str | None] = {}
    updated_ids: list[str] = []
    updated_docs: list[str] = []
    updated_metas: list[dict] = []

    for chunk_id, document, metadata in zip(ids, docs, metas):
        if not force and "Visual description:" in document and "To inspect the figure" not in document:
            continue
        filename = str(metadata.get("filename", ""))
        page = int(metadata.get("page_number", 0) or 0)
        key = (filename, page)
        if key not in descriptions:
            if limit_pages is not None and len(descriptions) >= limit_pages:
                break
            pdf_path = _document_path(filename)
            descriptions[key] = describe_figure_page(str(pdf_path), page) if pdf_path.exists() else None
            print(f"[figure_backfill] {filename} p.{page}: {'ok' if descriptions[key] else 'no description'}", flush=True)
        description = descriptions[key]
        if not description:
            continue
        updated = document.replace(
            f"To inspect the figure, open page {page} of {filename}.",
            f"Visual description: {description}",
        )
        updated = re.sub(
            r"Visual description: .*?(?=\n\[SYNONYMS:|$)",
            lambda _: f"Visual description: {description}",
            updated,
            flags=re.S,
        )
        if updated == document:
            updated = f"{document}\nVisual description: {description}"
        updated_ids.append(chunk_id)
        updated_docs.append(updated)
        updated_metas.append(metadata)

    if updated_ids:
        get_collection(with_embedding_function=True).upsert(ids=updated_ids, documents=updated_docs, metadatas=updated_metas)
    return {"pages_described": sum(1 for value in descriptions.values() if value), "chunks_updated": len(updated_ids)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Refresh existing visual descriptions.")
    parser.add_argument("--limit-pages", type=int, default=None)
    args = parser.parse_args()
    print(backfill_figure_vision(limit_pages=args.limit_pages, force=args.force))


if __name__ == "__main__":
    main()
