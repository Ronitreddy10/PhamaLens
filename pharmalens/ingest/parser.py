"""PDF text extraction kept separate for reuse and testing."""

import pdfplumber


def extract_pages(filepath: str) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(filepath) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            pages.append((number, page.extract_text() or ""))
    return pages
