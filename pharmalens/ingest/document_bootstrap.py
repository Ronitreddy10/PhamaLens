"""Download bundled demo PDFs when a deploy target cannot store binaries in git."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path


DOCUMENT_FILENAMES = [
    "FDA-2025-P-3239-0001_attachment_1.pdf",
    "PSG_209637.pdf",
    "PSG_213051.pdf",
    "PSG_215256.pdf",
    "ozempic-epar-product-information_en.pdf",
    "wegovy-epar-product-information_en.pdf",
]

DEFAULT_DOCUMENT_BASE_URL = (
    "https://raw.githubusercontent.com/Ronitreddy10/PhamaLens/main/"
    "pharmalens/data/documents"
)


def ensure_demo_documents(documents_dir: Path) -> list[Path]:
    """Ensure the bundled demo PDFs exist, downloading missing ones if configured.

    Hugging Face Spaces rejects ordinary git pushes containing binary PDFs unless
    Xet/LFS is used. For the demo Space, we keep the Space repo lightweight and
    download the public PDFs from the GitHub repo on first startup.
    """
    if os.getenv("PHARMALENS_DOWNLOAD_DEMO_DOCS", "1").lower() in {"0", "false", "no"}:
        return []

    base_url = os.getenv("PHARMALENS_DOCUMENT_BASE_URL", DEFAULT_DOCUMENT_BASE_URL).rstrip("/")
    documents_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename in DOCUMENT_FILENAMES:
        target = documents_dir / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"{base_url}/{filename}"
        print(f"[documents] Downloading {url}", flush=True)
        try:
            urllib.request.urlretrieve(url, target)
            downloaded.append(target)
        except Exception as exc:
            if target.exists():
                target.unlink()
            print(f"[documents] Failed to download {filename}: {exc}", flush=True)
    return downloaded
