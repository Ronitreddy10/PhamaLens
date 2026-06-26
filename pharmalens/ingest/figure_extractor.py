"""Figure-caption anchoring with optional Groq vision descriptions."""

import base64
import io
import os
import re

import pdfplumber
import yaml

from pharmalens.ingest.synonym_expander import expand_for_indexing
from pharmalens.models import ChunkMetadata, DocMetadata
from pharmalens.paths import CONFIG_DIR

FIGURE_CAPTION_RE = re.compile(r"(Figure\s+\d+[A-Za-z]?\s*[:\-–]?\s*.{10,260})", re.IGNORECASE)

with (CONFIG_DIR / "settings.yaml").open() as handle:
    SETTINGS = yaml.safe_load(handle)

VISION_ENABLED = bool(SETTINGS.get("retrieval", {}).get("vision_enabled", False))
VISION_MODEL = SETTINGS.get("retrieval", {}).get("vision_model", "meta-llama/llama-4-scout-17b-16e-instruct")
VISION_FALLBACK_MODEL = SETTINGS.get("retrieval", {}).get("vision_fallback_model", "qwen/qwen3.6-27b")
VISION_DPI = int(SETTINGS.get("retrieval", {}).get("vision_dpi", 300))
VISION_MAX_IMAGE_SIZE = tuple(SETTINGS.get("retrieval", {}).get("vision_max_image_size", [2400, 3400]))

VISION_PROMPT = """This is a page from a pharmaceutical regulatory document for semaglutide.

If this page contains a figure, graph, chart, plot, table, or diagram, describe:
1. Figure type
2. Axis labels and units
3. Treatment arms or groups
4. Key finding or visible trend
5. Any readable values, timepoints, or statistical markers

If the page contains no figure, respond with: NO_FIGURE
"""


def _vision_dependencies():
    try:
        from groq import Groq
        from pdf2image import convert_from_path
    except Exception:
        return None, None
    return Groq, convert_from_path


def _should_try_vision(text: str, page_num: int, doc_meta: DocMetadata) -> bool:
    if not VISION_ENABLED or not os.getenv("GROQ_API_KEY"):
        return False
    if FIGURE_CAPTION_RE.search(text):
        return True
    if doc_meta.doc_type == "EPAR" and 10 <= page_num <= 35:
        return True
    return False


def describe_figure_page(pdf_path: str, page_num: int) -> str | None:
    Groq, convert_from_path = _vision_dependencies()
    if Groq is None or convert_from_path is None:
        return None
    try:
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=VISION_DPI)
        if not images:
            return None
        buf = io.BytesIO()
        image = images[0].convert("RGB")
        image.thumbnail(VISION_MAX_IMAGE_SIZE)
        image.save(buf, format="JPEG", quality=90, optimize=True)
        img_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        models = [VISION_MODEL]
        if VISION_FALLBACK_MODEL and VISION_FALLBACK_MODEL != VISION_MODEL:
            models.append(VISION_FALLBACK_MODEL)
        for model in models:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ],
                    }],
                    temperature=0,
                    max_completion_tokens=700,
                )
                description = (response.choices[0].message.content or "").strip()
                return None if description == "NO_FIGURE" else description
            except Exception as exc:
                print(f"[figure_extractor] Groq vision error on p.{page_num} with {model}: {exc}")
        return None
    except Exception as exc:
        print(f"[figure_extractor] Vision error on p.{page_num}: {exc}")
        return None


def extract_figure_chunks(filepath: str, doc_meta: DocMetadata) -> list[ChunkMetadata]:
    chunks: list[ChunkMetadata] = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                vision_desc = describe_figure_page(filepath, page_num) if _should_try_vision(text, page_num, doc_meta) else None
                matched_caption = False
                for match in FIGURE_CAPTION_RE.finditer(text):
                    matched_caption = True
                    caption = re.sub(r"\s+", " ", match.group(1)).strip()
                    visual = f"Visual description: {vision_desc}" if vision_desc else f"To inspect the figure, open page {page_num} of {doc_meta.filename}."
                    chunk_text = expand_for_indexing(f"[Figure on page {page_num} of {doc_meta.filename}]\nCaption: {caption}\n{visual}")
                    chunks.append(ChunkMetadata(
                        doc_id=doc_meta.doc_id, filename=doc_meta.filename, doc_type=doc_meta.doc_type,
                        regulatory_body=doc_meta.regulatory_body, product_name=doc_meta.product_name, api=doc_meta.api,
                        formulation=doc_meta.formulation, route=doc_meta.route, version_date=doc_meta.version_date,
                        section_title=caption[:100], section_number="figure", page_number=page_num,
                        chunk_index=len(chunks), token_count=len(chunk_text.split()), text=chunk_text, chunk_kind="figure",
                    ))
                if vision_desc and not matched_caption:
                    chunk_text = expand_for_indexing(
                        f"[Figure on page {page_num} of {doc_meta.filename}]\nVisual description: {vision_desc}"
                    )
                    chunks.append(ChunkMetadata(
                        doc_id=doc_meta.doc_id, filename=doc_meta.filename, doc_type=doc_meta.doc_type,
                        regulatory_body=doc_meta.regulatory_body, product_name=doc_meta.product_name, api=doc_meta.api,
                        formulation=doc_meta.formulation, route=doc_meta.route, version_date=doc_meta.version_date,
                        section_title=f"Figure (p.{page_num})", section_number="figure", page_number=page_num,
                        chunk_index=len(chunks), token_count=len(chunk_text.split()), text=chunk_text, chunk_kind="figure",
                    ))
    except Exception as exc:
        print(f"[figure_extractor] Error extracting figures from {doc_meta.filename}: {exc}")
    return chunks
