"""Config-driven PDF classification and metadata extraction."""

import fnmatch
import hashlib
import re
from pathlib import Path

import pdfplumber
import yaml

from pharmalens.models import DocMetadata
from pharmalens.paths import CONFIG_DIR

with (CONFIG_DIR / "doc_types.yaml").open(encoding="utf-8") as handle:
    DOC_TYPES = yaml.safe_load(handle)["doc_types"]

KNOWN_APIS = {k: k.title() for k in ("semaglutide", "tirzepatide", "liraglutide", "dulaglutide", "metformin")}
KNOWN_PRODUCTS = {k: k.title() for k in ("ozempic", "wegovy", "rybelsus", "mounjaro", "zepbound", "victoza", "trulicity")}
FORMULATION_MAP = {
    r"\bsubcutaneous\b|\bsubcut\b|\bsc\b|\binjection\b|\bsolution for injection\b": "SubcutaneousSolution",
    r"\boral\b|\btablet\b|\bpill\b": "OralTablet",
    r"\bintravenous\b|\biv\b|\binfusion\b": "IntravenousInfusion",
}
ROUTE_MAP = {"SubcutaneousSolution": "SC", "OralTablet": "PO", "IntravenousInfusion": "IV"}
MONTH_MAP = {month.lower(): f"{index:02d}" for index, month in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"), 1
)}
DATE_PATTERNS = [
    re.compile(r"(" + "|".join(m.title() for m in MONTH_MAP) + r")\s+(\d{4})", re.I),
    re.compile(r"\b(20\d{2})[-/](0[1-9]|1[0-2])\b"),
    re.compile(r"\b(0[1-9]|1[0-2])/(20\d{2})\b"),
]


def compute_file_hash(filepath: str) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_first_page_text(filepath: str) -> str:
    try:
        with pdfplumber.open(filepath) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages[:2]).lower()
    except Exception:
        return ""


def classify_doc_type(filename: str, first_page_text: str) -> str:
    for type_name, config in DOC_TYPES.items():
        if type_name != "Unknown" and any(fnmatch.fnmatch(filename.lower(), pattern.lower()) for pattern in config.get("filename_patterns", [])):
            return type_name
    for type_name, config in DOC_TYPES.items():
        if type_name == "Unknown":
            continue
        keywords = config.get("first_page_keywords", [])
        required = min(2, len(keywords))
        if required and sum(keyword.lower() in first_page_text for keyword in keywords) >= required:
            return type_name
    return "Unknown"


def _known_value(text: str, values: dict[str, str]) -> str:
    return next((display for key, display in values.items() if key in text), "Unknown")


def extract_formulation(text: str) -> tuple[str, str]:
    for pattern, formulation in FORMULATION_MAP.items():
        if re.search(pattern, text, re.I):
            return formulation, ROUTE_MAP[formulation]
    return "Unknown", "Unknown"


def extract_version_date(text: str) -> str | None:
    for index, pattern in enumerate(DATE_PATTERNS):
        if match := pattern.search(text):
            first, second = match.groups()
            if index == 0:
                return f"{second}-{MONTH_MAP[first.lower()]}"
            if index == 2:
                return f"{second}-{first}"
            return f"{first}-{second}"
    return None


def classify(filepath: str) -> DocMetadata:
    path = Path(filepath)
    text = extract_first_page_text(filepath)
    doc_type = classify_doc_type(path.name, text)
    formulation, route = extract_formulation(text)
    try:
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
    except Exception:
        total_pages = 0
    return DocMetadata(
        doc_id=compute_file_hash(filepath), filename=path.name, filepath=str(path.resolve()),
        doc_type=doc_type, regulatory_body=DOC_TYPES[doc_type]["regulatory_body"],
        product_name=_known_value(text, KNOWN_PRODUCTS), api=_known_value(text, KNOWN_APIS),
        formulation=formulation, route=route, version_date=extract_version_date(text), total_pages=total_pages,
    )
