"""Direct lookup over structured EPAR clinical-trial table rows."""

from __future__ import annotations

import json
import re
from functools import lru_cache

from pharmalens.ingest.clinical_trial_table_extractor import CLINICAL_ROWS_PATH
from pharmalens.models import QueryResponse, RetrievedChunk


TRIAL_RE = re.compile(r"\b(SUSTAIN\s+FORTE|SUSTAIN\s*\d+|STEP\s*\d+|STRIDE)\b", re.I)
DOSE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*mg\b", re.I)
NUMERIC_QUERY_TERMS = (
    "what", "how many", "percentage", "percent", "%", "reduction", "change", "difference",
    "baseline", "hba", "hba1c", "glycated", "body weight", "weight loss", "fpg", "auc",
    "cmax", "endpoint", "trial", "population", "itt", "patients", "ci", "confidence",
    "p-value", "p value", "week",
)


@lru_cache(maxsize=1)
def load_clinical_trial_rows() -> list[dict]:
    if not CLINICAL_ROWS_PATH.exists():
        return []
    return json.loads(CLINICAL_ROWS_PATH.read_text())


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _norm_trial(text: str) -> str:
    text = re.sub(r"\s+", " ", text.upper()).strip()
    text = re.sub(r"SUSTAIN (\d+)", r"SUSTAIN \1", text)
    text = re.sub(r"STEP (\d+)", r"STEP \1", text)
    return text


def _trial_from_query(query: str) -> str | None:
    match = TRIAL_RE.search(query)
    return _norm_trial(match.group(1)) if match else None


def _dose_from_query(query: str) -> str | None:
    doses = [f"{float(match.group(1)):g} mg" for match in DOSE_RE.finditer(query)]
    if not doses:
        return None
    # Prefer the dose attached to semaglutide when present; otherwise use first dose.
    sema_match = re.search(r"semaglutide\s*(\d+(?:\.\d+)?)\s*mg", query, re.I)
    if sema_match:
        return f"{float(sema_match.group(1)):g} mg"
    return doses[0]


def _drug_mentions(query: str) -> list[str]:
    lower = query.lower()
    drugs = []
    for drug in ("semaglutide", "placebo", "sitagliptin", "dulaglutide", "exenatide", "insulin glargine", "liraglutide", "ozempic"):
        if drug in lower:
            drugs.append(drug)
    return drugs


def is_structured_numerical_trial_query(query: str) -> bool:
    lower = query.lower()
    return bool(_trial_from_query(query)) and any(term in lower for term in NUMERIC_QUERY_TERMS)


def _endpoint_predicate(query: str):
    lower = query.lower()
    if "itt" in lower or "population" in lower or "how many patients" in lower or re.search(r"\bpatients were in\b", lower):
        return lambda endpoint: endpoint.lower() == "population (n)"
    if ("hba" in lower or "glycated" in lower) and ("<7" in lower or "below 7" in lower or "achiev" in lower):
        return lambda endpoint: "patients (%) achieving hba1c <7%" in endpoint.lower()
    if "hba" in lower or "glycated" in lower:
        if "baseline" in lower:
            return lambda endpoint: "hba1c" in endpoint.lower() and "baseline" in endpoint.lower()
        if "difference" in lower and ("vs" in lower or "versus" in lower):
            return lambda endpoint: "hba1c" in endpoint.lower() and "treatment difference" in endpoint.lower()
        return lambda endpoint: "hba1c" in endpoint.lower() and "change from baseline" in endpoint.lower()
    if "fpg" in lower or "fasting plasma glucose" in lower:
        if "baseline" in lower:
            return lambda endpoint: "fpg" in endpoint.lower() and "baseline" in endpoint.lower()
        return lambda endpoint: "fpg" in endpoint.lower() and "change from baseline" in endpoint.lower()
    if "body weight" in lower or "weight loss" in lower or "weight change" in lower or "weight" in lower:
        unit_required = None
        if "kg" in lower:
            unit_required = "(kg)"
        elif "%" in lower or "percent" in lower or "percentage" in lower:
            unit_required = "(%)"
        def unit_ok(endpoint: str) -> bool:
            return unit_required is None or unit_required in endpoint
        if "baseline" in lower:
            return lambda endpoint: "body weight" in endpoint.lower() and "baseline" in endpoint.lower() and unit_ok(endpoint)
        if "difference" in lower and ("vs" in lower or "versus" in lower):
            return lambda endpoint: "body weight" in endpoint.lower() and "treatment difference" in endpoint.lower() and unit_ok(endpoint)
        return lambda endpoint: "body weight" in endpoint.lower() and "change from baseline" in endpoint.lower() and unit_ok(endpoint)
    if "waist" in lower:
        return lambda endpoint: "waist circumference" in endpoint.lower()
    if "systolic" in lower or "blood pressure" in lower:
        return lambda endpoint: "systolic blood pressure" in endpoint.lower()
    if "maximum walking" in lower:
        return lambda endpoint: "maximum walking distance" in endpoint.lower()
    if "pain-free walking" in lower:
        return lambda endpoint: "pain-free walking distance" in endpoint.lower()
    return lambda endpoint: True


def _row_matches_query(row: dict, query: str, trial: str, endpoint_ok) -> bool:
    if row.get("trial") != trial or not endpoint_ok(row.get("endpoint", "")):
        return False
    lower = query.lower()
    dose = _dose_from_query(query)
    drugs = _drug_mentions(query)
    if dose and row.get("dose") and row["dose"] != dose:
        return False
    if "each arm" in lower or "for each arm" in lower:
        return True
    if drugs:
        row_drug = row.get("drug", "").lower()
        row_comparator = row.get("comparator", "").lower()
        if not any(drug in row_drug or drug in row_comparator for drug in drugs):
            return False
    if week_match := re.search(r"week\s+(\d+)", lower):
        if row.get("timepoint") not in {"", f"week {week_match.group(1)}"}:
            return False
    return True


def _related_rows(rows: list[dict], selected: list[dict], query: str) -> list[dict]:
    lower = query.lower()
    if not selected:
        return []
    trial = selected[0]["trial"]
    base_endpoints = {row["endpoint"].replace(" treatment difference", " change from baseline") for row in selected}
    table_ids = {row["table_id"] for row in selected}
    timepoints = {row["timepoint"] for row in selected}
    related = list(selected)
    if "vs" in lower or "versus" in lower or "compared" in lower:
        comparator_mentions = [drug for drug in _drug_mentions(query) if drug != "semaglutide"]
        requested_dose = _dose_from_query(query)
        for row in rows:
            if row["trial"] != trial or row["table_id"] not in table_ids or row["timepoint"] not in timepoints:
                continue
            row_endpoint_base = row["endpoint"].replace(" treatment difference", " change from baseline")
            if row_endpoint_base not in base_endpoints:
                continue
            row_is_mentioned_comparator = any(drug in row["drug"].lower() for drug in comparator_mentions)
            if requested_dose and row.get("dose") and row["dose"] != requested_dose and not row_is_mentioned_comparator:
                continue
            if row not in related and (
                "treatment difference" in row["endpoint"].lower()
                or row_is_mentioned_comparator
                or any(drug in row["comparator"].lower() for drug in comparator_mentions)
            ):
                related.append(row)
    return related


def _format_value(row: dict) -> str:
    value = row.get("value", "")
    ci = row.get("CI", "")
    p_value = re.sub(r"^[ab*]\s*", "", row.get("p_value", ""), flags=re.I)
    parts = [value]
    if ci:
        parts.append(f"95% CI {ci}")
    if p_value:
        parts.append(p_value)
    return " ".join(part for part in parts if part)


def _citation(row: dict) -> str:
    return f"[{row.get('filename', 'Unknown')}, {row.get('table_id', '')}, p.{row.get('page_number', '?')}]"


def _row_sentence(row: dict) -> str:
    arm = " ".join(part for part in (row.get("drug", ""), row.get("dose", "")) if part).strip()
    comparator = f" vs {row['comparator']}" if row.get("comparator") else ""
    return f"{row['trial']} {row['timepoint']}: {arm}{comparator}, {row['endpoint']} = {_format_value(row)}"


def structured_trial_answer(query: str) -> QueryResponse | None:
    if not is_structured_numerical_trial_query(query):
        return None
    trial = _trial_from_query(query)
    if not trial:
        return None
    rows = load_clinical_trial_rows()
    endpoint_ok = _endpoint_predicate(query)
    selected = [row for row in rows if _row_matches_query(row, query, trial, endpoint_ok)]
    if not selected:
        return None
    selected = _related_rows(rows, selected, query)
    selected = selected[:8]
    lines = ["Structured clinical-trial table lookup:"]
    for row in selected:
        lines.append(f"- {_row_sentence(row)} {_citation(row)}")
    text = "\n".join(lines)
    source = RetrievedChunk(
        chunk_id=f"structured_clinical_trial:{trial}:{abs(hash(text))}",
        text="\n".join(json.dumps(row, ensure_ascii=False) for row in selected),
        score=999.0,
        metadata={
            "filename": selected[0].get("filename", ""),
            "doc_type": "EPAR",
            "regulatory_body": "EMA",
            "product_name": "Ozempic" if trial.startswith(("SUSTAIN", "STRIDE")) else "Wegovy",
            "section_number": "5.1",
            "section_title": "Pharmacodynamic properties",
            "page_number": selected[0].get("page_number", 0),
            "chunk_kind": "structured_clinical_trial_table",
            "table_id": selected[0].get("table_id", ""),
        },
    )
    return QueryResponse(answer=text, sources=[source])
