"""Extract structured clinical-trial result rows from EPAR table text.

The regular pdfplumber grid extractor catches simple tables, but the EPAR
clinical efficacy tables are often emitted as running text.  This module parses
the section 5.1 trial-result tables into normalized rows that can be looked up
without asking an LLM to reconstruct values from prose.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pharmalens.ingest.parser import extract_pages
from pharmalens.paths import DATA_DIR


TABLE_STORE_DIR = DATA_DIR / "tables"
CLINICAL_ROWS_PATH = TABLE_STORE_DIR / "clinical_trial_rows.json"


@dataclass(frozen=True)
class Arm:
    drug: str
    dose: str = ""


TRIAL_ARMS: dict[str, list[Arm]] = {
    "SUSTAIN 1": [Arm("Semaglutide", "0.5 mg"), Arm("Semaglutide", "1 mg"), Arm("Placebo")],
    "SUSTAIN 2": [Arm("Semaglutide", "0.5 mg"), Arm("Semaglutide", "1 mg"), Arm("Sitagliptin", "100 mg")],
    "SUSTAIN 3": [Arm("Semaglutide", "1 mg"), Arm("Exenatide ER", "2 mg")],
    "SUSTAIN 4": [Arm("Semaglutide", "0.5 mg"), Arm("Semaglutide", "1 mg"), Arm("Insulin glargine")],
    "SUSTAIN 5": [Arm("Semaglutide", "0.5 mg"), Arm("Semaglutide", "1 mg"), Arm("Placebo")],
    "SUSTAIN 7": [
        Arm("Semaglutide", "0.5 mg"),
        Arm("Semaglutide", "1 mg"),
        Arm("Dulaglutide", "0.75 mg"),
        Arm("Dulaglutide", "1.5 mg"),
    ],
    "SUSTAIN FORTE": [Arm("Semaglutide", "1 mg"), Arm("Semaglutide", "2 mg")],
    "SUSTAIN 9": [Arm("Semaglutide", "1 mg"), Arm("Placebo")],
    "STRIDE": [Arm("Ozempic", "1 mg"), Arm("Placebo")],
    "STEP 1": [Arm("Semaglutide", "2.4 mg"), Arm("Placebo")],
    "STEP 2": [Arm("Semaglutide", "2.4 mg"), Arm("Placebo")],
    "STEP 3": [Arm("Semaglutide", "2.4 mg"), Arm("Placebo")],
    "STEP 4": [Arm("Semaglutide", "2.4 mg"), Arm("Placebo")],
    "STEP 5": [Arm("Semaglutide", "2.4 mg"), Arm("Placebo")],
    "STEP 8": [Arm("Semaglutide", "2.4 mg"), Arm("Liraglutide", "3 mg")],
    "STEP 9": [Arm("Semaglutide", "2.4 mg"), Arm("Placebo")],
}


TRIAL_RE = re.compile(r"\b(SUSTAIN\s+FORTE|SUSTAIN\s+\d+|STEP\s+\d+|STRIDE)\b", re.I)
TABLE_TITLE_RE = re.compile(
    r"Table\s+(?P<table_no>\d+)\s*:?\s*(?P<title>(?:SUSTAIN\s+FORTE|SUSTAIN\s+\d+|STEP\s+\d+|STRIDE)[^\n:]*:?[\s\S]*?)(?=\n(?:Table\s+\d+|Figure\s+\d+|SUSTAIN\s+\d+|SUSTAIN\s+FORTE|STEP\s+\d+|Combination with|Cardiovascular disease|Effect on body composition|$))",
    re.I,
)
NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?")
CI_RE = re.compile(r"\[([^\]]+)\]")


def _clean(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _normal_trial(value: str) -> str:
    return re.sub(r"\s+", " ", value.upper()).replace("SUSTAIN FORTE", "SUSTAIN FORTE").strip()


def _page_for_offset(page_offsets: list[tuple[int, int]], offset: int) -> int:
    page = page_offsets[0][0] if page_offsets else 0
    for page_num, start in page_offsets:
        if start <= offset:
            page = page_num
        else:
            break
    return page


def _document_text_with_offsets(pdf_path: Path) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    for page_num, page_text in extract_pages(str(pdf_path)):
        offsets.append((page_num, sum(len(part) for part in parts)))
        parts.append(f"\n{page_text or ''}\n")
    return "".join(parts), offsets


def _numeric_values(line: str) -> list[str]:
    values: list[str] = []
    for match in NUMBER_RE.finditer(line):
        value = _clean(match.group(0)).replace(",", "").rstrip("abc*")
        if value and value not in {"1", "2", "3"}:
            values.append(value)
    return values


def _values_for_arms(line: str, arm_count: int) -> list[str]:
    values = _numeric_values(line)
    # PDF text often writes thousands as "1 306"; after tokenization this
    # becomes ["1", "306"].  Combine only when that split would otherwise leave
    # exactly one surplus token for the expected number of treatment arms.
    if (
        len(values) == arm_count + 1
        and values[0].isdigit()
        and values[1].isdigit()
        and len(values[0]) == 1
        and len(values[1]) == 3
    ):
        values = [values[0] + values[1], *values[2:]]
    return values[-arm_count:]


def _strip_label_numbers(label: str, line: str) -> str:
    """Drop the row label before value parsing so week/percent labels do not become values."""
    index = line.lower().find(label.lower())
    if index >= 0:
        return line[index + len(label):]
    return line


def _p_value(block: str) -> str:
    match = re.search(r"\b[ab*]?\s*p\s*<\s*0\.\d+", block, re.I)
    return _clean(match.group(0)) if match else ""


def _comparison_name(line: str) -> str:
    match = re.search(r"Difference (?:\([^)]+\) )?from ([^\[]+)", line, re.I)
    if not match:
        match = re.search(r"Treatment (?:ratio|difference).*", line, re.I)
        return "Placebo" if "placebo" in line.lower() else ""
    comparator = _clean(match.group(1))
    comparator = re.sub(r"\d+$", "", comparator).strip()
    return comparator


def _split_estimates_with_ci(line: str, arm_count: int) -> list[tuple[str, str]]:
    """Return estimate/CI pairs for active arms, preserving bracketed intervals."""
    tail = line
    tail = tail.replace("Treatment ratio (HL Estimate)", "").replace("Treatment difference (HL Estimate)", "")
    pairs = [
        (m.group(1), _clean(m.group(2)).rstrip("abc*"))
        for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*\[((?=[^\]]*[,;])[-+]?\d[0-9.,; \-+]*)\]", tail)
    ]
    if pairs:
        rows = [(_clean(value).rstrip("abc*"), ci) for value, ci in pairs]
        while len(rows) < arm_count:
            rows.append(("-", ""))
        return rows[:arm_count]
    values = _numeric_values(tail)
    rows = [(value.rstrip("abc*"), "") for value in values[-arm_count:]]
    while len(rows) < arm_count:
        rows.append(("-", ""))
    return rows[:arm_count]


def _endpoint_from_heading(heading: str) -> str:
    heading = _clean(heading)
    heading = re.sub(r"\bHbA\b", "HbA1c", heading, flags=re.I)
    heading = re.sub(r"\bHbA\s*1c\b", "HbA1c", heading, flags=re.I)
    if heading.lower() == "hba (%)":
        return "HbA1c (%)"
    return heading


def _row(
    *,
    trial: str,
    arm: Arm,
    comparator: str,
    endpoint: str,
    timepoint: str,
    value: str,
    ci: str = "",
    p_value: str = "",
    filename: str,
    page_number: int,
    table_id: str,
) -> dict:
    return {
        "trial": trial,
        "drug": arm.drug,
        "dose": arm.dose,
        "comparator": comparator,
        "endpoint": endpoint,
        "timepoint": timepoint,
        "value": value,
        "CI": ci,
        "p_value": p_value,
        "filename": filename,
        "page_number": page_number,
        "table_id": table_id,
    }


def _rows_for_values(
    *,
    trial: str,
    arms: list[Arm],
    endpoint: str,
    timepoint: str,
    values: Iterable[str],
    filename: str,
    page_number: int,
    table_id: str,
    comparator: str = "",
    p_value: str = "",
) -> list[dict]:
    rows = []
    for arm, value in zip(arms, values):
        if value == "-":
            continue
        rows.append(_row(
            trial=trial,
            arm=arm,
            comparator=comparator,
            endpoint=endpoint,
            timepoint=timepoint,
            value=value,
            filename=filename,
            page_number=page_number,
            table_id=table_id,
            p_value=p_value,
        ))
    return rows


def _parse_table_block(block: str, *, filename: str, page_number: int, table_number: str) -> list[dict]:
    block = _clean(block)
    trial_match = TRIAL_RE.search(block)
    if not trial_match:
        return []
    trial = _normal_trial(trial_match.group(1))
    arms = TRIAL_ARMS.get(trial)
    if not arms:
        return []
    time_match = re.search(r"week\s+(\d+)", block, re.I)
    default_timepoint = f"week {time_match.group(1)}" if time_match else ""
    table_id = f"{Path(filename).stem}_table_{table_number}_{trial.lower().replace(' ', '_')}"
    p_value = _p_value(block)
    rows: list[dict] = []
    current_endpoint = ""
    pending_difference = ""

    for raw_line in block.splitlines():
        line = _clean(raw_line)
        if not line or line.lower() in {"1c", "ci]"}:
            continue
        if pending_difference:
            line = f"{pending_difference} {line}"
            pending_difference = ""
        if re.search(r"Difference .*95%$", line, re.I) or re.search(r"Difference .*95%\s*$", line, re.I):
            pending_difference = line
            continue

        endpoint_heading = None
        if re.fullmatch(r"(HbA|HbA1c) \(%\)|FPG \(mmol/L\)|Body weight(?: \(kg\))?|Waist circumference \(cm\)|Systolic blood pressure ?\(mmHg\)|Maximum walking distance \(meters\)|Pain-free walking distance \(meters\),? week \d+|VascuQoL-6 total score,? week \d+", line, re.I):
            endpoint_heading = line
        if endpoint_heading:
            current_endpoint = _endpoint_from_heading(endpoint_heading)
            continue

        if line.startswith(("Semaglutide", "Placebo", "Dulaglutide", "Sitagliptin", "Exenatide", "Insulin", "Ozempic", "Liraglutide")):
            continue
        if line.startswith(("ap", "bp", "* p", "HL =", "CI =", "confidence interval", "Estimated using", "During the trial")):
            continue

        if "Intent-to-Treat" in line or "Full analysis set" in line or "Intention-to-treat" in line:
            values = _values_for_arms(line, len(arms))
            rows.extend(_rows_for_values(
                trial=trial, arms=arms, endpoint="Population (N)", timepoint="baseline",
                values=values, filename=filename, page_number=page_number, table_id=table_id,
            ))
            continue

        if re.match(r"Baseline", line, re.I) and current_endpoint:
            values = _values_for_arms(_strip_label_numbers("Baseline", line), len(arms))
            rows.extend(_rows_for_values(
                trial=trial, arms=arms, endpoint=f"{current_endpoint} baseline", timepoint="baseline",
                values=values, filename=filename, page_number=page_number, table_id=table_id,
            ))
            continue

        if re.match(r"Change .*from baseline", line, re.I) and current_endpoint:
            tp_match = re.search(r"week\s+(\d+)", line, re.I)
            timepoint = f"week {tp_match.group(1)}" if tp_match else default_timepoint
            label = re.match(r"Change .*?baseline(?:\d| at week \d+|1,?\s*2|1|2)?", line, re.I)
            values_text = line[label.end():] if label else line
            values = _values_for_arms(values_text, len(arms))
            endpoint = current_endpoint
            if current_endpoint.lower() == "body weight":
                if re.search(r"Change \(%\)", line, re.I):
                    endpoint = "Body weight (%)"
                elif re.search(r"Change \(kg\)", line, re.I):
                    endpoint = "Body weight (kg)"
            rows.extend(_rows_for_values(
                trial=trial, arms=arms, endpoint=f"{endpoint} change from baseline",
                timepoint=timepoint, values=values, filename=filename, page_number=page_number, table_id=table_id,
            ))
            continue

        if re.match(r"Patients \(%\)", line, re.I):
            label_match = re.match(r"(Patients \(%\).*?)(?=\s[-+]?\d)", line, re.I)
            endpoint = _endpoint_from_heading(label_match.group(1) if label_match else "Patients (%)")
            values_text = line[label_match.end():] if label_match else line
            values = [value.rstrip("*") for value in _values_for_arms(values_text, len(arms))]
            rows.extend(_rows_for_values(
                trial=trial, arms=arms, endpoint=endpoint, timepoint=default_timepoint,
                values=values, filename=filename, page_number=page_number, table_id=table_id,
            ))
            continue

        if re.match(r"(Difference|Treatment ratio|Treatment difference)", line, re.I):
            comparator = _comparison_name(line)
            pairs = _split_estimates_with_ci(line, len(arms))
            endpoint_base = current_endpoint
            if current_endpoint.lower() == "body weight":
                if re.search(r"Difference \(%\)", line, re.I):
                    endpoint_base = "Body weight (%)"
                elif re.search(r"Difference \(kg\)", line, re.I):
                    endpoint_base = "Body weight (kg)"
            endpoint = f"{endpoint_base} treatment difference" if endpoint_base else "Treatment difference"
            for arm, (value, ci) in zip(arms, pairs):
                if value == "-":
                    continue
                rows.append(_row(
                    trial=trial, arm=arm, comparator=comparator, endpoint=endpoint,
                    timepoint=default_timepoint, value=value, ci=ci, p_value=p_value,
                    filename=filename, page_number=page_number, table_id=table_id,
                ))
            continue

    return rows


def extract_clinical_trial_rows_from_pdf(pdf_path: Path) -> list[dict]:
    text, page_offsets = _document_text_with_offsets(pdf_path)
    rows: list[dict] = []
    for match in TABLE_TITLE_RE.finditer(text):
        title = match.group("title")
        if not TRIAL_RE.search(title):
            continue
        rows.extend(_parse_table_block(
            match.group(0),
            filename=pdf_path.name,
            page_number=_page_for_offset(page_offsets, match.start()),
            table_number=match.group("table_no"),
        ))
    return rows


def extract_all_clinical_trial_rows(documents_dir: Path | None = None) -> list[dict]:
    documents_dir = documents_dir or DATA_DIR / "documents"
    rows: list[dict] = []
    for pdf_path in sorted(documents_dir.glob("*epar*product-information*.pdf")):
        rows.extend(extract_clinical_trial_rows_from_pdf(pdf_path))
    return rows


def write_clinical_trial_store(rows: list[dict] | None = None) -> Path:
    TABLE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    rows = rows if rows is not None else extract_all_clinical_trial_rows()
    CLINICAL_ROWS_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return CLINICAL_ROWS_PATH


def main() -> None:
    path = write_clinical_trial_store()
    rows = json.loads(path.read_text())
    print({"clinical_trial_rows": len(rows), "path": str(path)})


if __name__ == "__main__":
    main()
