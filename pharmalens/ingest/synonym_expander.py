"""Bidirectional pharma synonym expansion for indexing and retrieval."""

from functools import lru_cache
import re

import yaml

from pharmalens.paths import CONFIG_DIR

_ENABLED: bool = True


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, list[str]], dict[str, str]]:
    data = yaml.safe_load((CONFIG_DIR / "synonyms.yaml").read_text()) or {}
    synonyms = data.get("synonyms", {})
    reverse_map: dict[str, str] = {}
    for canonical, aliases in synonyms.items():
        for alias in aliases:
            reverse_map[alias.lower()] = canonical
    return synonyms, reverse_map


def _whole_word(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def expand_for_indexing(text: str) -> str:
    """Append canonical terms found through aliases without altering source text."""
    if not _ENABLED:
        return text
    _, reverse_map = _load()
    additions: set[str] = set()
    lower = text.lower()
    for alias, canonical in reverse_map.items():
        if canonical.lower() not in lower and _whole_word(alias).search(text):
            additions.add(canonical)
    if not additions:
        return text
    return text + "\n[SYNONYMS: " + "; ".join(sorted(additions)) + "]"


def expand_query(query: str) -> str:
    """Append known synonyms/aliases so lexical retrieval can cross terminology gaps."""
    if not _ENABLED:
        return query
    synonyms, reverse_map = _load()
    additions: set[str] = set()
    lower = query.lower()
    if "hba1c" in lower or "hba 1c" in lower:
        additions.update({"HbA", "HbA 1c", "glycated haemoglobin"})

    for canonical, aliases in synonyms.items():
        if canonical.lower() in lower:
            additions.update(aliases)

    for alias, canonical in reverse_map.items():
        if _whole_word(alias).search(query):
            additions.add(canonical)
            additions.update(synonyms.get(canonical, []))

    additions = {item for item in additions if item.lower() not in lower}
    return query if not additions else query + " " + " ".join(sorted(additions))
