"""Regulatory modality tagging for should/must/may style language."""

from collections import Counter
from functools import lru_cache
import re

import yaml

from pharmalens.paths import CONFIG_DIR


@lru_cache(maxsize=1)
def _load() -> dict[str, list[str]]:
    data = yaml.safe_load((CONFIG_DIR / "synonyms.yaml").read_text()) or {}
    return data.get("modality", {})


def tag_sentence(sentence: str) -> str:
    sentence_lower = sentence.lower()
    keywords = _load()

    for phrase in keywords.get("PROHIBITED", []):
        if phrase in sentence_lower:
            return "PROHIBITED"

    for label in ("MANDATORY", "RECOMMENDED", "OPTIONAL", "PERMITTED"):
        for phrase in keywords.get(label, []):
            if re.search(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", sentence_lower):
                return label

    return "NEUTRAL"


def tag_chunk(text: str) -> dict:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    tagged = [{"sentence": sentence, "modality": tag_sentence(sentence)} for sentence in sentences]
    counts = Counter(item["modality"] for item in tagged if item["modality"] != "NEUTRAL")
    return {
        "dominant_modality": counts.most_common(1)[0][0] if counts else "NEUTRAL",
        "modality_sentences": tagged,
    }
