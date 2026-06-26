"""Chroma metadata-filter helpers."""


ALLOWED_FILTERS = {"doc_type", "regulatory_body", "product_name", "api", "formulation", "route", "version_date"}


def build_metadata_where(filters: dict | None) -> dict | None:
    conditions = [
        {key: {"$eq": value}}
        for key, value in (filters or {}).items()
        if key in ALLOWED_FILTERS and value not in (None, "", "Any", "Unknown")
    ]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}
