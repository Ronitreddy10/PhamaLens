from pharmalens.retrieval.hybrid import rrf_score
from pharmalens.retrieval.metadata_filter import build_metadata_where


def test_rrf_decreases_with_rank():
    assert rrf_score(0) > rrf_score(5)


def test_metadata_where_ignores_unknown_and_rejects_unapproved_keys():
    assert build_metadata_where({"regulatory_body": "FDA", "api": "Unknown", "oops": "x"}) == {
        "regulatory_body": {"$eq": "FDA"}
    }


def test_multiple_filters_use_and():
    where = build_metadata_where({"regulatory_body": "FDA", "doc_type": "PSG"})
    assert "$and" in where and len(where["$and"]) == 2


def test_ingest_is_idempotent(monkeypatch):
    from pharmalens.ingest import watcher
    from pharmalens.models import DocMetadata

    metadata = DocMetadata(doc_id="same-hash", filename="same.pdf", filepath="same.pdf",
                           doc_type="PSG", regulatory_body="FDA")
    monkeypatch.setattr(watcher, "classify", lambda _: metadata)
    monkeypatch.setattr(watcher, "is_already_indexed", lambda _: True)
    monkeypatch.setattr(watcher, "chunk_document", lambda *_: (_ for _ in ()).throw(AssertionError("must not chunk")))
    assert watcher.ingest_file("same.pdf")["status"] == "skipped"
