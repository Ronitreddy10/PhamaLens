from pharmalens.ingest.chunker import CHUNK_OVERLAP, CHUNK_SIZE, parse_epar_smpc
from pharmalens.models import DocMetadata


def metadata() -> DocMetadata:
    return DocMetadata(doc_id="abc", filename="test.pdf", filepath="test.pdf", doc_type="EPAR",
                       regulatory_body="EMA", api="Semaglutide", total_pages=2)


def test_epar_sections_preserve_page_and_heading():
    chunks = parse_epar_smpc([(1, "4.1 Therapeutic indications\nIndication text"),
                              (2, "4.2 Posology and method of administration\nDose text")], metadata())
    assert [chunk.section_number for chunk in chunks] == ["4.1", "4.2"]
    assert [chunk.page_number for chunk in chunks] == [1, 2]


def test_long_section_overlaps_and_is_bounded():
    text = "4.1 Long section\n" + "semaglutide dosage information. " * 1000
    chunks = parse_epar_smpc([(1, text)], metadata())
    assert len(chunks) > 1
    assert all(chunk.token_count <= CHUNK_SIZE for chunk in chunks)
    assert CHUNK_OVERLAP > 0


def test_chunker_adds_synonym_and_cross_reference():
    chunks = parse_epar_smpc([(1, "4.1 Effects\nHbA1c data. See section 5.2 for details.")], metadata())
    assert "glycated haemoglobin" in chunks[0].text
    assert "CROSS_REFERENCES" in chunks[0].text
