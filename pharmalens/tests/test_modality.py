from pharmalens.ingest.modality_tagger import tag_chunk, tag_sentence
from pharmalens.ingest.synonym_expander import expand_for_indexing, expand_query
from pharmalens.retrieval.disambiguator import detect_product_intent


def test_synonym_expands_abbreviation_to_canonical():
    assert "glycated haemoglobin" in expand_for_indexing("HbA1c decreased.")


def test_synonym_expands_canonical_to_aliases():
    expanded = expand_query("glycated haemoglobin reduction")
    assert "HbA1c" in expanded


def test_modality_prohibited_beats_mandatory():
    assert tag_sentence("The product must not be used in this population.") == "PROHIBITED"


def test_chunk_dominant_modality():
    assert tag_chunk("The applicant may use this design. It may be considered.")["dominant_modality"] == "OPTIONAL"


def test_disambiguates_wegovy():
    intent = detect_product_intent("What is Wegovy dosing?")
    assert intent["status"] == "clear"
    assert intent["product"] == "Wegovy"
