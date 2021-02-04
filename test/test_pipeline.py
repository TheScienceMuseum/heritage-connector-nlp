from hc_nlp import pipeline, constants, io
import spacy
import os

nlp = spacy.load("en_core_web_sm")
nlp_aug = spacy.load("en_core_web_sm")
test_data_path = os.path.join(os.path.dirname(__file__), "2020-11-25-11-43-02.zip")


def test_MapEntityTypes():
    """
    Test that entity type mapping works by creating two pipelines, one of which has the mapper and one of which doesn't.
    """
    mapping = constants.SPACY_TO_HC_ENTITY_MAPPING
    mapping_inputs = set(mapping.keys())

    nlp_aug.add_pipe("map_entity_types")

    data = io.load_text_and_annotations_from_labelstudio(
        test_data_path, spacy_model=nlp, adjust_entity_boundaries=False
    )

    for text, annotations in data[0:100]:
        doc = nlp(text)
        doc_aug = nlp_aug(text)

        doc_entlabels = [ent.label_ for ent in doc.ents]
        doc_aug_entlabels = [ent.label_ for ent in doc_aug.ents]

        assert all(
            [
                ent_aug == mapping[ent] if ent in mapping_inputs else ent_aug == ent
                for ent, ent_aug in zip(doc_entlabels, doc_aug_entlabels)
            ]
        )


def test_thesaurus_matcher():
    """
    Test that the component can be created and imported into a new nlp object.
    """

    nlp = spacy.blank("en")
    thesaurus_path = os.path.join(os.path.dirname(__file__), "test_thesaurus.jsonl")

    nlp.add_pipe("thesaurus_matcher", config={"thesaurus_path": thesaurus_path})

    assert "thesaurus_matcher" in nlp.pipe_names


def test_pattern_matcher():
    """
    Test that the component can be created and imported into a new nlp object.
    """

    nlp = spacy.blank("en")
    nlp.add_pipe("pattern_matcher", config={"patterns": constants.DATE_PATTERNS})

    assert "pattern_matcher" in nlp.pipe_names


def test_date_matcher():
    """
    Test that the component can be created and imported into a new nlp object.
    """

    nlp = spacy.blank("en")
    nlp.add_pipe("date_matcher")

    assert "date_matcher" in nlp.pipe_names

    # doc = nlp("someone was born in the second to first centuries BC")
    # assert len(doc.ents) == 1
    # assert [ent.label_ for ent in doc.ents][0] == "DATE"


def test_document_normalizer_join_consecutive_ents_with_same_label():
    nlp = spacy.blank("en")

    doc = nlp("London Victoria Station is often plagued by delayed trains")
    doc.ents = [
        spacy.tokens.Span(doc, 0, 1, "FAC"),
        spacy.tokens.Span(doc, 1, 2, "FAC"),
        spacy.tokens.Span(doc, 2, 3, "FAC"),
    ]

    # NOTE: this is not how it would be used in practice but allows us to just test one method
    e_j = pipeline.EntityJoiner(nlp, "entity_joiner")
    doc_modified = e_j._join_consecutive_ents_with_same_label(doc)

    assert len(doc_modified.ents) == 1
    assert doc_modified.ents[0].start == 0
    assert doc_modified.ents[0].end == 3
    assert doc_modified.ents[0].label_ == "FAC"
    assert doc_modified.ents[0].text == "London Victoria Station"


def test_document_normalizer_join_comma_separated_locs():
    nlp = spacy.blank("en")

    doc = nlp("Kate used to live in Cairo, Egypt")
    doc.ents = [
        spacy.tokens.Span(doc, 5, 6, "LOC"),
        spacy.tokens.Span(doc, 7, 8, "LOC"),
    ]

    # NOTE: this is not how it would be used in practice but allows us to just test one method
    e_j = pipeline.EntityJoiner(nlp, "entity_joiner")
    doc_modified = e_j._join_comma_separated_locs(doc, loc_ent_labels=["LOC"])

    assert len(doc_modified.ents) == 1
    assert (doc_modified.ents[0].start, doc_modified.ents[0].end) == (5, 8)
    assert doc_modified.ents[0].label_ == "LOC"
    assert doc_modified.ents[0].text == "Cairo, Egypt"

    # test case where LOCs contain more than one token
    doc = nlp("She now lives in live in Greater London , United Kingdom")
    doc.ents = [
        spacy.tokens.Span(doc, 6, 8, "LOC"),
        spacy.tokens.Span(doc, 9, 11, "LOC"),
    ]
    doc_modified = e_j._join_comma_separated_locs(doc, loc_ent_labels=["LOC"])

    assert len(doc_modified.ents) == 1
    assert (doc_modified.ents[0].start, doc_modified.ents[0].end) == (6, 11)


def test_duplicate_entity_detector_person():
    nlp = spacy.blank("en")
    dupl_ent_detector = pipeline.DuplicateEntityDetector(
        nlp, "duplicate_entity_detector"
    )

    doc = nlp(
        "Joseph Henry (December 17, 1797 – May 13, 1878) was an American scientist who served as the first Secretary of the Smithsonian Institution. He was the secretary for the National Institute for the Promotion of Science, a precursor of the Smithsonian Institution.[1] He was highly regarded during his lifetime. While building electromagnets, Henry discovered the electromagnetic phenomenon of self-inductance."
    )

    doc.ents = [
        spacy.tokens.Span(doc, start - 1, end - 1, label)
        for (start, end, label) in [
            (1, 3, "PERSON"),
            (4, 9, "DATE"),
            (9, 13, "DATE"),
            (16, 17, "NORP"),
            (25, 28, "ORG"),
            (34, 42, "ORG"),
            (46, 49, "ORG"),
            (62, 63, "PERSON"),
        ]
    ]

    doc_modified = dupl_ent_detector._detect_duplicate_person_mentions(doc)

    # the start, end, text and label of each entity should not have changed
    assert all(
        [
            (
                doc.ents[idx].start,
                doc.ents[idx].end,
                doc.ents[idx].text,
                doc.ents[idx].label_,
            )
            == (
                doc_modified.ents[idx].start,
                doc_modified.ents[idx].end,
                doc_modified.ents[idx].text,
                doc_modified.ents[idx].label_,
            )
            for idx in range(len(doc.ents))
        ]
    )

    # assert [(ent, ent._.entity_co_occurrence, ent._.entity_duplicate) for ent in doc_modified.ents] == []

    assert (
        doc_modified.ents[0]._.entity_co_occurrence
        == "joseph_henry"
        == doc_modified.ents[7]._.entity_co_occurrence
    )
    assert doc_modified.ents[7]._.entity_duplicate is True
    assert all(
        [doc_modified.ents[idx]._.entity_duplicate is False for idx in range(0, 7)]
    )
