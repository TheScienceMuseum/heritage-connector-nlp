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
    Test that the component can be created and imported into a new nlp object,
    and that a doc object is returned from a pipeline with the component in last position.
    """

    nlp = spacy.load("en_core_web_sm")
    thesaurus_path = os.path.join(os.path.dirname(__file__), "test_thesaurus.jsonl")

    nlp.add_pipe("thesaurus_matcher", config={"thesaurus_path": thesaurus_path})
    doc = nlp("This is a test sentence.")

    assert "thesaurus_matcher" in nlp.pipe_names
    assert isinstance(doc, spacy.tokens.Doc)


def test_pattern_matcher():
    """
    Test that the component can be created and imported into a new nlp object,
    and that a doc object is returned from a pipeline with the component in last position.
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("pattern_matcher", config={"patterns": constants.DATE_PATTERNS})

    doc = nlp("This is a test sentence")

    assert "pattern_matcher" in nlp.pipe_names
    assert isinstance(doc, spacy.tokens.Doc)


def test_date_matcher():
    """
    Test that the component can be created and imported into a new nlp object,
    and that a doc object is returned from a pipeline with the component in last position.
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("date_matcher")

    doc = nlp("This is a test phrase written in February 2021.")

    assert "date_matcher" in nlp.pipe_names
    assert isinstance(doc, spacy.tokens.Doc)


def test_entity_joiner():
    """
    Test that the component can be created and imported into a new nlp object,
    and that a doc object is returned from a pipeline with the component in last position.
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("entity_joiner")

    doc = nlp(
        "This is a sentence with some entities like London, Paris and Apple Inc. in."
    )

    assert "entity_joiner" in nlp.pipe_names
    assert isinstance(doc, spacy.tokens.Doc)


def test_entity_joiner_join_consecutive_ents_with_same_label():
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

    doc = nlp("London Victoria Station is often plagued by delayed trains")
    doc.ents = [
        spacy.tokens.Span(doc, 0, 2, "FAC"),
        spacy.tokens.Span(doc, 2, 3, "FAC"),
    ]

    doc_modified = e_j._join_consecutive_ents_with_same_label(doc)

    assert len(doc_modified.ents) == 1
    assert doc_modified.ents[0].start == 0
    assert doc_modified.ents[0].end == 3
    assert doc_modified.ents[0].label_ == "FAC"
    assert doc_modified.ents[0].text == "London Victoria Station"


def test_entity_joiner_join_comma_separated_locs():
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

    # no entities apart from LOC entities are changed
    assert {
        (e.text, e.start, e.end, e.label_) for e in doc.ents if e.label_ != "LOC"
    } == {
        (e.text, e.start, e.end, e.label_)
        for e in doc_modified.ents
        if e.label_ != "LOC"
    }


def test_entity_joiner_join_and_separated_people():
    nlp = spacy.blank("en")
    doc = nlp(
        "Katharine and Charles Parsons had a lasting impact on electrical engineering beyond their lifetime. Charles' work on electrical turbines, supported by Katherine, led to the large-scale introduction of electricity generated by steam turbines."
    )
    doc.ents = [
        spacy.tokens.Span(doc, 0, 1, "PERSON"),
        spacy.tokens.Span(doc, 2, 4, "PERSON"),
        spacy.tokens.Span(doc, 15, 16, "PERSON"),
        spacy.tokens.Span(doc, 24, 25, "PERSON"),
    ]

    e_j = pipeline.EntityJoiner(nlp, "entity_joiner")
    doc_modified = e_j._detect_joined_person_entities(doc)

    assert len(doc_modified.ents) == len(doc.ents)
    assert [(ent.start, ent.end, ent.text, ent.label_) for ent in doc.ents] == [
        (ent.start, ent.end, ent.text, ent.label_) for ent in doc_modified.ents
    ]
    assert doc.ents[0]._.alt_ent_text == "Katharine Parsons"
    assert [ent._.alt_ent_text is None for ent in doc.ents[1:]]


def test_duplicate_entity_detector():
    """
    Test that the component can be created and imported into a new nlp object,
    and that a doc object is returned from a pipeline with the component in last position.
    """

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("duplicate_entity_detector")
    doc = nlp(
        "This is a sentence with some entities like London, Paris and Apple Inc. in."
    )

    assert "duplicate_entity_detector" in nlp.pipe_names
    assert isinstance(doc, spacy.tokens.Doc)


def test_entity_joiner_and_separated_people_and_duplicate_entity_detector_person():
    """Test that the `ent._.alt_ent_text` attribute set by EntityJoiner is used by DuplicateEntityDetector
    to exclude future mentions of that person. E.g.:
    a) 'Katharine and Charles Parsons' mentioned, followed by a separate mention of 'Katharine' later on
    b) EntityJoiner adds 'Katharine Parsons' as alt_ent_text attribute to 'Katharine'
    c) DuplicateEntityDetector uses alt_ent_text attribute to mark later mention of 'Katharine' as duplicated
    """

    nlp = spacy.blank("en")
    ent_joiner = pipeline.EntityJoiner(nlp, "entity_joiner")
    dupl_ent_detector = pipeline.DuplicateEntityDetector(
        nlp, "duplicate_entity_detector"
    )

    text = "Katharine and Charles Parsons had a lasting impact on electrical engineering beyond their lifetime. Charles' work on electrical turbines, supported by Katharine, led to the large-scale introduction of electricity generated by steam turbines. This is a system still in use today."
    doc = nlp(text)
    doc.ents = [
        spacy.tokens.Span(doc, 0, 1, "PERSON"),
        spacy.tokens.Span(doc, 2, 4, "PERSON"),
        spacy.tokens.Span(doc, 15, 16, "PERSON"),
        spacy.tokens.Span(doc, 24, 25, "PERSON"),
        spacy.tokens.Span(doc, 47, 48, "DATE"),
    ]
    modified_doc = dupl_ent_detector(ent_joiner(doc))

    assert len(modified_doc.ents) == len(doc.ents)

    # first 'Katharine' has attribute with her full name
    assert modified_doc.ents[0]._.alt_ent_text == "Katharine Parsons"

    # second 'Katharine' marked as duplicate
    assert modified_doc.ents[3]._.entity_duplicate is True


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

    assert (
        doc_modified.ents[0]._.entity_co_occurrence
        == "joseph_henry"
        == doc_modified.ents[7]._.entity_co_occurrence
    )
    assert doc_modified.ents[7]._.entity_duplicate is True
    assert all(
        [doc_modified.ents[idx]._.entity_duplicate is False for idx in range(0, 7)]
    )


def test_duplicate_entity_detector_org():
    nlp = spacy.blank("en")
    dupl_ent_detector = pipeline.DuplicateEntityDetector(
        nlp, "duplicate_entity_detector"
    )

    doc = nlp("Apple Newton eMate 300 laptop, made by Apple Inc, 1997.")

    doc.ents = [
        spacy.tokens.Span(doc, start - 1, end - 1, label)
        for (start, end, label, _) in [
            (1, 2, "ORG", "Apple"),
            (2, 5, "OBJECT", "Newton eMate 300"),
            (9, 11, "ORG", "Apple Inc"),
            (12, 13, "DATE", "1997"),
        ]
    ]

    doc_modified = dupl_ent_detector._detect_duplicate_org_mentions(doc)

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

    assert (
        doc_modified.ents[0]._.entity_co_occurrence
        == "apple_inc"
        == doc_modified.ents[2]._.entity_co_occurrence
    )
    assert doc_modified.ents[0]._.entity_duplicate is True
    assert all(
        [doc_modified.ents[idx]._.entity_duplicate is False for idx in range(1, -1)]
    )


def test_duplicate_entity_detector_loc():
    nlp = spacy.blank("en")
    dupl_ent_detector = pipeline.DuplicateEntityDetector(
        nlp, "duplicate_entity_detector"
    )

    doc = nlp(
        """A2A: http://www.a2a.org.uk/html/094-738.htm 
        1846-1859, coachbuilder, London  Hooper & Co. (Coachbuilders) Ltd of St James's St. and Park Royal, London NW10;  1808 - J and G Adams opened a coachmaking business at 57 Haymarket, London; 1811 - George Adams moved to 28 Haymarket  1833 - George Adams joined by George Hooper  1846 - became known as Hooper & Co, coachmakers, began trading at 28 Haymarket;  1867 the company moved their premises to Victoria Street and in 1897 they moved to 54 St James's Street. Hooper and Co closed in 1959, by which time they were a subsidiary company of the Birmingham Small Arms Company. 
        Hooper & Co. (Coachbuilders) Ltd of St James's St. and Park Royal, London NW10;  1808 - J and G Adams opened a coachmaking business at 57 Haymarket, London; 1811 - George Adams moved to 28 Haymarket  1833 - George Adams joined by George Hooper  1846 - became known as Hooper & Co, coachmakers, began trading at 28 Haymarket;  1867 the company moved their premises to Victoria Street and in 1897 they moved to 54 St James's Street. Hooper and Co closed in 1959, by which time they were a subsidiary company of the Birmingham Small Arms Company."""
    )

    doc.ents = [
        spacy.tokens.Span(doc, start, end, label)
        for (start, end, label, _) in [
            (20, 23, "LOC", "St James's"),
            (25, 27, "LOC", "Park Royal"),
            (44, 47, "LOC", "Haymarket, London"),
            (55, 56, "LOC", "Haymarket"),
            (81, 82, "LOC", "Haymarket"),
            (136, 139, "LOC", "St James's"),
            (141, 143, "LOC", "Park Royal"),
            (160, 163, "LOC", "Haymarket, London"),
            (171, 172, "LOC", "Haymarket"),
            (197, 198, "LOC", "Haymarket"),
        ]
    ]

    doc_modified = dupl_ent_detector._detect_duplicate_loc_mentions(doc)

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

    assert all(
        [
            doc_modified.ents[idx]._.entity_co_occurrence == "haymarket,_london"
            for idx in (2, 3, 4, 7, 8, 9)
        ]
    )
    assert all(
        [doc_modified.ents[idx]._.entity_duplicate is True for idx in (3, 4, 8, 9)]
    )
    assert all(
        [doc_modified.ents[idx]._.entity_duplicate is False for idx in (0, 1, 5, 6, 7)]
    )
