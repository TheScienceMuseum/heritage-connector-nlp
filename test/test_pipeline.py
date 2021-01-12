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
