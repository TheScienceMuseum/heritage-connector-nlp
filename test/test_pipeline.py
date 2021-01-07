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

    nlp_aug.add_pipe("MapEntityTypes")

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
