from hc_nlp import spacy_helpers
import spacy

nlp = spacy.load("en_core_web_sm")


def test_correct_entity_boundaries():
    text = "There are 59 bulbs in Hungary."
    doc = nlp(text)
    spacy_annotations = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

    # these annotations are off compared to the correct Spacy ones
    incorrect_annotations = [(11, 12, "CARDINAL"), (22, 27, "GPE")]

    corrected_annotations = spacy_helpers.correct_entity_boundaries(
        nlp, text, incorrect_annotations
    )

    assert corrected_annotations == spacy_annotations
