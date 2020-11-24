import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from hc_nlp.io import load_text_and_annotations_from_labelstudio


def test_ner(spacy_model, results_set: str = None, examples: list = None) -> dict:
    """
    Return precision, recall and F-score for a Spacy NER model based on a set of gold-standard
    labels returned by Label Studio.

    Args:
        spacy_model: model with NER component
        results_set (str): name of results set from labelling/export folder

    Returns:
        dict: keys ents_p; ents_r; ents_f; ents_per_type
    """

    assert "ner" in spacy_model.pipe_names

    if (results_set and examples) or (not results_set and not examples):
        raise ValueError("Please provide exactly one of `results_set` and `examples`.")

    elif results_set:
        examples = load_text_and_annotations_from_labelstudio(results_set, spacy_model)

    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = spacy_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot)
        pred_value = spacy_model(input_)
        scorer.score(pred_value, gold)

    entity_measures = ["ents_p", "ents_r", "ents_f", "ents_per_type"]
    all_results = scorer.scores

    return {k: all_results[k] for k in entity_measures}
