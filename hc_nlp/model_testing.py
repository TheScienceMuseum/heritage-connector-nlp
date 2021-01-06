import spacy
from spacy.training import Example
from spacy.scorer import Scorer
from typing import List, Tuple
from collections import Counter

from hc_nlp.io import load_text_and_annotations_from_labelstudio
from hc_nlp import logging, errors, spacy_helpers

logger = logging.get_logger(__name__)


def test_ner(
    spacy_model,
    results_set: str = None,
    examples: List[List[Tuple[str, List[Tuple[int, int, str]]]]] = None,
) -> dict:
    """
    Return precision, recall and F-score for a Spacy NER model based on a set of gold-standard
    labels returned by Label Studio. One of `results_set` and `examples` should be provided.

    Args:
        spacy_model: model with NER component
        results_set (str, optional): name of results set from labelling/export folder
        examples (List[List[Tuple[str, List[Tuple[int, int, str]]]]], optional): from `io.load_text_and_annotations_from_labelstudio`

    Returns:
        dict: keys ents_p; ents_r; ents_f; ents_per_type
    """

    if "ner" not in spacy_model.pipe_names:
        errors.raise_spacy_component_does_not_exist("ner")

    if (results_set and examples) or (not results_set and not examples):
        raise ValueError("Please provide exactly one of `results_set` and `examples`.")

    elif results_set:
        examples = load_text_and_annotations_from_labelstudio(results_set, spacy_model)

    scorer = Scorer()
    results = []
    for input_, annot in examples:
        pred_value = spacy_model(input_)
        annot = spacy_helpers.remove_duplicate_annotations(annot)
        try:
            gold = Example.from_dict(pred_value, {"entities": annot})
        except Exception as e:
            print("Failed: ", pred_value)
            # print(annot)
            print(e)
            continue
        results.append(gold)
    score_res = scorer.score(results)

    entity_measures = ["ents_p", "ents_r", "ents_f", "ents_per_type"]
    ent_results = {k: score_res[k] for k in entity_measures}

    label_count = _count_labels(examples)
    total_support = 0
    labels_with_support = []

    for label, count in label_count.items():
        if label in ent_results["ents_per_type"].keys():
            ent_results["ents_per_type"][label]["support"] = count
            labels_with_support.append(label)
            total_support += count

        else:
            logger.warn(
                f"Label {label} not in the model. To get a list of labels in a Spacy model `nlp` run `nlp.entity.labels`."
            )

    ent_results["support"] = total_support
    ent_results["labels_missing_from_annotations"] = list(
        set(ent_results["ents_per_type"].keys()) - set(labels_with_support)
    )
    ent_results["ents_per_type"] = {
        k: ent_results["ents_per_type"][k] for k in labels_with_support
    }

    return ent_results


def _count_labels(examples: List[List[Tuple[str, List[Tuple[int, int, str]]]]]) -> dict:
    """
    Return the number of examples present for each label in examples.

    Args:
        examples (List[List[Tuple[str, List[Tuple[int, int, str]]]]]): ([text, [(annotation_start, annotation_end, annotation_label), ...]])

    Returns:
        dict: {annotation_label: label_count, ...}
    """

    label_list = [item[2] for ex in examples for item in ex[1]]

    return dict(Counter(label_list))
