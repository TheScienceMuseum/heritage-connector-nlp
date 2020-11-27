import json
from zipfile import ZipFile
import os
from typing import Tuple, List
from hc_nlp.spacy_helpers import correct_entity_boundaries


def load_raw_labelstudio_results(results_path: str) -> dict:
    """
    Load results from Label Studio export (in the folder labelling/export/<date>.zip by default).

    Args:
        results_path (str): path to results

    Returns:
        dict: see Label Studio docs
    """

    if not results_path.endswith(".zip"):
        raise ValueError(
            "`results_path` must point to a .zip file exported by Label Studio, or equivalent."
        )

    if not os.path.exists(results_path):
        raise ValueError(f"Results at {results_path} do not exist.")

    with ZipFile(results_path, "r") as z:
        data = json.loads(z.read("result.json"))

    return data


def load_text_and_annotations_from_labelstudio(
    results_path: str, spacy_model=None, adjust_entity_boundaries=True
) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """
    Load text and completed annotations from Label Studio, in the form [("my text", [(1, 3, "ORG"), (4, 10, "PERSON")]), ...].
    `spacy_model` is required for tokenization if `adjust_entity_boundaries` is set to True.

    Args:
        results_path (str): path to results
        spacy_model: model used for tokenization
        adjust_entity_boundaries: whether to correct entity boundaries according to token boundaries found by the tokenizer.
            If true, will adjust the start and end of the location of each entity to the closest true start or end.

    Returns:
        List[str, List[Tuple[int, int, str]]]: text, annotations
    """

    docs = load_raw_labelstudio_results(results_path)
    results = []

    for doc in docs:
        text = doc["data"]["text"]

        annotations = []

        for completion in doc["completions"]:
            for annot in completion["result"]:
                annotations.append(
                    (
                        annot["value"]["start"],
                        annot["value"]["end"],
                        annot["value"]["labels"][0],
                    )
                )

        if adjust_entity_boundaries:
            annotations = correct_entity_boundaries(spacy_model, text, annotations)

        results.append((text, annotations))

    return results
