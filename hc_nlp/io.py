import json
from zipfile import ZipFile
import os
from typing import Tuple, List
from hc_nlp.spacy_helpers import correct_entity_boundaries


def load_raw_labelstudio_results(results_set: str) -> dict:
    """
    Load results specified by their date in the filename (the name of the zip file in 
    labelling/export/NAME.zip without '.zip').

    Args:
        results_set (str)

    Returns:
        dict: see Label Studio docs
    """

    file_name = os.path.join(
        os.path.dirname(__file__), f"../labelling/export/{results_set}.zip"
    )

    if not os.path.exists(file_name):
        raise ValueError(
            f"Results with date {results_set} (at {file_name}) do not exist."
        )

    with ZipFile(file_name, "r") as z:
        data = json.loads(z.read("result.json"))

    return data


def load_text_and_annotations_from_labelstudio(
    results_set: str, spacy_model=None, adjust_entity_boundaries=True
) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """
    Load text and completed annotations from Label Studio, in the form [("my text", [(1, 3, "ORG"), (4, 10, "PERSON")]), ...].
    `spacy_model` is required for tokenization if `adjust_entity_boundaries` is set to True.

    Args:
        results_set (str): name of the zip file in labelling/export/NAME.zip without '.zip'
        spacy_model: model used for tokenization
        adjust_entity_boundaries: whether to correct entity boundaries according to token boundaries found by the tokenizer.
            If true, will adjust the start and end of the location of each entity to the closest true start or end.

    Returns:
        List[str, List[Tuple[int, int, str]]]: text, annotations
    """

    docs = load_raw_labelstudio_results(results_set)
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
