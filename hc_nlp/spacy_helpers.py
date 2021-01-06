"""
Functions to help with Spacy things.
"""

import spacy
import numpy as np
from typing import List, Union


def correct_entity_boundaries(
    spacy_model, text: str, annotations: List[tuple]
) -> List[tuple]:
    """
    Correct start and end positions of entities so that they are on token boundaries, with tokens
    defined by the tokenizer in `spacy_model`.

    Args:
        spacy_model: Spacy model with tokenizer, e.g. the result of spacy.load("en_core_web_sm")
        text (str)
        annotations (List[tuple]): e.g. [(93, 104, 'PERSON'), (113, 129, 'ORG')]. Class names do not 
            need to match the names in the NER component of `spacy_model`.

    Returns:
        List[tuple]: corrected annotations
    """

    # tokens is only needed for debugging
    # tokens = []
    starts = []
    ends = []

    for token in spacy_model(text):
        # tokens.append(token.text)
        starts.append(token.idx)
        ends.append(token.idx + len(token.text))

    new_annotations = []

    for start, end, label in annotations:
        corrected_start = starts[
            min(range(len(starts)), key=lambda i: abs(starts[i] - start))
        ]
        corrected_end = ends[min(range(len(ends)), key=lambda i: abs(ends[i] - end))]

        new_annotations.append((corrected_start, corrected_end, label))

    return new_annotations


def remove_duplicate_annotations(annotations: List[tuple]) -> List[tuple]:
    """
    Removes duplicate annotations which cause the spaCy error '[E103] Trying to set conflicting doc.ents'.

    Args:
        annotations (List[tuple]): e.g. [(93, 104, 'PERSON'), (113, 129, 'ORG'), (93, 104, 'PERSON')]

    Returns:
        List[tuple]: corrected list of annotations
    """

    return list(set(annotations))


def display_manual_annotations(text: str, annotations: List[tuple], **kwargs):
    """
    Use spacy.displacy to display manual annotations created using Label Studio.
    Kwargs are passed to `spacy.displacy.render`.

    Args:
        text (str): [description]
        annotations (List[tuple]): e.g. [(93, 104, 'PERSON'), (113, 129, 'ORG')]
    """

    data = {
        "text": text,
        "ents": [
            {"start": item[0], "end": item[1], "label": item[2]} for item in annotations
        ],
    }

    spacy.displacy.render(data, style="ent", manual=True, **kwargs)


def display_ner_annotations(
    text_or_doc: Union[str, spacy.tokens.Doc], spacy_model=None, **kwargs
):
    """
    Use spacy.display to display annotations created using the NER component of spacy_model or a doc object.
    Kwargs are passed to `spacy.displacy.render`.

    Args:
        text (Union[str, spacy.tokens.Doc]): string or Spacy Doc object. `spacy_model` is not required if Doc object is passed.
        spacy_model: must contain an NER component
        spacy_doc (spacy.tokens.Doc): spacy Doc object
    """

    # assert ("ner" in spacy_model.pipe_names) or ("entity_ruler" in spacy_model.pipe_names)

    if isinstance(text_or_doc, spacy.tokens.Doc):
        spacy.displacy.render(text_or_doc, style="ent", **kwargs)
    elif spacy_model:
        spacy.displacy.render(spacy_model(text_or_doc), style="ent", **kwargs)
    else:
        raise ValueError(
            "Please provide either a Spacy Doc or both a string and a spacy model."
        )
