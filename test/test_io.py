import sys

sys.path.append("..")

from hc_nlp import io
import spacy
import os

nlp = spacy.load("en_core_web_sm")
test_data_path = os.path.join(os.path.dirname(__file__), "2020-11-25-11-43-02.zip")


def test_load_raw_labelstudio_results():
    data = io.load_raw_labelstudio_results(test_data_path)

    assert isinstance(data, list)
    assert isinstance(data[0], dict)

    assert all([item in data[0].keys() for item in ["completions", "data", "id"]])


def test_load_text_and_annotations_from_labelstudio():
    """
    Data should be in the format [[text, [(start1, end1, label1), (start2, end2, label2), ...]]]
    """
    data = io.load_text_and_annotations_from_labelstudio(
        test_data_path, spacy_model=nlp, adjust_entity_boundaries=False
    )

    # test that all 'text' parts can be converted to string
    assert all([str(item[0]) for item in data])

    # test that all starts and ends are integers
    assert [all([isinstance(x, int) for x in i[0:1]]) for item in data for i in item[1]]

    # test that all labels are strings
    assert all([isinstance(i[2], str) for item in data for i in item[1]])
