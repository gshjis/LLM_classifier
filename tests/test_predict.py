from llm_classifier.llm.dataset import ID_TO_LABEL, LABEL_TO_ID


def test_llm_label_mappings_are_consistent():
    assert LABEL_TO_ID == {"not_spam": 0, "spam": 1}
    assert {value: key for key, value in LABEL_TO_ID.items()} == ID_TO_LABEL
