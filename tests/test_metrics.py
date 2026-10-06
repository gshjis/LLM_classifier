from llm_classifier.metrics import calculate_metrics


def test_metrics_identify_perfect_predictions():
    metrics = calculate_metrics(["spam", "not_spam"], ["spam", "not_spam"])
    assert metrics["accuracy"] == 1.0
    assert metrics["precision_spam"] == 1.0
    assert metrics["recall_spam"] == 1.0
