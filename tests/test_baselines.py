from llm_classifier.baselines.models import build_models


def test_baseline_models_fit_and_predict(sample_frame):
    models = build_models({"random_state": 42, "tfidf": {"ngram_range": (1, 2)}})
    for model in models.values():
        model.fit(sample_frame["text"], sample_frame["label"])
        assert set(model.predict(["free prize", "meeting schedule"])) <= {"spam", "not_spam"}
