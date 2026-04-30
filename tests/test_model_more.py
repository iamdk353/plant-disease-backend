import numpy as np

from app import model
from app.state import state


def test_preprocess_image_efficientnet_uses_preprocess_input(sample_image_bytes, monkeypatch):
    called = {"count": 0}

    def fake_preprocess(arr):
        called["count"] += 1
        return arr + 1

    monkeypatch.setattr(model, "preprocess_input", fake_preprocess)

    array = model.preprocess_image(sample_image_bytes, img_size=(6, 6), efficientnet=True)

    assert called["count"] == 1
    assert array.shape == (1, 6, 6, 3)
    assert array[0, 0, 0, 0] == 256


def test_preprocess_image_without_efficientnet_keeps_raw_pixels(sample_image_bytes):
    array = model.preprocess_image(sample_image_bytes, img_size=(6, 6), efficientnet=False)

    assert array.shape == (1, 6, 6, 3)
    assert array[0, 0, 0, 0] == 255.0


def test_leaf_gate_returns_false_below_threshold(sample_image_bytes, monkeypatch):
    class LeafModel:
        input_shape = (None, 4, 4, 3)

        def predict(self, x, verbose=0):
            return np.array([[0.2]], dtype=np.float32)

    state.leaf_classifier = LeafModel()
    state.leaf_classifier_input_size = (4, 4)
    monkeypatch.setattr(model, "CLASSIFIER_THRESHOLD", 0.5)

    is_leaf, score = model.leaf_gate(sample_image_bytes)

    assert is_leaf is False
    assert round(score, 3) == 0.2


def test_leaf_gate_returns_true_at_threshold(sample_image_bytes, monkeypatch):
    class LeafModel:
        input_shape = (None, 4, 4, 3)

        def predict(self, x, verbose=0):
            return np.array([[0.5]], dtype=np.float32)

    state.leaf_classifier = LeafModel()
    state.leaf_classifier_input_size = (4, 4)
    monkeypatch.setattr(model, "CLASSIFIER_THRESHOLD", 0.5)

    is_leaf, score = model.leaf_gate(sample_image_bytes)

    assert is_leaf is True
    assert score == 0.5


def test_leaf_gate_uses_default_size_when_input_size_missing(sample_image_bytes, monkeypatch):
    captured = {}

    class LeafModel:
        def predict(self, x, verbose=0):
            return np.array([[0.9]], dtype=np.float32)

    def fake_preprocess(data, img_size=None, *, efficientnet=True):
        captured["img_size"] = img_size
        return np.zeros((1, 224, 224, 3), dtype=np.float32)

    state.leaf_classifier = LeafModel()
    state.leaf_classifier_input_size = None
    monkeypatch.setattr(model, "preprocess_image", fake_preprocess)

    is_leaf, score = model.leaf_gate(sample_image_bytes)
    assert is_leaf is True
    assert round(score, 3) == 0.9
    assert captured["img_size"] == (224, 224)


def test_run_inference_caps_top_k_to_class_count(monkeypatch):
    class DiseaseModel:
        def predict(self, x, verbose=0):
            return np.array([[0.1, 0.7, 0.2]], dtype=np.float32)

    state.model = DiseaseModel()
    state.idx_to_label = {0: "healthy", 1: "rust", 2: "blight"}
    monkeypatch.setattr(model, "TOP_K", 10)

    results = model.run_inference(np.zeros((1, 3, 3, 3), dtype=np.float32))

    assert len(results[0]) == 3


def test_run_inference_falls_back_to_generated_label(monkeypatch):
    class DiseaseModel:
        def predict(self, x, verbose=0):
            return np.array([[0.1, 0.7]], dtype=np.float32)

    state.model = DiseaseModel()
    state.idx_to_label = {0: "healthy"}
    monkeypatch.setattr(model, "TOP_K", 2)

    results = model.run_inference(np.zeros((1, 3, 3, 3), dtype=np.float32))

    assert results[0][0].label == "class_1"
    assert results[0][1].label == "healthy"


def test_run_inference_sorts_predictions_descending(monkeypatch):
    class DiseaseModel:
        def predict(self, x, verbose=0):
            return np.array([[0.25, 0.75, 0.5]], dtype=np.float32)

    state.model = DiseaseModel()
    state.idx_to_label = {0: "a", 1: "b", 2: "c"}
    monkeypatch.setattr(model, "TOP_K", 3)

    results = model.run_inference(np.zeros((1, 3, 3, 3), dtype=np.float32))

    assert [item.label for item in results[0]] == ["b", "c", "a"]
