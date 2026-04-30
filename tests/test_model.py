import numpy as np
import pytest
from fastapi import HTTPException

from app import model
from app.state import state


class FakeLeafModel:
    def __init__(self, score: float):
        self.score = score

    def predict(self, x, verbose=0):
        return np.array([[self.score]], dtype=np.float32)


class FakeDiseaseModel:
    def __init__(self, probs):
        self.probs = np.array(probs, dtype=np.float32)

    def predict(self, x, verbose=0):
        return self.probs


def test_preprocess_image_decodes_and_resizes(sample_image_bytes):
    array = model.preprocess_image(sample_image_bytes, img_size=(8, 8), efficientnet=False)

    assert array.shape == (1, 8, 8, 3)
    assert array.dtype == np.float32


def test_preprocess_image_rejects_invalid_bytes():
    with pytest.raises(HTTPException) as exc:
        model.preprocess_image(b"not-an-image")

    assert exc.value.status_code == 422


def test_leaf_gate_uses_threshold(sample_image_bytes, monkeypatch):
    state.leaf_classifier = FakeLeafModel(score=0.75)
    state.leaf_classifier_input_size = (4, 4)
    monkeypatch.setattr(model, "CLASSIFIER_THRESHOLD", 0.5)

    is_leaf, score = model.leaf_gate(sample_image_bytes)

    assert is_leaf is True
    assert score == 0.75


def test_leaf_gate_returns_none_without_classifier(sample_image_bytes):
    state.leaf_classifier = None

    assert model.leaf_gate(sample_image_bytes) is None


def test_run_inference_returns_top_k_labels(monkeypatch):
    state.model = FakeDiseaseModel([[0.1, 0.7, 0.2], [0.4, 0.3, 0.3]])
    state.idx_to_label = {0: "healthy", 1: "rust"}
    monkeypatch.setattr(model, "TOP_K", 2)

    results = model.run_inference(np.zeros((2, 3, 3, 3), dtype=np.float32))

    assert len(results) == 2
    assert [item.label for item in results[0]] == ["rust", "class_2"]
    assert results[0][0].confidence == 0.7
    assert results[1][0].label == "healthy"
