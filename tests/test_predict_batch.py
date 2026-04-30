import asyncio
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from app.routes import predict
from app.schemas import BatchPredictRequest, Prediction as PredictionSchema, PredictRequest
from app.state import state


class FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class FakeDB:
    def __init__(self, results):
        self._results = list(results)
        self.added = []

    async def execute(self, stmt):
        return FakeResult(self._results.pop(0))

    def add(self, item):
        self.added.append(item)

    async def flush(self):
        for item in self.added:
            if hasattr(item, "id") and item.id is None:
                item.id = uuid.uuid4()


def test_predict_rejects_non_leaf_image(monkeypatch, uuid_pair):
    user_id, upload_id = uuid_pair
    state.model = object()
    db = FakeDB([SimpleNamespace(id=user_id, firebase_uid="u"), SimpleNamespace(id=upload_id, object_name="x")])
    monkeypatch.setattr(predict, "fetch_from_oci", lambda name: b"bytes")
    monkeypatch.setattr(predict, "leaf_gate", lambda data: (False, 0.12))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(predict.predict(PredictRequest(uid="u", object_name="x"), db=db))

    assert exc.value.status_code == 422


def test_predict_rejects_missing_upload(monkeypatch):
    state.model = object()
    db = FakeDB([SimpleNamespace(id=uuid.uuid4(), firebase_uid="u"), None])
    monkeypatch.setattr(predict, "fetch_from_oci", lambda name: b"bytes")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(predict.predict(PredictRequest(uid="u", object_name="missing"), db=db))

    assert exc.value.status_code == 404


def test_predict_batch_requires_model():
    state.model = None

    with pytest.raises(HTTPException) as exc:
        asyncio.run(predict.predict_batch(BatchPredictRequest(object_names=["a"])))

    assert exc.value.status_code == 503


def test_predict_batch_rejects_empty_object_names():
    state.model = object()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(predict.predict_batch(BatchPredictRequest(object_names=[])))

    assert exc.value.status_code == 422


def test_predict_batch_rejects_too_many_objects():
    state.model = object()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(predict.predict_batch(BatchPredictRequest(object_names=[str(i) for i in range(17)])))

    assert exc.value.status_code == 422


def test_predict_batch_rejects_non_leaf_item(monkeypatch):
    state.model = object()
    monkeypatch.setattr(predict, "fetch_from_oci", lambda name: b"bytes")
    monkeypatch.setattr(predict, "leaf_gate", lambda data: (False, 0.2))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(predict.predict_batch(BatchPredictRequest(object_names=["a"])))

    assert exc.value.status_code == 422


def test_predict_batch_returns_predictions(monkeypatch):
    state.model = object()
    monkeypatch.setattr(predict, "fetch_from_oci", lambda name: b"bytes")
    monkeypatch.setattr(predict, "leaf_gate", lambda data: (True, 0.9))
    monkeypatch.setattr(predict, "preprocess_image", lambda data: np.zeros((1, 4, 4, 3), dtype=np.float32))
    monkeypatch.setattr(
        predict,
        "run_inference",
        lambda x: [
            [PredictionSchema(label="rust", confidence=0.9)],
            [PredictionSchema(label="healthy", confidence=0.8)],
        ],
    )

    response = asyncio.run(predict.predict_batch(BatchPredictRequest(object_names=["a", "b"])))

    assert [item.object_name for item in response] == ["a", "b"]
    assert response[0].top_predictions[0].label == "rust"
    assert response[1].top_predictions[0].label == "healthy"


def test_predict_batch_uses_same_batch_length(monkeypatch):
    state.model = object()
    monkeypatch.setattr(predict, "fetch_from_oci", lambda name: b"bytes")
    monkeypatch.setattr(predict, "leaf_gate", lambda data: None)
    monkeypatch.setattr(predict, "preprocess_image", lambda data: np.zeros((1, 4, 4, 3), dtype=np.float32))
    monkeypatch.setattr(
        predict,
        "run_inference",
        lambda x: [[PredictionSchema(label="rust", confidence=0.9)] for _ in range(x.shape[0])],
    )

    response = asyncio.run(predict.predict_batch(BatchPredictRequest(object_names=["a", "b", "c"])))

    assert len(response) == 3
    assert all(item.inference_ms >= 0 for item in response)


def test_predict_batch_keeps_name_order(monkeypatch):
    state.model = object()
    monkeypatch.setattr(predict, "fetch_from_oci", lambda name: b"bytes")
    monkeypatch.setattr(predict, "leaf_gate", lambda data: None)
    monkeypatch.setattr(predict, "preprocess_image", lambda data: np.zeros((1, 4, 4, 3), dtype=np.float32))
    monkeypatch.setattr(
        predict,
        "run_inference",
        lambda x: [[PredictionSchema(label=f"label-{i}", confidence=1.0)] for i in range(x.shape[0])],
    )

    response = asyncio.run(predict.predict_batch(BatchPredictRequest(object_names=["first", "second"])))

    assert [item.object_name for item in response] == ["first", "second"]
