import asyncio
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from app.db_models import Prediction as DbPrediction, PredictionResult
from app.routes import predict
from app.schemas import PredictRequest, Prediction as PredictionSchema
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


def test_predict_returns_saved_prediction(monkeypatch, uuid_pair):
    user_id, upload_id = uuid_pair
    state.model = object()
    user = SimpleNamespace(id=user_id, firebase_uid="firebase-1")
    upload = SimpleNamespace(id=upload_id, object_name="leaf.png")
    db = FakeDB([user, upload])

    monkeypatch.setattr(predict, "fetch_from_oci", lambda object_name: b"image-bytes")
    monkeypatch.setattr(predict, "leaf_gate", lambda data: (True, 0.98))
    monkeypatch.setattr(predict, "preprocess_image", lambda data: np.zeros((1, 4, 4, 3), dtype=np.float32))
    monkeypatch.setattr(
        predict,
        "run_inference",
        lambda x: [[
            PredictionSchema(label="rust", confidence=0.9),
            PredictionSchema(label="healthy", confidence=0.1),
        ]],
    )

    response = asyncio.run(
        predict.predict(PredictRequest(uid="firebase-1", object_name="leaf.png"), db=db)
    )

    assert response.object_name == "leaf.png"
    assert response.prediction_id is not None
    assert [item.label for item in response.top_predictions] == ["rust", "healthy"]
    assert any(isinstance(item, DbPrediction) for item in db.added)
    assert sum(isinstance(item, PredictionResult) for item in db.added) == 2


def test_predict_rejects_missing_model():
    state.model = None

    with pytest.raises(HTTPException) as exc:
        asyncio.run(predict.predict(PredictRequest(uid="firebase-1", object_name="leaf.png"), db=FakeDB([])))

    assert exc.value.status_code == 503


def test_predict_rejects_missing_user():
    state.model = object()
    db = FakeDB([None])

    with pytest.raises(HTTPException) as exc:
        asyncio.run(predict.predict(PredictRequest(uid="missing", object_name="leaf.png"), db=db))

    assert exc.value.status_code == 404
