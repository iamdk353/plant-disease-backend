import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.db_models import Upload as DbUpload, User
from app.routes import activities, upload, user
from app.schemas import RegisterUserRequest


class FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return self

    def all(self):
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
            if getattr(item, "id", None) is None:
                item.id = uuid.uuid4()

    async def refresh(self, item):
        return None


class FakeUploadFile:
    def __init__(self, filename="leaf.jpg", content_type="image/jpeg", data=b"bytes"):
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self):
        return self._data


def test_register_user_creates_new_user():
    db = FakeDB([None])

    response = asyncio.run(
        user.register_user(RegisterUserRequest(firebase_uid="firebase-1", email="a@b.com"), db=db)
    )

    assert response.firebase_uid == "firebase-1"
    assert response.email == "a@b.com"
    assert any(isinstance(item, User) for item in db.added)


def test_register_user_returns_existing_user():
    existing = SimpleNamespace(id=uuid.uuid4(), firebase_uid="firebase-1", email="old@b.com")
    db = FakeDB([existing])

    response = asyncio.run(
        user.register_user(RegisterUserRequest(firebase_uid="firebase-1"), db=db)
    )

    assert response is existing


def test_upload_image_success(monkeypatch):
    uid = "firebase-1"
    fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(upload.uuid, "uuid4", lambda: fixed_uuid)
    monkeypatch.setattr(upload, "OCI_BUCKET", "bucket-1")

    client = SimpleNamespace(put_object=lambda *args, **kwargs: None)
    monkeypatch.setattr(upload, "get_oci_client", lambda: (client, "namespace"))
    db = FakeDB([SimpleNamespace(id=uuid.uuid4(), firebase_uid=uid)])

    response = asyncio.run(
        upload.upload_image(uid=uid, file=FakeUploadFile(), db=db)
    )

    assert response.bucket == "bucket-1"
    assert response.object_name == f"{uid}_{fixed_uuid}.jpg"
    assert any(isinstance(item, DbUpload) for item in db.added)


def test_upload_image_rejects_bad_content_type():
    db = FakeDB([SimpleNamespace(id=uuid.uuid4(), firebase_uid="firebase-1")])

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            upload.upload_image(
                uid="firebase-1",
                file=FakeUploadFile(content_type="text/plain"),
                db=db,
            )
        )

    assert exc.value.status_code == 415


def test_upload_image_rejects_missing_user():
    db = FakeDB([None])

    with pytest.raises(HTTPException) as exc:
        asyncio.run(upload.upload_image(uid="missing", file=FakeUploadFile(), db=db))

    assert exc.value.status_code == 404


def test_get_user_uploads_returns_object_names(monkeypatch):
    response_data = SimpleNamespace(objects=[SimpleNamespace(name="firebase-1_a.jpg"), SimpleNamespace(name="firebase-1_b.jpg")])
    client = SimpleNamespace(list_objects=lambda **kwargs: SimpleNamespace(data=response_data))
    monkeypatch.setattr(upload, "get_oci_client", lambda: (client, "namespace"))

    response = asyncio.run(upload.get_user_uploads("firebase-1"))

    assert response.uid == "firebase-1"
    assert response.images == ["firebase-1_a.jpg", "firebase-1_b.jpg"]


def test_get_activity_image_returns_response(monkeypatch):
    monkeypatch.setattr(activities, "fetch_from_oci", lambda object_name: b"image-bytes")

    response = asyncio.run(activities.get_activity_image("leaf.jpg"))

    assert response.body == b"image-bytes"
    assert response.media_type == "image/jpeg"


def test_get_user_activities_formats_results():
    user_row = SimpleNamespace(id=uuid.uuid4(), firebase_uid="firebase-1")
    upload_row = SimpleNamespace(object_name="leaf.jpg")
    prediction = SimpleNamespace(
        id=uuid.uuid4(),
        upload=upload_row,
        inference_ms=12.345,
        created_at=None,
        prediction_results=[
            SimpleNamespace(rank=2, label="rust", confidence=0.3),
            SimpleNamespace(rank=1, label="healthy", confidence=0.7),
        ],
    )
    db = FakeDB([user_row, [prediction]])
    request = SimpleNamespace(url_for=lambda *args, **kwargs: "http://test/leaf.jpg")

    response = asyncio.run(activities.get_user_activities(request=request, uid="firebase-1", db=db))

    assert len(response) == 1
    assert response[0].image_name == "leaf.jpg"
    assert response[0].results[0].label == "healthy"
    assert response[0].image_url == "http://test/leaf.jpg"
