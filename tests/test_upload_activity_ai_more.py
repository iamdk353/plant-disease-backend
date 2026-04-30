import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.routes import activities, ai, upload
from app.schemas import AiReportRequest, CropPlanRequest, LocationItem


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


def test_upload_rejects_empty_uid():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(upload.upload_image(uid="  ", file=FakeUploadFile(), db=FakeDB([])))

    assert exc.value.status_code == 400


def test_upload_rejects_empty_file_content(monkeypatch):
    monkeypatch.setattr(upload, "get_oci_client", lambda: (SimpleNamespace(put_object=lambda *args, **kwargs: None), "ns"))
    db = FakeDB([SimpleNamespace(id=uuid.uuid4(), firebase_uid="firebase-1")])

    with pytest.raises(HTTPException) as exc:
        asyncio.run(upload.upload_image(uid="firebase-1", file=FakeUploadFile(data=b""), db=db))

    assert exc.value.status_code == 400


def test_upload_without_filename_uses_uuid_only(monkeypatch):
    fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(upload.uuid, "uuid4", lambda: fixed_uuid)
    monkeypatch.setattr(upload, "OCI_BUCKET", "bucket-1")
    monkeypatch.setattr(upload, "get_oci_client", lambda: (SimpleNamespace(put_object=lambda *args, **kwargs: None), "ns"))
    db = FakeDB([SimpleNamespace(id=uuid.uuid4(), firebase_uid="firebase-1")])

    response = asyncio.run(upload.upload_image(uid="firebase-1", file=FakeUploadFile(filename=None), db=db))

    assert response.object_name == f"firebase-1_{fixed_uuid}"


def test_get_user_uploads_rejects_empty_uid():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(upload.get_user_uploads(" "))

    assert exc.value.status_code == 400


def test_get_user_uploads_returns_empty_list(monkeypatch):
    client = SimpleNamespace(list_objects=lambda **kwargs: SimpleNamespace(data=SimpleNamespace(objects=[])))
    monkeypatch.setattr(upload, "get_oci_client", lambda: (client, "ns"))

    response = asyncio.run(upload.get_user_uploads("firebase-1"))

    assert response.images == []


def test_get_activity_image_returns_404_on_generic_error(monkeypatch):
    monkeypatch.setattr(activities, "fetch_from_oci", lambda object_name: (_ for _ in ()).throw(Exception("boom")))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(activities.get_activity_image("leaf.jpg"))

    assert exc.value.status_code == 404


def test_get_user_activities_rejects_missing_user():
    db = FakeDB([None])

    with pytest.raises(HTTPException) as exc:
        asyncio.run(activities.get_user_activities(request=SimpleNamespace(url_for=lambda *a, **k: "url"), uid="missing", db=db))

    assert exc.value.status_code == 404


def test_get_user_activities_uses_unknown_image_when_upload_missing():
    user_row = SimpleNamespace(id=uuid.uuid4(), firebase_uid="firebase-1")
    prediction = SimpleNamespace(
        id=uuid.uuid4(),
        upload=None,
        inference_ms=None,
        created_at=None,
        prediction_results=[],
    )
    db = FakeDB([user_row, [prediction]])
    request = SimpleNamespace(url_for=lambda *args, **kwargs: "http://test/unknown")

    response = asyncio.run(activities.get_user_activities(request=request, uid="firebase-1", db=db))

    assert response[0].image_name == "unknown_image"


def test_get_ai_report_wraps_generic_exception(monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(ai, "orchestrate_ai_report", boom)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(ai.get_ai_report(prediction_id=uuid.uuid4(), db=FakeDB([])))

    assert exc.value.status_code == 500


def test_generate_ai_report_wraps_generic_exception(monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(ai, "orchestrate_ai_report", boom)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(ai.generate_ai_report(body=AiReportRequest(prediction_id=uuid.uuid4()), db=FakeDB([])))

    assert exc.value.status_code == 500


def test_generate_crop_plan_wraps_generic_exception(monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(ai, "orchestrate_crop_plan", boom)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            ai.generate_crop_plan(
                body=CropPlanRequest(uid="firebase-1", location=LocationItem(lat=12.0, lon=77.0)),
                db=FakeDB([]),
            )
        )

    assert exc.value.status_code == 500
