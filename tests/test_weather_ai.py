import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.routes import ai, weather
from app.schemas import AiReportRequest, CropPlanRequest, LocationItem


class FakeWeatherResponse:
    def __init__(self, status_code=200, payload=None, text="error"):
        self.status_code = status_code
        self._payload = payload or {"temp": 28}
        self.text = text

    def json(self):
        return self._payload


class FakeWeatherClient:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None):
        return self.response


class FakeDB:
    def __init__(self):
        self.committed = False

    async def execute(self, stmt):
        raise AssertionError("execute should not be called in these tests")

    def add(self, item):
        return None

    async def flush(self):
        return None

    async def commit(self):
        self.committed = True


def test_get_weather_requires_api_key(monkeypatch):
    monkeypatch.setattr(weather, "WEATHER_API_KEY", None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(weather.get_weather(lat=12.0, lon=77.0))

    assert exc.value.status_code == 500


def test_get_weather_returns_json(monkeypatch):
    monkeypatch.setattr(weather, "WEATHER_API_KEY", "key")
    monkeypatch.setattr(weather.httpx, "AsyncClient", lambda: FakeWeatherClient(FakeWeatherResponse()))

    response = asyncio.run(weather.get_weather(lat=12.0, lon=77.0))

    assert response["temp"] == 28


def test_get_weather_propagates_api_error(monkeypatch):
    monkeypatch.setattr(weather, "WEATHER_API_KEY", "key")
    monkeypatch.setattr(
        weather.httpx,
        "AsyncClient",
        lambda: FakeWeatherClient(FakeWeatherResponse(status_code=503, text="down")),
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(weather.get_weather(lat=12.0, lon=77.0))

    assert exc.value.status_code == 503


def test_generate_ai_report_wraps_value_error(monkeypatch):
    async def boom(**kwargs):
        raise ValueError("missing prediction")

    monkeypatch.setattr(ai, "orchestrate_ai_report", boom)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(ai.generate_ai_report(body=AiReportRequest(prediction_id=uuid.uuid4()), db=FakeDB()))

    assert exc.value.status_code == 404


def test_get_ai_report_returns_orchestrator_result(monkeypatch):
    expected = {"prediction_id": uuid.uuid4(), "report": {"is_diseased": False, "report_text": "ok"}}

    async def ok(**kwargs):
        return expected

    monkeypatch.setattr(ai, "orchestrate_ai_report", ok)

    response = asyncio.run(ai.get_ai_report(prediction_id=uuid.uuid4(), db=FakeDB()))

    assert response == expected


def test_generate_crop_plan_returns_result(monkeypatch):
    expected = {"deduced_soil_type": "loamy", "recommended_crops": []}

    async def ok(**kwargs):
        return expected

    monkeypatch.setattr(ai, "orchestrate_crop_plan", ok)

    response = asyncio.run(
        ai.generate_crop_plan(body=CropPlanRequest(uid="firebase-1", location=LocationItem(lat=12.0, lon=77.0)), db=FakeDB())
    )

    assert response == expected
