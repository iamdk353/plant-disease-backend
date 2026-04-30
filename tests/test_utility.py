import pytest
from fastapi import HTTPException

from app.routes import utility
from app.state import state


def test_health_reports_loaded_model(monkeypatch):
    state.model = object()
    state.idx_to_label = {0: "healthy", 1: "rust"}
    state.load_time = 1.23456
    monkeypatch.setattr(utility.tf.config, "list_physical_devices", lambda kind=None: ["GPU"])

    response = utility.health()

    assert response.status == "ok"
    assert response.num_classes == 2
    assert response.model_load_time_s == 1.235
    assert response.gpu_available is True


def test_health_reports_loading_state(monkeypatch):
    state.model = None
    state.idx_to_label = None
    state.load_time = None
    monkeypatch.setattr(utility.tf.config, "list_physical_devices", lambda kind=None: [])

    response = utility.health()

    assert response.status == "loading"
    assert response.num_classes == 0
    assert response.model_load_time_s is None
    assert response.gpu_available is False


def test_list_classes_requires_loaded_model():
    state.idx_to_label = None

    with pytest.raises(HTTPException) as exc:
        utility.list_classes()

    assert exc.value.status_code == 503


def test_list_classes_returns_classes():
    state.idx_to_label = {0: "healthy", 1: "rust"}

    response = utility.list_classes()

    assert response["num_classes"] == 2
    assert response["classes"][1] == "rust"
