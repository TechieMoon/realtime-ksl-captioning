from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_healthz_and_readyz_with_mock_model() -> None:
    app = create_app(Settings(model_backend="mock"))

    with TestClient(app) as client:
        health = client.get("/healthz")
        ready = client.get("/readyz")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["model_backend"] == "mock"

