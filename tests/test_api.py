from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "student_dropout_model"
    assert data["model_alias"] == "champion"
    assert data["version"].startswith("version:")
