import pytest
from starlette.testclient import TestClient

from app import app

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
ROW = [69.0, 1.0, 0.0, 160.0, 234.0, 1.0, 2.0, 131.0, 0.1, 1.0, 1.0, 1.0, 0.0]


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_predict_request(client):
    response = client.get("predict/",
                          json={"data": [ROW],
                                "feature_names": FEATURES})
    assert 200 == response.status_code
    assert len(response.json()) > 0


def test_predict_request_no_data(client):
    response = client.get("/predict/",
                          json={"data": [],
                                "feature_names": FEATURES})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]
