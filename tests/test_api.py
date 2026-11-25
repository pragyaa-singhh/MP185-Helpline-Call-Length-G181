from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def valid_payload():
    return {
        "queue_wait_sec": 120,
        "agent_exp_months": 24,
        "previous_calls": 3,
        "issue_count": 2,
        "hour": 14,
        "caller_age": 35,
        "language": "english",
        "team": "Beta"
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction():
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_previous_calls_validation():
    payload = valid_payload()
    payload["previous_calls"] = 21

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_queue_wait_validation():
    payload = valid_payload()
    payload["queue_wait_sec"] = -1

    response = client.post("/predict", json=payload)
    assert response.status_code == 422