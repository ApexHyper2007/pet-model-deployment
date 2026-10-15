import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client(monkeypatch):
    """Use a lightweight fake model so API tests focus on HTTP behavior."""
    monkeypatch.setattr(api, "load_model", lambda device: object())
    monkeypatch.setattr(
        api,
        "predict_image_bytes",
        lambda model, image_bytes, device: [
            ("Saint Bernard", 0.98),
            ("Boxer", 0.01),
        ],
    )

    with TestClient(api.app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_top_predictions(client):
    response = client.post(
        "/predict",
        files={"file": ("pet.jpg", b"image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "pet.jpg",
        "predicted_breed": "Saint Bernard",
        "confidence": 0.98,
        "top_predictions": [
            {"breed": "Saint Bernard", "confidence": 0.98},
            {"breed": "Boxer", "confidence": 0.01},
        ],
    }


def test_predict_rejects_non_image_upload(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Upload a valid image file."


def test_predict_rejects_empty_upload(client):
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."


def test_predict_returns_bad_request_for_corrupt_image(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "predict_image_bytes",
        lambda model, image_bytes, device: (_ for _ in ()).throw(
            ValueError("Could not read the uploaded file.")
        ),
    )

    response = client.post(
        "/predict",
        files={"file": ("broken.jpg", b"broken", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Could not read the uploaded file."
