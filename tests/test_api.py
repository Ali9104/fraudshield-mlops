from unittest.mock import patch, MagicMock

with patch("src.predict.load_model", return_value=MagicMock()):
    from api.main import app

from fastapi.testclient import TestClient

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "FraudShield API is running"
    }