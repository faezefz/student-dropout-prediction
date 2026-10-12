from fastapi.testclient import TestClient

from api.main import app
from api.schemas import StudentData

client = TestClient(app)

EXAMPLE_STUDENT = StudentData.model_config["json_schema_extra"]["examples"][0]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_student():
    response = client.post("/predict", json=EXAMPLE_STUDENT)
    assert response.status_code == 200

    result = response.json()
    assert 0 <= result["dropout_probability"] <= 1
    assert isinstance(result["at_risk"], bool)


def test_predict_invalid_gender():
    bad_student = EXAMPLE_STUDENT.copy()
    bad_student["gender"] = 5
    response = client.post("/predict", json=bad_student)
    assert response.status_code == 422