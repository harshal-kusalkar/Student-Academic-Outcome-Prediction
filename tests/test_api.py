from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def get_sample_payload():
    return {
        "marital_status": 1,
        "application_mode": 44,
        "application_order": 1,
        "course": 9003,
        "daytime_evening_attendance": 1,
        "previous_qualification": 39,
        "previous_qualification_grade": 160.0,
        "nationality": 1,
        "mothers_qualification": 3,
        "fathers_qualification": 3,
        "mothers_occupation": 2,
        "fathers_occupation": 10,
        "admission_grade": 160.0,
        "displaced": 1,
        "educational_special_needs": 0,
        "debtor": 0,
        "tuition_fees_up_to_date": 1,
        "gender": 1,
        "scholarship_holder": 0,
        "age_at_enrollment": 20,
        "international": 0,

        "curricular_units_1st_sem_credited": 0,
        "curricular_units_1st_sem_enrolled": 6,
        "curricular_units_1st_sem_evaluations": 7,
        "curricular_units_1st_sem_approved": 6,
        "curricular_units_1st_sem_grade": 14.0,
        "curricular_units_1st_sem_without_evaluations": 0,

        "curricular_units_2nd_sem_credited": 0,
        "curricular_units_2nd_sem_enrolled": 6,
        "curricular_units_2nd_sem_evaluations": 7,
        "curricular_units_2nd_sem_approved": 6,
        "curricular_units_2nd_sem_grade": 14.6667,
        "curricular_units_2nd_sem_without_evaluations": 0,

        "unemployment_rate": 12.4,
        "inflation_rate": 0.5,
        "gdp": 1.79,
    }


def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "student_dropout_model"
    assert data["alias"] == "champion"


def test_prediction():

    payload = get_sample_payload()

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert data["prediction"] in {
        "Dropout",
        "Enrolled",
        "Graduate",
    }


def test_missing_required_field():

    payload = get_sample_payload()

    del payload["age_at_enrollment"]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_field_type():

    payload = get_sample_payload()

    payload["age_at_enrollment"] = "invalid"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_extra_field():

    payload = get_sample_payload()

    payload["unknown_feature"] = 123

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422