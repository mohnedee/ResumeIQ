from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "ResumeIQ API is running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_rejects_short_resume():
    response = client.post(
        "/predict",
        json={"resume_text": "Too short"},
    )

    assert response.status_code == 422


def test_predict_accepts_resume_text():
    response = client.post(
        "/predict",
        json={
            "resume_text": (
                "Computer engineering graduate with experience in Python, "
                "SQL, software development, data analysis, and machine learning."
            )
        },
    )

    assert response.status_code == 200
    result = response.json()

    assert "predicted_category" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 100


def test_my_resumes_requires_login():
    response = client.get("/my-resumes")

    assert response.status_code == 401


def test_hr_resumes_requires_login():
    response = client.get("/hr/resumes")

    assert response.status_code == 401


# Role-based access control tests using a temporary database
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src import api as api_module
from src import auth_routes as auth_routes_module
from src.auth import hash_password
from src.database import Base, User


@pytest.fixture
def role_test_setup():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine,
    )
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    api_module.app.dependency_overrides[api_module.get_db] = override_get_db
    api_module.app.dependency_overrides[
        auth_routes_module.get_db
    ] = override_get_db

    db = TestSessionLocal()
    db.add_all([
        User(
            first_name="Test",
            last_name="Candidate",
            email="candidate_test@example.com",
            password_hash=hash_password("CandidateTest123!"),
            role="candidate",
        ),
        User(
            first_name="Test",
            last_name="HR",
            email="hr_test@example.com",
            password_hash=hash_password("HrTestPassword123!"),
            role="hr",
        ),
    ])
    db.commit()
    db.close()

    try:
        with TestClient(api_module.app) as test_client:
            yield test_client
    finally:
        api_module.app.dependency_overrides.clear()
        test_engine.dispose()


def test_candidate_cannot_access_hr_resumes(role_test_setup):
    response = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert response.status_code == 200

    response = role_test_setup.get("/hr/resumes")
    assert response.status_code == 403


def test_hr_cannot_access_candidate_resumes(role_test_setup):
    response = role_test_setup.post(
        "/auth/login",
        json={
            "email": "hr_test@example.com",
            "password": "HrTestPassword123!",
        },
    )
    assert response.status_code == 200

    response = role_test_setup.get("/my-resumes")
    assert response.status_code == 403


def test_predict_pdf_rejects_wrong_content_type(role_test_setup):
    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert login.status_code == 200

    response = role_test_setup.post(
        "/predict-pdf",
        files={"file": ("resume.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_pdf_rejects_oversized_file(role_test_setup):
    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert login.status_code == 200

    content = b"%PDF-" + b"A" * (5 * 1024 * 1024)

    response = role_test_setup.post(
        "/predict-pdf",
        files={"file": ("large.pdf", content, "application/pdf")},
    )
    assert response.status_code == 413


def test_predict_pdf_rejects_invalid_signature(role_test_setup):
    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert login.status_code == 200

    response = role_test_setup.post(
        "/predict-pdf",
        files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
    )
    assert response.status_code == 400


def test_predict_pdf_accepts_valid_resume(role_test_setup):
    from pathlib import Path

    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert login.status_code == 200

    pdf_content = Path("uploads/sample_resume.pdf").read_bytes()

    response = role_test_setup.post(
        "/predict-pdf",
        files={
            "file": (
                "sample_resume.pdf",
                pdf_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_category"]
    assert 0 <= data["confidence"] <= 100
    assert data["message"] == "PDF resume classified and saved successfully"
    assert data["resume_id"]


def test_hr_download_requires_login():
    response = client.get("/hr/resumes/1/download")
    assert response.status_code == 401


def test_candidate_cannot_download_hr_resume(role_test_setup):
    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert login.status_code == 200

    response = role_test_setup.get("/hr/resumes/1/download")
    assert response.status_code == 403


def test_hr_download_returns_404_for_missing_resume(role_test_setup):
    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "hr_test@example.com",
            "password": "HrTestPassword123!",
        },
    )
    assert login.status_code == 200

    response = role_test_setup.get("/hr/resumes/99999/download")
    assert response.status_code == 404


def test_hr_can_download_resume_pdf(role_test_setup, tmp_path):
    from src.database import SessionLocal, Resume, User
    from src import api as api_module

    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert login.status_code == 200

    from pathlib import Path
    pdf_content = Path("uploads/sample_resume.pdf").read_bytes()
    upload_response = role_test_setup.post(
        "/predict-pdf",
        files={"file": ("sample_resume.pdf", pdf_content, "application/pdf")},
    )
    assert upload_response.status_code == 200
    resume_id = upload_response.json()["resume_id"]

    login = role_test_setup.post(
        "/auth/logout",
    )
    assert login.status_code == 200

    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "hr_test@example.com",
            "password": "HrTestPassword123!",
        },
    )
    assert login.status_code == 200

    response = role_test_setup.get(
        f"/hr/resumes/{resume_id}/download"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


def test_update_profile_requires_login():
    response = client.put(
        "/auth/profile",
        json={"first_name": "Updated", "last_name": "Name"},
    )
    assert response.status_code == 401


def test_change_password_requires_login():
    response = client.post(
        "/auth/change-password",
        json={
            "current_password": "CandidateTest123!",
            "new_password": "NewCandidate123!",
        },
    )
    assert response.status_code == 401


def test_update_profile_successfully(role_test_setup):
    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert login.status_code == 200

    response = role_test_setup.put(
        "/auth/profile",
        json={"first_name": "Updated", "last_name": "Candidate"},
    )
    assert response.status_code == 200
    assert response.json()["user"]["first_name"] == "Updated"
    assert response.json()["user"]["last_name"] == "Candidate"


def test_change_password_rejects_wrong_current_password(role_test_setup):
    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert login.status_code == 200

    response = role_test_setup.post(
        "/auth/change-password",
        json={
            "current_password": "WrongPassword123!",
            "new_password": "NewCandidate123!",
        },
    )
    assert response.status_code == 400


def test_change_password_successfully(role_test_setup):
    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert login.status_code == 200

    response = role_test_setup.post(
        "/auth/change-password",
        json={
            "current_password": "CandidateTest123!",
            "new_password": "NewCandidate123!",
        },
    )
    assert response.status_code == 200

    old_login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert old_login.status_code == 401

    new_login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "NewCandidate123!",
        },
    )
    assert new_login.status_code == 200


def test_candidate_cannot_access_user_list(role_test_setup):
    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "candidate_test@example.com",
            "password": "CandidateTest123!",
        },
    )
    assert login.status_code == 200

    response = role_test_setup.get("/auth/users")
    assert response.status_code == 403


def test_hr_can_access_user_list(role_test_setup):
    login = role_test_setup.post(
        "/auth/login",
        json={
            "email": "hr_test@example.com",
            "password": "HrTestPassword123!",
        },
    )
    assert login.status_code == 200

    response = role_test_setup.get("/auth/users")
    assert response.status_code == 200

    data = response.json()
    assert len(data["users"]) == 2
    assert all("password_hash" not in user for user in data["users"])
