from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database import Base, User
from src.auth_service import create_user, authenticate_user, get_user_by_email


def make_test_db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    return engine, TestingSession()


def test_create_user_stores_hashed_password_and_candidate_role():
    engine, db = make_test_db()
    try:
        user = create_user(
            db, " Test ", " Candidate ", "TEST@example.com", "TestPassword123!"
        )

        assert user.email == "test@example.com"
        assert user.first_name == "Test"
        assert user.last_name == "Candidate"
        assert user.role == "candidate"
        assert user.password_hash != "TestPassword123!"
    finally:
        db.close()
        engine.dispose()


def test_authenticate_user_accepts_correct_password():
    engine, db = make_test_db()
    try:
        create_user(db, "Test", "User", "test@example.com", "TestPassword123!")

        user = authenticate_user(db, " TEST@example.com ", "TestPassword123!")

        assert user is not None
        assert user.email == "test@example.com"
    finally:
        db.close()
        engine.dispose()


def test_authenticate_user_rejects_wrong_password():
    engine, db = make_test_db()
    try:
        create_user(db, "Test", "User", "test@example.com", "TestPassword123!")

        user = authenticate_user(db, "test@example.com", "WrongPassword123!")

        assert user is None
    finally:
        db.close()
        engine.dispose()


def test_duplicate_email_is_rejected():
    engine, db = make_test_db()
    try:
        create_user(db, "Test", "User", "test@example.com", "TestPassword123!")

        try:
            create_user(
                db, "Another", "User", " TEST@example.com ", "AnotherPassword123!"
            )
            assert False, "Expected duplicate email to be rejected"
        except ValueError:
            pass

        assert get_user_by_email(db, "test@example.com") is not None
    finally:
        db.close()
        engine.dispose()
