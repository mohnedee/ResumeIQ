from src.auth import hash_password, verify_password


def test_password_is_hashed():
    password = "TestPassword123!"
    password_hash = hash_password(password)

    assert password_hash != password


def test_correct_password_is_verified():
    password = "TestPassword123!"
    password_hash = hash_password(password)

    assert verify_password(password, password_hash) is True


def test_wrong_password_is_rejected():
    password_hash = hash_password("TestPassword123!")

    assert verify_password("WrongPassword123!", password_hash) is False
