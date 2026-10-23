from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database import User
from src.auth import hash_password, verify_password


def get_user_by_email(db: Session, email: str):
    normalized_email = email.strip().lower()
    return db.scalar(
        select(User).where(User.email == normalized_email)
    )


def create_user(
    db: Session,
    first_name: str,
    last_name: str,
    email: str,
    password: str,
):
    normalized_email = email.strip().lower()

    if get_user_by_email(db, normalized_email):
        raise ValueError("An account with this email already exists.")

    user = User(
        first_name=first_name.strip(),
        last_name=last_name.strip(),
        email=normalized_email,
        password_hash=hash_password(password),
        role="candidate",
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)

    if user is None:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user
