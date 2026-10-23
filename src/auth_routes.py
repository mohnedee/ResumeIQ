import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from src.auth_schemas import RegisterRequest, LoginRequest, UpdateProfileRequest, ChangePasswordRequest
from src.auth_service import create_user, authenticate_user
from src.database import SessionLocal, User
from src.auth import hash_password, verify_password

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register", status_code=201)
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    email = data.email.strip().lower()

    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid email address.",
        )

    try:
        user = create_user(
            db=db,
            first_name=data.first_name,
            last_name=data.last_name,
            email=email,
            password=data.password,
        )
    except ValueError:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists.",
        )

    logger.info("Account registered: user_id=%s, role=%s", user.id, user.role)

    return {
        "message": "Account created successfully.",
        "user": {
            "id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "role": user.role,
        },
    }


@router.post("/login")
def login(
    data: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    email = data.email.strip().lower()
    user = authenticate_user(db, email, data.password)

    if user is None:
        logger.warning("Failed login attempt.")
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password.",
        )

    request.session.clear()
    request.session["user_id"] = user.id

    logger.info(
        "Successful login: user_id=%s, role=%s",
        user.id,
        user.role,
    )

    return {
        "message": "Login successful.",
        "user": {
            "id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "role": user.role,
        },
    }


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    logger.info("User logged out.")
    return {"message": "Logged out successfully."}


@router.get("/me")
def current_user(
    request: Request,
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")

    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Please log in first.",
        )

    user = db.get(User, user_id)

    if user is None:
        request.session.clear()
        raise HTTPException(
            status_code=401,
            detail="User account not found.",
        )

    return {
        "id": user.id,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "role": user.role,
    }



@router.put("/profile")
def update_profile(
    data: UpdateProfileRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Please log in first.")

    user = db.get(User, user_id)
    if user is None:
        request.session.clear()
        raise HTTPException(status_code=401, detail="User account not found.")

    first_name = data.first_name.strip()
    last_name = data.last_name.strip()
    if not first_name or not last_name:
        raise HTTPException(
            status_code=400,
            detail="First name and last name cannot be blank.",
        )

    user.first_name = first_name
    user.last_name = last_name
    db.commit()
    db.refresh(user)

    logger.info("Profile updated: user_id=%s", user.id)
    return {
        "message": "Profile updated successfully.",
        "user": {
            "id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "role": user.role,
        },
    }


@router.post("/change-password")
def change_password(
    data: ChangePasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Please log in first.")

    user = db.get(User, user_id)
    if user is None:
        request.session.clear()
        raise HTTPException(status_code=401, detail="User account not found.")

    if not verify_password(data.current_password, user.password_hash):
        raise HTTPException(
            status_code=400,
            detail="Current password is incorrect.",
        )

    if verify_password(data.new_password, user.password_hash):
        raise HTTPException(
            status_code=400,
            detail="New password must be different from the current password.",
        )

    user.password_hash = hash_password(data.new_password)
    db.commit()

    logger.info("Password changed: user_id=%s", user.id)
    return {"message": "Password changed successfully."}


@router.get("/users")
def list_users_for_hr(
    request: Request,
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")

    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Please log in first.",
        )

    current_user = db.get(User, user_id)

    if current_user is None:
        request.session.clear()
        raise HTTPException(
            status_code=401,
            detail="User account not found.",
        )

    if current_user.role != "hr":
        raise HTTPException(
            status_code=403,
            detail="HR account required.",
        )

    users = db.query(User).order_by(User.created_at.desc()).all()

    return {
        "users": [
            {
                "id": user.id,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "role": user.role,
                "created_at": (
                    user.created_at.isoformat()
                    if user.created_at
                    else None
                ),
            }
            for user in users
        ]
    }
