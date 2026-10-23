
from getpass import getpass
from src.database import SessionLocal, User
from src.auth import hash_password

db = SessionLocal()

try:
    first = input("HR first name: ").strip()
    last = input("HR last name: ").strip()
    email = input("HR email: ").strip().lower()
    password = getpass("HR password: ")

    if not first or not last or not email:
        print("Error: name and email are required.")
    elif len(password) < 8:
        print("Error: password must be at least 8 characters.")
    elif db.query(User).filter(User.email == email).first():
        print("Error: this email already exists.")
    else:
        user = User(
            first_name=first,
            last_name=last,
            email=email,
            password_hash=hash_password(password),
            role="hr",
        )
        db.add(user)
        db.commit()
        print("HR account created successfully.")

finally:
    db.close()
