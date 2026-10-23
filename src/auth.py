from pwdlib import PasswordHash

# Use a secure password-hashing algorithm.
password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Return a secure hash of a user's password."""
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Check a password against its stored hash."""
    return password_hasher.verify(password, password_hash)
