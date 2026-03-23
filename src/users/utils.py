from pwdlib import PasswordHash
import os
from dotenv import load_dotenv

load_dotenv()

password_hash = PasswordHash.recommended() # Gunakan Argon2

def get_password_hash(password: str) -> str:
    """Hash password saat registrasi."""
    return password_hash.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Cocokkan password saat login."""
    return password_hash.verify(plain_password, hashed_password)


from fastapi_jwt import JwtAuthorizationCredentials, JwtAccessBearer, JwtRefreshBearer

secret_key = os.getenv("SECRET_KEY")
access_security = JwtAccessBearer(secret_key=secret_key, auto_error=True)
refresh_security = JwtRefreshBearer(secret_key=secret_key, auto_error=True)