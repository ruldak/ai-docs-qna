from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# ---------- User Schemas ----------
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    role: str
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime

# ---------- Document Schemas ----------
class DocumentBase(BaseModel):
    title: str

class Document(DocumentBase):
    id: int
    description: Optional[str] = None

    class Config:
        from_attributes = True

# ---------- Query Schemas ----------
class Query(BaseModel):
    document_id: int
    message: str