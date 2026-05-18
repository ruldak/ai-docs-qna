"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================================
# AUTH SCHEMAS
# ============================================================================
class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=3, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str


# ============================================================================
# USER SCHEMAS
# ============================================================================
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=255)


class UserCreate(UserBase):
    password: str = Field(..., min_length=3, max_length=128)


class User(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str = "user"
    created_at: datetime
    is_active: bool


# ============================================================================
# DOCUMENT SCHEMAS
# ============================================================================
class DocumentBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)


class DocumentCreate(DocumentBase):
    description: Optional[str] = Field(None, max_length=5000)


class DocumentResponseList(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    description: Optional[str] = None
    status: str = "PENDING"
    chunk_count: int = 0
    indexed_at: datetime
    file_path: Optional[str] = None


class DocumentResponse(DocumentResponseList):
    signed_url: Optional[str] = None


class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=5000)


# ============================================================================
# QUERY SCHEMAS
# ============================================================================
class QueryRequest(BaseModel):
    document_id: int = Field(..., gt=0)
    message: str = Field(..., min_length=1, max_length=10000)


class QueryResponse(BaseModel):
    response: str


# ============================================================================
# CHAT SCHEMAS
# ============================================================================
class ChatSessionCreate(BaseModel):
    title: Optional[str] = Field("untitled", max_length=500)


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    title: str
    created_at: datetime


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    user_id: int
    role: str
    content: str
    created_at: datetime


# ============================================================================
# EVALUATION SCHEMAS
# ============================================================================
class EvaluationDetailResponse(BaseModel):
    id: int
    message_id: int
    session_id: int
    answer: Optional[str] = None
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    faithfulness_reasoning: Optional[str] = None
    relevancy_reasoning: Optional[str] = None
    contexts: Optional[List[str]] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class EvaluationListResponse(BaseModel):
    """List semua evaluations untuk dipilih (dengan ID)."""
    id: int
    message_id: int
    session_id: int
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    status: str
    created_at: datetime
    updated_at: datetime


class EvaluationStatsResponse(BaseModel):
    session_id: int
    avg_faithfulness: Optional[float] = None
    avg_relevancy: Optional[float] = None
    total_evaluated: int


# ============================================================================
# TASK SCHEMAS
# ============================================================================
class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    ready: bool
    result: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None