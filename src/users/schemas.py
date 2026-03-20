from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

# ---------- User Schemas ----------
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime

# ---------- Document Schemas ----------
class DocumentBase(BaseModel):
    title: Optional[str] = None   # Judul opsional, jika tidak diisi akan pakai nama file

class Document(DocumentBase):
    id: int
    user_id: int
    filename: str
    file_path: str
    upload_date: datetime
    status: str                     # processing, completed, error
    metadata: Optional[Dict[str, Any]] = None   # Info tambahan (jumlah halaman, dll)

    class Config:
        from_attributes = True

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Any, Dict
from datetime import datetime

# ---------- Chat Session Schemas ----------
class ChatSessionBase(BaseModel):
    document_id: int

class ChatSessionCreate(ChatSessionBase):
    pass

class ChatSession(ChatSessionBase):
    id: int
    user_id: int
    started_at: datetime
    ended_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---------- Chat Message Schemas ----------
class ChatMessageBase(BaseModel):
    role: str                       # 'user' atau 'assistant'
    content: str

class ChatMessageCreate(ChatMessageBase):
    session_id: int

class ChatMessage(ChatMessageBase):
    id: int
    session_id: int
    timestamp: datetime

    class Config:
        from_attributes = True


# ---------- Query Request / Response ----------
class QueryRequest(BaseModel):
    document_id: int
    question: str = Field(..., min_length=1, description="Pertanyaan user")

class SourceNode(BaseModel):
    text: str
    score: Optional[float] = None   # Skor similarity dari retrieval

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceNode]] = None   # Chunk sumber jawaban
