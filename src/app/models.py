from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database import Base

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(String, default="user")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    title = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    status = Column(String, default="PENDING")
    chunk_count = Column(Integer, nullable=False)
    file_path = Column(String, nullable=True)
    indexed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    title = Column(String, default="untitled")
    created_at = Column(DateTime, server_default=func.now())

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    role = Column(String) # "user" | "assistant" | "system"
    content = Column(Text)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    __table_args__ = (
        Index("idx_session_created", "session_id", "created_at"),
    )
