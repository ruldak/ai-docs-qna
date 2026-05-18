"""
SQLAlchemy database models.
Defines User, Document, ChatSession, ChatMessage, and ChatEvaluation.
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Text, 
    ForeignKey, JSON, Index, Float, func
)
from sqlalchemy.orm import relationship
from src.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(String(50), default="user", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="owner", cascade="all, delete-orphan")
    messages = relationship("ChatMessage", back_populates="owner")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    title = Column(String(500), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="PENDING", nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    file_path = Column(String(1000), nullable=True)
    indexed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    owner = relationship("User", back_populates="documents")

    __table_args__ = (
        Index("idx_document_user_status", "user_id", "status"),
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    title = Column(String(500), default="untitled")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    owner = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    role = Column(String(50), nullable=False)  # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    owner = relationship("User", back_populates="messages")
    evaluation = relationship("ChatEvaluation", back_populates="message", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_session_created", "session_id", "created_at"),
    )


class ChatEvaluation(Base):
    __tablename__ = "chat_evaluations"

    id = Column(Integer, primary_key=True)
    message_id = Column(
        Integer, 
        ForeignKey("chat_messages.id", ondelete="CASCADE"), 
        unique=True, 
        index=True, 
        nullable=False
    )

    faithfulness_score = Column(Float, nullable=True)
    answer_relevancy_score = Column(Float, nullable=True)

    faithfulness_reasoning = Column(Text, nullable=True)
    relevancy_reasoning = Column(Text, nullable=True)

    contexts = Column(JSON, nullable=True)
    raw_eval_response = Column(JSON, nullable=True)
    status = Column(String(50), default="PENDING", nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    message = relationship("ChatMessage", back_populates="evaluation")

    __table_args__ = (
        Index("idx_eval_status_created", "status", "created_at"),
    )