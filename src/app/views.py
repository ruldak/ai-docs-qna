import os
import io
import asyncio
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import fitz
import docx2txt
import chardet
from fastapi import APIRouter, Depends, HTTPException, Security, UploadFile, File, Form, status
from sqlalchemy import select, update, delete, desc, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi_jwt import JwtAuthorizationCredentials
from celery.result import AsyncResult
from llama_index.core.llms import ChatMessage

from src.database import get_db
from src.app import models, schemas, utils
from src.app.lancedb_manager import LanceDBDocumentManager
from src.tasks import celery_app, process_document, evaluate_chat_message
from src.constants import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# ============================================================================
# CONSTANTS
# ============================================================================
ALLOWED_CONTENT_TYPES = [
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "application/pdf",
    "text/markdown",
    "text/x-markdown",
]


# ============================================================================
# DEPENDENCIES
# ============================================================================
async def get_current_user(
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(utils.access_security)
) -> models.User:
    """Get current authenticated user."""
    user_id = credentials.subject.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    result = await db.execute(select(models.User).where(models.User.id == user_id))
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


# ============================================================================
# TEXT EXTRACTION HELPERS
# ============================================================================

def safe_decode(content: bytes) -> str:
    """Decode bytes to string with automatic encoding detection."""
    if not content:
        return ""

    if len(content) < 100:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("utf-8", errors="ignore")

    detected = chardet.detect(content)
    encoding = detected.get("encoding", "utf-8")

    try:
        return content.decode(encoding or "utf-8")
    except (UnicodeDecodeError, LookupError, TypeError):
        return content.decode("utf-8", errors="ignore")


def extract_text_from_file(content: bytes, content_type: str) -> str:
    """Extract text from various file types."""
    if not content:
        raise ValueError("File content is empty")

    if content_type == "application/pdf":
        doc = fitz.open(stream=content, filetype="pdf")
        texts = [page.get_text() for page in doc]
        return "\n".join(texts)

    elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_like = io.BytesIO(content)
        return docx2txt.process(file_like)

    elif content_type in ["text/markdown", "text/x-markdown", "text/plain"]:
        return safe_decode(content)

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def validate_file(file: UploadFile) -> str:
    """Validate uploaded file and return normalized content type."""
    content_type = file.content_type or ""
    filename = file.filename or ""

    if content_type not in ALLOWED_CONTENT_TYPES:
        if filename.lower().endswith(".md"):
            content_type = "text/markdown"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File type not supported. Allowed: PDF, DOCX, TXT, Markdown (.md)"
            )

    file_size = getattr(file, "size", None)
    if file_size is not None and file_size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum limit ({settings.max_file_size_mb}MB)"
        )

    upload_content_type = content_type
    if content_type in ["text/markdown", "text/x-markdown"]:
        upload_content_type = "text/plain"

    return upload_content_type


# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

@router.get("/auth/me", response_model=schemas.User, status_code=status.HTTP_200_OK)
async def get_current_user_endpoint(
    current_user: models.User = Depends(get_current_user)
):
    """Get current authenticated user information."""
    return current_user


@router.post("/auth/register", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
async def register(
    user: schemas.UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register new user."""
    try:
        result = await db.execute(select(models.User).where(models.User.email == user.email))
        if result.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        hashed_pw = utils.get_password_hash(user.password)
        user_instance = models.User(
            full_name=user.full_name,
            email=user.email,
            password=hashed_pw
        )

        db.add(user_instance)
        await db.commit()
        await db.refresh(user_instance)
        return user_instance

    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Register error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/auth/login", response_model=schemas.LoginResponse)
async def login(
    user: schemas.UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """Login user and return JWT tokens."""
    try:
        result = await db.execute(select(models.User).where(models.User.email == user.email))
        user_data = result.scalars().first()

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User with that email not found"
            )

        if not utils.verify_password(user.password, user_data.password):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Incorrect password"
            )

        access_token = utils.access_security.create_access_token(
            subject={"user_id": user_data.id}
        )
        refresh_token = utils.access_security.create_refresh_token(
            subject={"user_id": user_data.id}
        )

        return {"access_token": access_token, "refresh_token": refresh_token}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# ============================================================================
# DOCUMENT ENDPOINTS
# ============================================================================

@router.get("/documents", response_model=List[schemas.DocumentResponseList], status_code=status.HTTP_200_OK)
async def get_documents(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get all documents for current user."""
    try:
        result = await db.execute(
            select(models.Document).where(models.Document.user_id == current_user.id)
        )
        documents = result.scalars().all()
        return list(documents) if documents else []

    except Exception as e:
        logger.error(f"Get documents error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/documents/{document_id}", response_model=schemas.DocumentResponse, status_code=status.HTTP_200_OK)
async def get_document_by_id(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get single document by ID with a signed URL."""
    try:
        result = await db.execute(
            select(models.Document).where(
                models.Document.id == document_id,
                models.Document.user_id == current_user.id
            )
        )
        document = result.scalars().first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        response_data = {
            "id": document.id,
            "user_id": document.user_id,
            "title": document.title,
            "description": document.description,
            "status": document.status,
            "chunk_count": document.chunk_count,
            "indexed_at": document.indexed_at,
            "file_path": document.file_path,
            "signed_url": None
        }

        if document.file_path:
            try:
                signed_url_res = utils.supabase.storage.from_(settings.bucket_name).create_signed_url(
                    path=document.file_path,
                    expires_in=3600
                )
                response_data["signed_url"] = signed_url_res.get("signedUrl")
            except Exception as e:
                logger.warning(f"Failed to generate signed URL: {e}")

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put("/documents/{document_id}", status_code=status.HTTP_200_OK)
async def update_document(
    document_id: int,
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Update document metadata or upload a new file."""
    if not title and not description and not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one field must be provided"
        )

    try:
        result = await db.execute(
            select(models.Document).where(
                models.Document.id == document_id,
                models.Document.user_id == current_user.id
            )
        )
        document = result.scalars().first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        task = None
        new_values = {}

        if file:
            upload_content_type = validate_file(file)

            content = await file.read()
            if not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File is empty"
                )

            if len(content) > settings.max_file_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File size exceeds maximum limit ({settings.max_file_size_mb}MB)"
                )

            try:
                text = extract_text_from_file(content, file.content_type or upload_content_type)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to extract text: {str(e)}"
                )

            if document.file_path:
                try:
                    utils.supabase.storage.from_(settings.bucket_name).remove([document.file_path])
                except Exception as e:
                    logger.warning(f"Failed to delete old file: {e}")

            file_path = f"uploads/{document_id}/{file.filename}"

            try:
                res = utils.supabase.storage.from_(settings.bucket_name).upload(
                    path=file_path,
                    file=content,
                    file_options={"content-type": upload_content_type, "x-upsert": "true"}
                )
                document.file_path = res.path
            except Exception as e:
                await db.rollback()
                logger.error(f"Upload failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Upload failed: {str(e)}"
                )

            document.status = "PENDING"
            new_values["indexed_at"] = datetime.now(timezone.utc)

            task = process_document.delay(
                filename=file.filename,
                document_id=document.id,
                title=document.title,
                description=document.description,
                content_type=file.content_type or upload_content_type,
                text=text,
                is_update=True
            )

        if title:
            new_values["title"] = title
        if description:
            new_values["description"] = description

        if new_values:
            stmt = update(models.Document).where(models.Document.id == document_id).values(**new_values)
            await db.execute(stmt)

        await db.commit()

        response = {"message": "Successfully updated"}
        if task:
            response["task_id"] = task.id

        return response

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Update document error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def post_document(
    title: str = Form(...),
    description: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Upload new document and trigger processing."""
    try:
        result = await db.execute(select(models.Document).where(models.Document.title == title))
        if result.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Title already in use"
            )

        upload_content_type = validate_file(file)

        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )

        if len(content) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum limit ({settings.max_file_size_mb}MB)"
            )

        try:
            text = extract_text_from_file(content, file.content_type or upload_content_type)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to extract text: {str(e)}"
            )

        document = models.Document(
            title=title,
            description=description,
            chunk_count=0,
            user_id=current_user.id,
            status="PENDING"
        )
        db.add(document)
        await db.flush()
        await db.refresh(document)

        file_path = f"uploads/{document.id}/{file.filename}"

        try:
            res = utils.supabase.storage.from_(settings.bucket_name).upload(
                path=file_path,
                file=content,
                file_options={"content-type": upload_content_type, "x-upsert": "true"}
            )
            document.file_path = res.path
            await db.commit()
        except Exception as e:
            await db.rollback()
            logger.error(f"Upload failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Upload failed: {str(e)}"
            )

        task = process_document.delay(
            filename=file.filename,
            document_id=document.id,
            title=title,
            description=description,
            content_type=file.content_type or upload_content_type,
            text=text,
            is_update=False
        )

        return {
            "id": document.id,
            "title": document.title,
            "description": document.description,
            "chunk_count": document.chunk_count,
            "user_id": document.user_id,
            "task_id": task.id
        }

    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title already in use"
        )
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Post document error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Delete document from DB, vector store, and Supabase storage."""
    try:
        result = await db.execute(
            select(models.Document).where(
                models.Document.id == document_id,
                models.Document.user_id == current_user.id
            )
        )
        document = result.scalars().first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or unauthorized"
            )

        doc_manager = LanceDBDocumentManager()

        def delete_from_vector_stores():
            try:
                doc_manager.delete_document(doc_id=str(document_id))
            except Exception as e:
                logger.warning(f"Vector store deletion warning: {e}")

        await asyncio.to_thread(delete_from_vector_stores)

        if document.file_path:
            try:
                utils.supabase.storage.from_(settings.bucket_name).remove([document.file_path])
            except Exception as e:
                logger.warning(f"Supabase storage deletion warning: {e}")

        await db.delete(document)
        await db.commit()
        return None

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Delete document error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# ============================================================================
# SESSION & CHAT ENDPOINTS
# ============================================================================

@router.get("/sessions", response_model=List[schemas.ChatSessionResponse], status_code=status.HTTP_200_OK)
async def get_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get all chat sessions for the current user."""
    try:
        result = await db.execute(
            select(models.ChatSession).where(models.ChatSession.user_id == current_user.id)
        )
        sessions = result.scalars().all()
        return list(sessions)

    except Exception as e:
        logger.error(f"Get sessions error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/sessions", response_model=schemas.ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Create new chat session."""
    try:
        chat_session = models.ChatSession(user_id=current_user.id)
        db.add(chat_session)
        await db.commit()
        await db.refresh(chat_session)
        return chat_session

    except Exception as e:
        await db.rollback()
        logger.error(f"Create session error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/sessions/{chat_session_id}/query", response_model=schemas.QueryResponse)
async def session_query(
    chat_session_id: int,
    input: schemas.QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Query document with RAG. Save history and trigger evaluation."""
    try:
        query_engine = utils.get_query_engine()

        result = await db.execute(
            select(models.ChatSession).where(
                models.ChatSession.id == chat_session_id,
                models.ChatSession.user_id == current_user.id
            )
        )
        session = result.scalars().first()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )

        result = await db.execute(
            select(models.Document).where(models.Document.id == input.document_id)
        )
        document = result.scalars().first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        if document.status == "FAILED":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document indexing failed. Please re-upload the document."
            )
        elif document.status == "PENDING":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is pending indexing. Please wait and try again later."
            )

        user_message = models.ChatMessage(
            session_id=chat_session_id,
            user_id=current_user.id,
            role="user",
            content=input.message
        )
        db.add(user_message)
        await db.flush()

        result = await db.execute(
            select(models.ChatMessage)
            .where(models.ChatMessage.session_id == chat_session_id)
            .order_by(models.ChatMessage.created_at)
            .limit(20)
        )
        messages = result.scalars().all()

        history_text = "\n".join([
            f"{msg.role}: {msg.content}" for msg in messages
        ]) if messages else "No previous messages"

        system_prompt = utils.load_system_prompt(input.document_id)

        answer, contexts = await query_engine.query_with_sources(
            query=input.message,
            document_id=input.document_id
        )

        llm_messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=f"""Berikut riwayat percakapan sebelumnya:
{history_text}

Berikut informasi relevan dari dokumen:
{answer}

Pertanyaan user: {input.message}

Jawablah pertanyaan di atas secara detail, komprehensif, dan lengkap berdasarkan informasi dari dokumen. 
Jika informasi tidak cukup, jelaskan alasannya. Jangan berikan jawaban singkat.""")
        ]

        llm = query_engine.llm
        response = await llm.achat(llm_messages)
        final_answer = response.message.content

        assistant_message = models.ChatMessage(
            session_id=chat_session_id,
            user_id=current_user.id,
            role="assistant",
            content=final_answer
        )
        db.add(assistant_message)
        await db.flush()

        evaluation = models.ChatEvaluation(
            message_id=assistant_message.id,
            contexts=contexts,
            status="PENDING"
        )
        db.add(evaluation)
        await db.commit()

        evaluate_chat_message.delay(evaluation.id, question=input.message)

        return {"response": final_answer}

    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Session query error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/sessions/{session_id}/history", response_model=List[schemas.ChatMessageResponse])
async def get_session_history(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get chat history for a session."""
    try:
        result = await db.execute(
            select(models.ChatMessage).where(
                models.ChatMessage.session_id == session_id,
                models.ChatMessage.user_id == current_user.id
            )
            .order_by(models.ChatMessage.created_at)
        )
        messages = result.scalars().all()
        return list(messages)

    except Exception as e:
        logger.error(f"Get history error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# ============================================================================
# EVALUATION ENDPOINTS (Admin)
# ============================================================================

@router.get("/admin/evaluations", response_model=List[schemas.EvaluationListResponse])
async def get_evaluations_list(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get a list of all evaluations with IDs.
    Used to select an evaluation before viewing details.
    """
    try:
        result = await db.execute(
            select(
                models.ChatEvaluation.id,
                models.ChatEvaluation.message_id,
                models.ChatMessage.session_id,
                models.ChatEvaluation.faithfulness_score,
                models.ChatEvaluation.answer_relevancy_score,
                models.ChatEvaluation.status,
                models.ChatEvaluation.created_at,
                models.ChatEvaluation.updated_at
            )
            .join(models.ChatMessage, models.ChatMessage.id == models.ChatEvaluation.message_id)
            .order_by(desc(models.ChatEvaluation.created_at))
        )

        evaluations = result.all()

        return [
            {
                "id": row.id,
                "message_id": row.message_id,
                "session_id": row.session_id,
                "faithfulness_score": row.faithfulness_score,
                "answer_relevancy_score": row.answer_relevancy_score,
                "status": row.status,
                "created_at": row.created_at,
                "updated_at": row.updated_at
            }
            for row in evaluations
        ]

    except Exception as e:
        logger.error(f"Get evaluations list error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/admin/evaluations/stats", response_model=List[schemas.EvaluationStatsResponse])
async def get_evaluation_stats(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get evaluation statistics per session."""
    try:
        result = await db.execute(
            select(
                models.ChatMessage.session_id,
                func.avg(models.ChatEvaluation.faithfulness_score).label("avg_faithfulness"),
                func.avg(models.ChatEvaluation.answer_relevancy_score).label("avg_relevancy"),
                func.count(models.ChatEvaluation.id).label("total_evaluated")
            )
            .join(models.ChatEvaluation, models.ChatMessage.id == models.ChatEvaluation.message_id)
            .where(models.ChatEvaluation.status == "SUCCESS")
            .group_by(models.ChatMessage.session_id)
            .order_by(desc("avg_faithfulness"))
        )

        stats = result.all()

        return [
            {
                "session_id": row.session_id,
                "avg_faithfulness": round(row.avg_faithfulness, 4) if row.avg_faithfulness else None,
                "avg_relevancy": round(row.avg_relevancy, 4) if row.avg_relevancy else None,
                "total_evaluated": row.total_evaluated
            }
            for row in stats
        ]

    except Exception as e:
        logger.error(f"Get evaluation stats error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/admin/evaluations/{evaluation_id}", response_model=schemas.EvaluationDetailResponse)
async def get_evaluation_detail(
    evaluation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get detail of a single evaluation complete with question and answer.
    """
    try:
        result = await db.execute(
            select(
                models.ChatEvaluation.id,
                models.ChatEvaluation.message_id,
                models.ChatMessage.session_id,
                models.ChatMessage.content.label("answer"),
                models.ChatEvaluation.faithfulness_score,
                models.ChatEvaluation.answer_relevancy_score,
                models.ChatEvaluation.faithfulness_reasoning,
                models.ChatEvaluation.relevancy_reasoning,
                models.ChatEvaluation.contexts,
                models.ChatEvaluation.raw_eval_response,
                models.ChatEvaluation.status,
                models.ChatEvaluation.error_message,
                models.ChatEvaluation.created_at,
                models.ChatEvaluation.updated_at
            )
            .join(models.ChatMessage, models.ChatMessage.id == models.ChatEvaluation.message_id)
            .where(models.ChatEvaluation.id == evaluation_id)
        )

        row = result.first()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found"
            )

        return {
            "id": row.id,
            "message_id": row.message_id,
            "session_id": row.session_id,
            "answer": row.answer,
            "faithfulness_score": row.faithfulness_score,
            "answer_relevancy_score": row.answer_relevancy_score,
            "faithfulness_reasoning": row.faithfulness_reasoning,
            "relevancy_reasoning": row.relevancy_reasoning,
            "contexts": row.contexts,
            "status": row.status,
            "error_message": row.error_message,
            "created_at": row.created_at,
            "updated_at": row.updated_at
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get evaluation detail error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# ============================================================================
# TASK ENDPOINTS
# ============================================================================

@router.get("/tasks/{task_id}", response_model=schemas.TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get Celery task status."""
    try:
        result = AsyncResult(task_id, app=celery_app)

        response = {
            "task_id": task_id,
            "status": result.state,
            "ready": result.ready(),
        }

        if result.state == "SUCCESS":
            response["result"] = "Task completed"
        elif result.state == "FAILURE":
            response["error"] = "An error occurred while processing. Please try again."
        elif result.state == "PENDING":
            response["message"] = "Task is pending or not found"
        elif result.state == "STARTED":
            response["message"] = "Task is currently running"
        elif result.state == "RETRY":
            response["message"] = "Task is being retried"

        return response

    except Exception as e:
        logger.error(f"Get task status error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )