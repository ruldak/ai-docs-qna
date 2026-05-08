from fastapi import APIRouter, Depends, HTTPException, Security, UploadFile, File, Form
from . import utils, service, models, schemas, constants
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_db
from sqlalchemy import select, update, delete, desc
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi_jwt import JwtAuthorizationCredentials
from sqlalchemy.exc import IntegrityError
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader, DocxReader
import os
from dotenv import load_dotenv
import fitz
from llama_index.core import Document
import docx2txt
import io
from typing import Optional, List
import asyncio
from datetime import datetime, timezone
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.storage.chat_store.postgres import PostgresChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from src.tasks import process_document, celery_task
from celery.result import AsyncResult
from .pinecone import PineconeDocumentManager
from pathlib import Path
from llama_index.core.llms import ChatMessage as LLMChatMessage
import chardet

load_dotenv()

router = APIRouter(prefix="/api")

# =====================================================================
# Konstanta tipe file yang didukung
# =====================================================================
ALLOWED_CONTENT_TYPES = [
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "application/pdf",
    "text/markdown",
    "text/x-markdown",
]

# --- AUTH ENDPOINTS ---

@router.get("/auth/me", response_model=schemas.User, status_code=200)
async def get_current_user(db: AsyncSession = Depends(get_db), credentials: JwtAuthorizationCredentials = Security(utils.access_security)):
    try:
        get_user = await db.execute(select(models.User).where(
            models.User.id == credentials.subject["user_id"]
        ))

        user_exist = get_user.scalars().first()

        if not user_exist:
            raise HTTPException(status_code=404, detail="user not found.")
        
        return user_exist
    except HTTPException:
        raise
    except Exception as e:
        print(f"user get error 500: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")

@router.post("/auth/register", response_model=schemas.User, status_code=201)
async def register(user: schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    try:
        get_user = await db.execute(select(models.User.email).where(
            models.User.email == user.email
        ))

        is_user_exist = get_user.scalars().first()

        if is_user_exist:
            raise HTTPException(status_code=400, detail="email already taken")

        hashed_pw = utils.get_password_hash(user.password)

        user_instance = models.User(full_name=user.full_name, email=user.email, password=hashed_pw)
        db.add(user_instance)
        await db.commit()
        await db.refresh(user_instance)

        return user_instance
    except IntegrityError as e:
        await db.rollback()

        print(f"register integrity error: {e}")

        if "unique" in str(e.orig).lower():
            raise HTTPException(status_code=400, detail="email already taken")
                
        raise HTTPException(status_code=400, detail="An error occurred in the data.")
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/auth/login", response_model=schemas.LoginResponse)
async def login(user: schemas.UserLogin, db: AsyncSession = Depends(get_db)):
    try:
        get_user = await db.execute(select(models.User).where(
            models.User.email == user.email
        ))

        user_data = get_user.scalars().first()

        if not user_data:
            raise HTTPException(status_code=404, detail="user with that email not found.")
    
        if utils.verify_password(user.password, user_data.password):
            access_token = utils.access_security.create_access_token(subject={"user_id": user_data.id})
            refresh_token = utils.access_security.create_refresh_token(subject={"user_id": user_data.id})
        else:
            raise HTTPException(status_code=403, detail="password incorrect")

        return {"access_token": access_token, "refresh_token": refresh_token}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


# --- HELPER: Ekstraksi teks dari file ---
def safe_decode(content: bytes) -> str:
    """
    Decode bytes ke string dengan deteksi encoding otomatis.
    File < 100 bytes langsung pakai utf-8 tanpa chardet.
    """
    # File kecil: langsung decode utf-8
    if len(content) < 100:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("utf-8", errors="ignore")

    # File besar: pakai chardet
    detected = chardet.detect(content)
    encoding = detected.get("encoding", "utf-8")
    confidence = detected.get("confidence", 0)

    print(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")

    try:
        return content.decode(encoding)
    except (UnicodeDecodeError, LookupError, TypeError):
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("utf-8", errors="ignore")


def extract_text_from_file(content: bytes, content_type: str) -> str:
    """
    Ekstrak teks dari berbagai tipe file.
    Mendukung: PDF, DOCX, TXT, Markdown
    """
    if content_type == "application/pdf":
        doc = fitz.open(stream=content, filetype="pdf")
        texts = []
        for page in doc:
            texts.append(page.get_text())
        return "\n".join(texts)
    elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_like = io.BytesIO(content)
        return docx2txt.process(file_like)
    elif content_type in ["text/markdown", "text/x-markdown", "text/plain"]:
        return safe_decode(content)

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


# --- DOCUMENT ENDPOINTS ---

@router.get("/documents", response_model=List[schemas.DocumentResponseList], status_code=200)
async def get_documents(db: AsyncSession = Depends(get_db), credentials: JwtAuthorizationCredentials = Security(utils.access_security)):
    try:
        get_user = await db.execute(select(models.User).where(
            models.User.id == credentials.subject["user_id"]
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        get_documents = await db.execute(select(models.Document).where(
            models.Document.user_id == credentials.subject["user_id"]
        ))

        documents = get_documents.scalars().all()

        if not documents:
            raise HTTPException(status_code=404, detail="No documents found.")

        return documents
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/documents/{document_id}", response_model=schemas.DocumentResponse, status_code=200)
async def get_document_by_id(
        document_id: int,
        db: AsyncSession = Depends(get_db),
        credentials: JwtAuthorizationCredentials = Security(utils.access_security)
    ):
    try:
        get_user = await db.execute(select(models.User).where(
            models.User.id == credentials.subject["user_id"]
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        get_document = await db.execute(select(models.Document).where(
            models.Document.id == document_id,
            models.Document.user_id == credentials.subject["user_id"]
        ))

        document_instance = get_document.scalars().first()

        if not document_instance:
            raise HTTPException(status_code=404, detail="Document not found.")

        response_data = document_instance.__dict__

        if document_instance.file_path:
            signed_url_res = utils.supabase.storage.from_(os.getenv("BUCKET_NAME")).create_signed_url(
                path=document_instance.file_path,
                expires_in=3600
            )

            response_data["signed_url"] = signed_url_res.get("signedUrl")
        
        return response_data
    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.put("/documents/{document_id}", status_code=200)
async def update_document(
        document_id: int,
        title: Optional[str] = Form(None),
        description: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        db: AsyncSession = Depends(get_db),
        credentials: JwtAuthorizationCredentials = Security(utils.access_security)
    ):
    try:
        get_user = await db.execute(select(models.User).where(
            models.User.id == credentials.subject["user_id"]
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        if not title and not description and not file:
            raise HTTPException(status_code=400, detail="One of the fields must be filled in.")

        new_values = {}

        task = None

        if file:
            if file.content_type not in ALLOWED_CONTENT_TYPES:
                filename = file.filename or ""
                if filename.lower().endswith(".md"):
                    file.content_type = "text/markdown"
                else:
                    raise HTTPException(status_code=400, detail="file type is not supported.")

            # ---------- make sure the file type is supported ----------
            if file.content_type not in ALLOWED_CONTENT_TYPES:
                # Fallback: cek ekstensi filename jika content_type tidak dikenali browser
                filename = file.filename or ""
                if filename.lower().endswith(".md"):
                    file.content_type = "text/markdown"
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="file type is not supported. Allowed: PDF, DOCX, TXT, Markdown (.md)"
                    )

            # ---------- make sure file size is under 5mb ----------
            if file.size > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="file size exceeds the specified maximum limit (5mb).")

            get_document = await db.execute(select(models.Document).where(
                models.Document.id == document_id,
                models.Document.user_id == credentials.subject["user_id"]
            ))

            document = get_document.scalars().first()

            if not document:
                raise HTTPException(status_code=404, detail="Document not found")

            if document.file_path:
                storage_response = utils.supabase.storage.from_(os.getenv("BUCKET_NAME")).remove([document.file_path])

            # ---------- Read and extract text ----------
            content = await file.read()
            text = extract_text_from_file(content, file.content_type)

            document.status = "PENDING"

            bucket_name = os.getenv("BUCKET_NAME")
            file_path = f"uploads/{document_id}/{file.filename}"

            upload_content_type = file.content_type
            if upload_content_type == "text/markdown" or upload_content_type == "text/x-markdown":
                upload_content_type = "text/plain"
            
            try:
                res = utils.supabase.storage.from_(bucket_name).upload(
                    path=file_path,
                    file=content,
                    file_options={"content-type": upload_content_type, "x-upsert": "true"}
                )
                uploaded_path = res.path
                document.file_path = uploaded_path
                await db.commit()
                
            except Exception as e:
                await db.rollback()
                raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

            task = process_document.delay(
                filename=file.filename,
                document_id=document.id,
                title=document.title,
                description=document.description,
                content_type=file.content_type,
                text=text,
                is_update=True
            )

            new_values["indexed_at"] = datetime.now(timezone.utc)

        if title:
            new_values["title"] = title

        if description:
            new_values["description"] = description

        if new_values:
            stmt = update(models.Document).where(models.Document.id == document_id).values(**new_values)
            await db.execute(stmt)
            await db.commit()

        response_data = {"message": "Successfully updated."}
        
        if task:
            response_data["task_id"] = task.id
        
        return response_data
    except HTTPException as e:
        await db.rollback()
        raise
    except Exception as e:
        print("======== ERROR =========")
        print(f"error update document: {e}")
        print("========================")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/documents", status_code=201)
async def post_document(
        title: str = Form(...),
        description: str = Form(...),
        file: UploadFile = File(...),
        db: AsyncSession = Depends(get_db),
        credentials: JwtAuthorizationCredentials = Security(utils.access_security)
    ):
    try:
        user_id = credentials.subject["user_id"]
        get_user = await db.execute(select(models.User.role).where(
            models.User.id == user_id
        ))

        user_role = get_user.scalars().first()

        if not user_role:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        get_document = await db.execute(select(models.Document.title).where(
            models.Document.title == title
        ))

        document_exist = get_document.scalars().first()

        if document_exist:
            raise HTTPException(status_code=400, detail="title already in use.")


        # ---------- make sure the file type is supported ----------
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            # Fallback: cek ekstensi filename jika content_type tidak dikenali browser
            filename = file.filename or ""
            if filename.lower().endswith(".md"):
                file.content_type = "text/markdown"
            else:
                raise HTTPException(
                    status_code=400,
                    detail="file type is not supported. Allowed: PDF, DOCX, TXT, Markdown (.md)"
                )

        # ---------- make sure file size is under 5mb ----------
        if file.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="file size exceeds the specified maximum limit (5mb).")

        # ---------- Insert the file's informations into the database ----------
        document_instance = models.Document(title=title, description=description, chunk_count=0, user_id=user_id)
        db.add(document_instance)
        await db.commit()
        await db.refresh(document_instance)

        # ---------- Read and extract text ----------
        content = await file.read()
        text = extract_text_from_file(content, file.content_type)

        bucket_name = os.getenv("BUCKET_NAME")
        file_path = f"uploads/{document_instance.id}/{file.filename}"

        upload_content_type = file.content_type
        if upload_content_type == "text/markdown" or upload_content_type == "text/x-markdown":
            upload_content_type = "text/plain"
        
        try:
            res = utils.supabase.storage.from_(bucket_name).upload(
                path=file_path,
                file=content,
                file_options={"content-type": upload_content_type, "x-upsert": "true"}
            )
            uploaded_path = res.path
            
            document_instance.file_path = uploaded_path
            await db.commit()
            
        except Exception as e:
            await db.rollback()
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

        task = process_document.delay(
            filename=file.filename,
            document_id=document_instance.id,
            title=title,
            description=description,
            content_type=file.content_type,
            text=text,
            is_update=False
        )

        response_data = {
            "id": document_instance.id,
            "title": document_instance.title,
            "description": document_instance.description,
            "chunk_count": document_instance.chunk_count,
            "user_id": document_instance.user_id,
            "task_id": task.id
        }

        return response_data
    except IntegrityError as e:
        await db.rollback()

        print(f"post document integrity error: {e}")

        if "unique" in str(e.orig).lower():
            raise HTTPException(status_code=400, detail="title already in use.")
                
        raise HTTPException(status_code=400, detail="An error occurred in the data.")
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        print("======== ERROR =========")
        print(f"error post document: {e}")
        print("========================")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(utils.access_security)
):
    try:
        result = await db.execute(select(models.Document).where(
            models.Document.id == document_id,
            models.Document.user_id == credentials.subject["user_id"]
        ))
        document = result.scalars().first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found or unauthorized")

        doc_manager = PineconeDocumentManager()

        def delete_from_vector_stores():
            doc_manager.delete_document(doc_id=str(document_id))

        await asyncio.to_thread(delete_from_vector_stores)

        if document.file_path:
            try:
                utils.supabase.storage.from_(os.getenv("BUCKET_NAME")).remove([document.file_path])
            except Exception as e:
                print(f"⚠️ Supabase storage deletion warning: {e}")

        await db.delete(document)
        await db.commit()

    except HTTPException as e:
        await db.rollback()
        raise
    except Exception as e:
        print(f"======= error deleting document {document_id} =======")
        print(e)
        print("=======================================")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")

# --- SESSION & CHAT ENDPOINTS ---

@router.get("/sessions", status_code=200)
async def get_sessions(db: AsyncSession = Depends(get_db), credentials: JwtAuthorizationCredentials = Security(utils.access_security)):
    try:
        user_id = credentials.subject["user_id"]

        get_user = await db.execute(select(models.User).where(
            models.User.id == user_id
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        get_chat_sessions = await db.execute(select(models.ChatSession).where(
            models.ChatSession.user_id == user_id
        ))

        chat_sessions = get_chat_sessions.scalars().all()

        if not chat_sessions:
            raise HTTPException(status_code=404, detail="No session found.")

        return chat_sessions
    except HTTPException as e:
        await db.rollback()
        raise e
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/sessions", status_code=201)
async def create_session(db: AsyncSession = Depends(get_db), credentials: JwtAuthorizationCredentials = Security(utils.access_security)):
    try:
        user_id = credentials.subject["user_id"]

        get_user = await db.execute(select(models.User).where(
            models.User.id == user_id
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        chat_session = models.ChatSession(user_id=user_id)
        db.add(chat_session)
        await db.commit()

        return chat_session
    except HTTPException as e:
        await db.rollback()
        raise e
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/sessions/{chat_session_id}/query", status_code=200)
async def session_query(
        chat_session_id: int,
        input: schemas.Query,
        db: AsyncSession = Depends(get_db),
        credentials: JwtAuthorizationCredentials = Security(utils.access_security)
    ):
    try:
        query_engine = utils.get_query_engine()
        user_id = credentials.subject["user_id"]

        get_user = await db.execute(select(models.User).where(
            models.User.id == user_id
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        if not chat_session_id:
            raise HTTPException(status_code=400, detail="Session id is required.")

        get_chat_sessions = await db.execute(
            select(models.ChatSession)
            .where(
                models.ChatSession.id == chat_session_id,
                models.ChatSession.user_id == user_id
            )
        )

        chat_sessions = get_chat_sessions.scalars().first()

        if not chat_sessions:
            raise HTTPException(status_code=404, detail="No session found.")

        get_document = await db.execute(select(models.Document).where(
            models.Document.id == input.document_id
        ))

        document = get_document.scalars().first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found.")

        if document.status == "FAILED":
            raise HTTPException(
                status_code=500, 
                detail="Document indexing failed. Please re-upload the document."
            )
        elif document.status == "PENDING":
            raise HTTPException(
                status_code=409, 
                detail="Document is pending indexing. Please wait and try again later."
            )

        user_chat_message = models.ChatMessage(session_id=chat_session_id, user_id=user_id, role="user", content=input.message)
        db.add(user_chat_message)

        async def load_conversation_history():
            get_messages = await db.execute(
                select(models.ChatMessage)
                .where(models.ChatMessage.session_id == chat_session_id)
                .order_by(models.ChatMessage.created_at)
                .limit(20)
            )

            messages = get_messages.scalars().all()

            if not messages:
                return f"No messages in session id {chat_session_id}"

            str_messages = ""
            for msg in messages:
                str_messages += f"{msg.role}: {msg.content}\n"

            return str_messages

        async def query_docs(query: str):
            res = await query_engine.query(query, document_id=input.document_id)
            contexts = res.get("contexts", [])
    
            if not contexts:
                return "Tidak ditemukan informasi relevan dalam dokumen."
            
            parts = []
            for i, ctx in enumerate(contexts, 1):
                parts.append(f"QUOTE {i} (Relevance: {ctx['relevance_score']:.2f}):\n{ctx['text']}")
            
            return "\n\n".join(parts)

        prompt_file_path = Path(__file__).parent / "system_prompt.txt"
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt = prompt_template.format(document_id=input.document_id)

        llm_messages = [
            LLMChatMessage(role="system", content=prompt),
            LLMChatMessage(role="user", content=f"""Berikut riwayat percakapan sebelumnya:
{await load_conversation_history()}

Berikut informasi relevan dari dokumen:
{await query_docs(input.message)}

Pertanyaan user: {input.message}

Jawablah pertanyaan di atas secara detail, komprehensif, dan lengkap berdasarkan informasi dari dokumen. Jika informasi tidak cukup, jelaskan alasannya. Jangan berikan jawaban singkat."""),
        ]

        llm = query_engine.llm
        response = await llm.achat(llm_messages)
        answer = response.message.content

        assistant_chat_message = models.ChatMessage(session_id=chat_session_id, user_id=user_id, role="assistant", content=str(answer))
        db.add(assistant_chat_message)
        await db.commit()

        return {"response": str(answer)}
    except HTTPException as e:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        print("========= ERROR =========")
        print(f"error: {e}")
        print("=========================")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/sessions/{session_id}/history", status_code=200)
async def get_session_history(
        session_id: int,
        db: AsyncSession = Depends(get_db),
        credentials: JwtAuthorizationCredentials = Security(utils.access_security)
    ):
    try:
        user_id = credentials.subject["user_id"]

        get_user = await db.execute(select(models.User).where(
            models.User.id == user_id
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        get_messages = await db.execute(select(models.ChatMessage).where(
            models.ChatMessage.session_id == session_id,
            models.ChatMessage.user_id == user_id
        ))

        messages = get_messages.scalars().all()

        if not messages:
            raise HTTPException(status_code=404, detail="No message found.")
        
        return messages
    except HTTPException as e:
        await db.rollback()
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

# --- TASK ENDPOINTS ---

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    try:
        result = AsyncResult(task_id, app=celery_task)
    
        response = {
            "task_id": task_id,
            "status": result.state,
            "ready": result.ready(),
        }
        
        if result.state == "SUCCESS":
            response["result"] = "Task completed"
        elif result.state == "FAILURE":
            response["error"] = "An error occurred while processing the document. Please try again."  
        elif result.state == "PENDING":
            response["message"] = "Task has not been processed or not found."
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")