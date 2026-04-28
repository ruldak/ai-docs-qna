from fastapi import APIRouter, Depends, HTTPException, Security, UploadFile, File, Form
from . import utils, service, models, schemas, constants
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_db
from sqlalchemy import select, update, delete
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi_jwt import JwtAuthorizationCredentials
from sqlalchemy.exc import IntegrityError
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader, DocxReader
import os
from dotenv import load_dotenv
import fitz
from llama_index.core import Document
import docx2txt
import io
from typing import Optional
import asyncio
from datetime import datetime, timezone
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.storage.chat_store.postgres import PostgresChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.core.agent.workflow import FunctionAgent
from src.tasks import upload_document, celery_task
from celery.result import AsyncResult
from typing import List
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from .rag import get_index

load_dotenv()

router = APIRouter(prefix="/api")

@router.get("/users/me", response_model=schemas.User, status_code=200)
async def users(db: AsyncSession = Depends(get_db), credentials: JwtAuthorizationCredentials = Security(utils.access_security)):
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

@router.post("/users/register", response_model=schemas.User, status_code=201)
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

@router.post("/users/login", response_model=schemas.LoginResponse)
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


@router.get("/users/documents", response_model=List[schemas.DocumentResponseList], status_code=200)
async def get_documents(db: AsyncSession = Depends(get_db), credentials: JwtAuthorizationCredentials = Security(utils.access_security)):
    try:
        # ---------- Check token's owner ----------
        get_user = await db.execute(select(models.User).where(
            models.User.id == credentials.subject["user_id"]
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        # ---------- Get documents ----------
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


@router.get("/users/documents/{document_id}", response_model=schemas.DocumentResponse, status_code=200)
async def get_document_by_id(
        document_id: int,
        db: AsyncSession = Depends(get_db),
        credentials: JwtAuthorizationCredentials = Security(utils.access_security)
    ):
    try:
        # ---------- Check token's owner ----------
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
            # Create Signed URL
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


@router.get("/documents", response_model=List[schemas.DocumentResponseList], status_code=200)
async def get_documents(db: AsyncSession = Depends(get_db)):
    try:
        # ---------- Get documents ----------
        get_documents = await db.execute(select(models.Document))

        documents = get_documents.scalars().all()

        if not documents:
            raise HTTPException(status_code=404, detail="No documents found.")

        return documents
    except HTTPException:
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
        # ---------- Check token's owner ----------
        get_user = await db.execute(select(models.User).where(
            models.User.id == credentials.subject["user_id"]
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        # ---------- Check if all fields are empty ----------
        if not title and not description and not file:
            raise HTTPException(status_code=400, detail="One of the fields must be filled in.")

        new_values = {}

        task = None

        if file:
            client = chromadb.PersistentClient(path="./chroma_db")

            collection = client.get_or_create_collection(os.getenv("COLLECTION_NAME"))
            vstore = ChromaVectorStore(chroma_collection=collection)

            # ---------- make sure the file type is supported ----------
            if file.content_type not in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain", "application/pdf"]:
                raise HTTPException(status_code=400, detail="file type is not supported.")

            # ---------- make sure file size is under 5mb ----------
            if file.size > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="file size exceeds the specified maximum limit (5mb).")

            # ---------- retrieve the document based on the given ID ----------
            get_document = await db.execute(select(models.Document).where(
                models.Document.id == document_id,
                models.Document.user_id == credentials.subject["user_id"]
            ))

            document = get_document.scalars().first()

            # ---------- check if the document exist ----------
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")

            if document.file_path:
                # ---------- delete the document from the supabase storage -----------
                storage_response = utils.supabase.storage.from_(os.getenv("BUCKET_NAME")).remove([document.file_path])

            # ---------- Read and insert file into the vector store ----------
            content = await file.read()
            
            if file.content_type == "application/pdf":
                doc = fitz.open(stream=content, filetype="pdf")

                texts = []
                for page in doc:
                    texts.append(page.get_text())

                text = "\n".join(texts)
            
            elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # EN: wrap the read binary data with BytesIO() so that it can be read() again because docx2txt.process() needs it.
                # ID: bungkus data biner yang telah dibaca dengan BytesIO() supaya bisa di read() ulang karena docx2txt.process() butuh itu.
                file_like = io.BytesIO(content)
                text = docx2txt.process(file_like)
            elif file.content_type == "text/plain":
                text = content.decode("utf-8")

            docs = Document(
                text=text,
                metadata={
                    "filename": file.filename,
                    "id": document_id
                },
                doc_id=str(document_id)
            )

            index = get_index()
            await asyncio.to_thread(index.delete_ref_doc, str(document_id), delete_from_store=True)
            await asyncio.to_thread(index.insert, docs)

            task = upload_document.delay(
                contents=content,
                filename=file.filename,
                title=document.title,
                description=document.description,
                document_id=document_id,
                content_type=file.content_type
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
        # ---------- Check token's owner ----------
        get_user = await db.execute(select(models.User.role).where(
            models.User.id == user_id
        ))

        user_role = get_user.scalars().first()

        if not user_role:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        # ---------- Check if the title is already in use ----------
        get_document = await db.execute(select(models.Document.title).where(
            models.Document.title == title
        ))

        document_exist = get_document.scalars().first()

        if document_exist:
            raise HTTPException(status_code=400, detail="title already in use.")

        # ---------- make sure the file type is supported ----------
        if file.content_type not in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain", "application/pdf"]:
            raise HTTPException(status_code=400, detail="file type is not supported.")

        # ---------- make sure file size is under 5mb ----------
        if file.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="file size exceeds the specified maximum limit (5mb).")


        # ---------- Insert the file's informations into the database ----------
        document_instance = models.Document(title=title, description=description, chunk_count=0, user_id=user_id)
        db.add(document_instance)
        await db.commit()

        # ---------- Read and insert file into the vector store ----------
        content = await file.read()
        
        if file.content_type == "application/pdf":
            doc = fitz.open(stream=content, filetype="pdf")

            texts = []
            for page in doc:
                texts.append(page.get_text())

            text = "\n".join(texts)
        
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # EN: wrap the read binary data with BytesIO so that it can be read() again because docx2txt.process() needs it.
            # ID: bungkus data biner yang telah dibaca dengan BytesIO supaya bisa di read() ulang karena docx2txt.process() butuh itu.
            file_like = io.BytesIO(content)
            text = docx2txt.process(file_like)
        elif file.content_type == "text/plain":
            text = content.decode("utf-8")

        docs = Document(
            text=text,
            metadata={
                "filename": file.filename,
                "id": document_instance.id
            },
            doc_id=str(document_instance.id)
        )

        index = get_index()
        await asyncio.to_thread(index.insert, docs)

        task = upload_document.delay(
            contents=content,
            filename=file.filename,
            title=title,
            description=description,
            document_id=document_instance.id,
            content_type=file.content_type
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
        # ---------- Check token's owner ----------
        get_user = await db.execute(select(models.User.role).where(
            models.User.id == credentials.subject["user_id"]
        ))

        user_role = get_user.scalars().first()

        if not user_role:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        # ---------- Get the document ----------
        get_document = await db.execute(select(models.Document).where(
            models.Document.id == document_id,
            models.Document.user_id == credentials.subject["user_id"]
        ))

        document_exist = get_document.scalars().first()

        if not document_exist:
            raise HTTPException(status_code=404, detail="Document not found.")

        client = chromadb.PersistentClient(path="./chroma_db")

        collection = client.get_or_create_collection(os.getenv("COLLECTION_NAME"))
        vstore = ChromaVectorStore(chroma_collection=collection)

        index = get_index()
        
        # delete the document from the vector database
        index.delete_ref_doc(str(document_id), delete_from_store=True)

        # delete the document from the supabase storage
        storage_response = utils.supabase.storage.from_(os.getenv("BUCKET_NAME")).remove([document_exist.file_path])
        
        await db.delete(document_exist)
        await db.commit()

        return {"detail": "Deleted Successfully."}
    except HTTPException as e:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/c/{chat_session_id}", status_code=200)
async def query(
        chat_session_id: int,
        input: schemas.Query,
        db: AsyncSession = Depends(get_db),
        credentials: JwtAuthorizationCredentials = Security(utils.access_security)
    ):
    try:
        query_tools = utils.QueryTools()
        user_id = credentials.subject["user_id"]

        # ---------- Check token's owner ----------
        get_user = await db.execute(select(models.User).where(
            models.User.id == user_id
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")


        if not chat_session_id:
            raise HTTPException(status_code=400, detail="Session id is required.")

        # ---------- Check if the session id exist ----------
        get_chat_sessions = await db.execute(select(models.ChatSession).where(
            models.ChatSession.id == chat_session_id
        ))

        chat_sessions = get_chat_sessions.scalars().first()

        if not chat_sessions:
            raise HTTPException(status_code=404, detail="No session found.")

        # ---------- Check if the document exist ----------
        get_document = await db.execute(select(models.Document).where(
            models.Document.id == input.document_id
        ))

        document = get_document.scalars().first()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found.")
            
        async def load_conversation_history():
            get_messages = await db.execute(select(models.ChatMessage).where(
                models.ChatMessage.session_id == chat_session_id
            ))

            messages = get_messages.scalars().all()

            if not messages:
                return f"No messages in session id {chat_session_id}"

            str_messages = ""
            for msg in messages:
                str_messages += f"{msg.role}: {msg.content}\n"

            return str_messages

        # ---------- Insert user message into the database ----------
        user_chat_message = models.ChatMessage(session_id=chat_session_id, user_id=user_id, role="user", content=input.message)
        db.add(user_chat_message)

        llm = query_tools.llm(0.6)

        query_documents = FunctionTool.from_defaults(
            async_fn=query_tools.query_documents,
            name="query_documents",
            description="Query documents based on questions related to the document and doc_id.",
            partial_params={"document_id": input.document_id},
        )
        
        load_conversation_history = FunctionTool.from_defaults(
            async_fn=load_conversation_history,
            name="load_conversation_history",
            description="Use this tool when you need the previous conversation context"
        )

        prompt = f"""You are an AI assistant specialized in document analysis.

        IMPORTANT RULES:
        - Always use the tool `load_conversation_history` before answering questions to understand the context.
        - ONLY call `query_documents` if the user's question explicitly asks for information from a specific document.
        - If the user asks about your behavior, asks you to not use tools, or says anything unrelated to document content, respond directly WITHOUT calling `query_documents`.
        - When you do call `query_documents`, use the document_id provided in the user's message (extract it from the text).
        - Otherwise, reply conversationally.

        Current document_id in context: {input.document_id}
        """
        
        agent = FunctionAgent(
            tools=[query_documents, load_conversation_history],
            llm=llm,
            system_prompt=prompt,
            streaming=False,
            verbose=True
        )
        
        response = await agent.run(input.message)

        # ---------- Insert the ai answer into the database and commit ----------
        assistant_chat_message = models.ChatMessage(session_id=chat_session_id, user_id=user_id, role="assistant", content=str(response))
        db.add(assistant_chat_message)
        await db.commit()

        return {"response": str(response)}
    except HTTPException as e:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        print("========= ERROR =========")
        print(f"error: {e}")
        print("=========================")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/c/history/{session_id}", status_code=200)
async def get_chat_history(
        session_id: int,
        db: AsyncSession = Depends(get_db),
        credentials: JwtAuthorizationCredentials = Security(utils.access_security)
    ):
    try:
        user_id = credentials.subject["user_id"]

        # ---------- Check token's owner ----------
        get_user = await db.execute(select(models.User).where(
            models.User.id == user_id
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        # ---------- retrieve chat messages from the given session id and user id. ----------
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

@router.get("/users/sessions", status_code=200)
async def get_session(db: AsyncSession = Depends(get_db), credentials: JwtAuthorizationCredentials = Security(utils.access_security)):
    try:
        user_id = credentials.subject["user_id"]

        # ---------- Check token's owner ----------
        get_user = await db.execute(select(models.User).where(
            models.User.id == user_id
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        # ---------- Load all of the chat sessions ----------
        get_chat_sessions = await db.execute(select(models.ChatSession).where(
            models.ChatSession.user_id == user_id
        ))

        chat_sessions = get_chat_sessions.scalars().all()

        if not chat_sessions:
            raise HTTPException(status_code=404, detail="No session found.")

        return chat_sessions
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    except HTTPException as e:
        await db.rollback()
        raise e

@router.post("/sessions", status_code=201)
async def create_session(db: AsyncSession = Depends(get_db), credentials: JwtAuthorizationCredentials = Security(utils.access_security)):
    try:
        user_id = credentials.subject["user_id"]

        # ---------- Check token's owner ----------
        get_user = await db.execute(select(models.User).where(
            models.User.id == user_id
        ))

        user = get_user.scalars().first()

        if not user:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        # ---------- Create new chat session ----------
        chat_session = models.ChatSession(user_id=user_id)
        db.add(chat_session)
        await db.commit()

        return chat_session
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    except HTTPException as e:
        await db.rollback()
        raise e

@router.get("/task/{task_id}")
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