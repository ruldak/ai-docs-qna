from fastapi import APIRouter, Depends, HTTPException, Security, UploadFile, File, Form
from . import utils, service, models, schemas
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
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"error: {e}")

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
        raise HTTPException(status_code=500, detail=f"error: {e}")


@router.get("/documents", status_code=200)
async def get_documents(db: AsyncSession = Depends(get_db)):
    try:
        # ---------- Get documents ----------
        get_documents = await db.execute(select(models.Document))

        documents = get_documents.scalars().all()

        if not documents:
            raise HTTPException(status_code=404, detail="No documents found.")

        return {"detail": documents}
    except HTTPException:
        raise
    except Exception as e:
        raise e

@router.post("/documents/post", status_code=201)
async def post_document(
        title: str = Form(...),
        description: str = Form(...),
        file: UploadFile = File(...),
        db: AsyncSession = Depends(get_db),
        credentials: JwtAuthorizationCredentials = Security(utils.access_security)
    ):
    try:
        # ---------- Check token owner ----------
        get_user = await db.execute(select(models.User.role).where(
            models.User.id == credentials.subject["user_id"]
        ))

        user_role = get_user.scalars().first()

        if not user_role:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        if user_role != "admin":
            raise HTTPException(status_code=403, detail="Forbidden")

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

        # ---------- Read and insert file into the vector store ----------
        content = await file.read()
        
        if file.content_type == "application/pdf":
            doc = fitz.open(stream=content, filetype="pdf")


        texts = []
        for page in doc:
            texts.append(page.get_text())


        text = "\n".join(texts)

        docs = [
            Document(
                text=text,
                metadata={
                    "title": title,
                    "description": description,
                    "filename": file.filename
                }
            ),
        ]

        client = chromadb.CloudClient(
            api_key=os.getenv("API_KEY"),
            tenant=os.getenv("TENANT"),
            database='fastapi_qna_legal_documents'
        )

        collection = client.get_or_create_collection("legal_documents")
        vstore = ChromaVectorStore(chroma_collection=collection)

        embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name="intfloat/multilingual-e5-large",
            token=os.getenv("HUGGING_FACE_API_KEY")
        )

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512),
                embed_model
            ],
            vector_store=vstore
        )

        nodes = await pipeline.arun(documents=docs)

        # ---------- Insert the file's informations into the database ----------
        document_instance = models.Document(title=title, description=description, chunk_count=len(nodes))
        db.add(document_instance)
        await db.commit()
        await db.refresh(document_instance)

        return document_instance
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
        await db.rollback()
        raise e

@router.delete("/documents/delete", status_code=204)
async def delete_document(
        id: str,
        db: AsyncSession = Depends(get_db),
        credentials: JwtAuthorizationCredentials = Security(utils.access_security)
    ):
    try:
        # ---------- Check token owner ----------
        get_user = await db.execute(select(models.User.role).where(
            models.User.id == credentials.subject["user_id"]
        ))

        user_role = get_user.scalars().first()

        if not user_role:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials")

        if user_role != "admin":
            raise HTTPException(status_code=403, detail="Forbidden")

        # ---------- Get the document ----------
        get_document = await db.execute(select(models.Document).where(
            models.Document.id == int(id)
        ))

        document_exist = get_document.scalars().first()

        if not document_exist:
            raise HTTPException(status_code=404, detail="Document not found.")

        client = chromadb.CloudClient(
            api_key=os.getenv("API_KEY"),
            tenant=os.getenv("TENANT"),
            database='fastapi_qna_legal_documents'
        )

        collection = client.get_or_create_collection("legal_documents")
        collection.delete(where={"title": document_exist.title})
        
        await db.delete(document_exist)
        await db.commit()

        return {"detail": "Deleted Successfully."}
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        raise e