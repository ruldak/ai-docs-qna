from celery import Celery
from fastapi.encoders import jsonable_encoder
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from src.tasks_database import SessionLocal
from src.users import models
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
from dotenv import load_dotenv
from llama_index.core import Document
import httpx
from .users import utils

custom_timeout = httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=10.0)
custom_limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)


transport = httpx.HTTPTransport(retries=3)

client = httpx.Client(timeout=custom_timeout, limits=custom_limits, transport=transport)

load_dotenv()

app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@app.task(bind=True, max_retries=3, default_retry_delay=30, queue="io_task")
def insert_to_vector(self, contents, text, filename, document_id, title, description, content_type):
    """Sebuah tugas untuk memasukan document ke vector database."""
    print("Menjalankan tugas: memasukan document ke vector database")
    
    db = SessionLocal()

    try:
        # Upload document to the supabase storage
        print("Start uploading document...")
        bucket_name = os.getenv("BUCKET_NAME")
        file_path = f"uploads/{document_id}/{filename}"
        
        res = utils.supabase.storage.from_(bucket_name).upload(
            path=file_path,
            file=contents,
            file_options={"content-type": content_type, "x-upsert": "true"}
        )

        client = chromadb.CloudClient(
            api_key=os.getenv("API_KEY"),
            tenant=os.getenv("TENANT"),
            database='fastapi_qna_legal_documents'
        )

        collection = client.get_or_create_collection("legal_documents")
        vstore = ChromaVectorStore(chroma_collection=collection)

        embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name="BAAI/bge-m3",
            token=os.getenv("HUGGING_FACE_API_KEY"),
            http_client=client
        )

        docs = [
            Document(
                text=text,
                metadata={
                    "title": title,
                    "description": description,
                    "filename": filename,
                    "id": document_id
                }
            ),
        ]
        
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512),
                embed_model
            ],
            vector_store=vstore
        )

        record = db.query(models.Document).filter_by(id=document_id).first()
        if not record:
            record = models.Document(title=title, description=description, chunk_count=0, status="PENDING")
            db.add(record)
    
        # Insert into chroma
        print(f"Start embedding, text length: {len(text)} characters")
        nodes = pipeline.run(documents=docs)
        print(f"Embedding finish, total nodes: {len(nodes)}")
        
        record.status = "SUCCESS"
        record.chunk_count = len(nodes)
        record.file_path = res.path
        db.commit()
    except Exception as exc:
        try:
            if self.request.retries >= self.max_retries:
                record.status = "FAILED"
                db.commit()
        except:
            db.rollback()

        print("====== error ======")
        print(exc)
        print("============")

        raise self.retry(exc=exc)
    finally:
        db.close()
    
    print("Tugas selesai.")
