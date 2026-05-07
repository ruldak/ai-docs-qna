from celery import Celery
from src.tasks_database import SessionLocal
from src.app import models
import os
from dotenv import load_dotenv
from .app import utils
from src.app.pinecone import PineconeDocumentManager
from llama_index.core import Document, StorageContext

load_dotenv()

celery_task = Celery(
    'tasks',
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_BACKEND_URL")
)

doc_manager = PineconeDocumentManager()

# tasks.py
@celery_task.task(bind=True, max_retries=3, default_retry_delay=30, queue="document_task")
def process_document(self, contents: str, filename: str, document_id: int, title: str, description: str, content_type: str, text: str, is_update: bool = False):
    """
    Single task yang menangani upload DAN ingest secara berurutan.
    Status diupdate SEKALI di akhir berdasarkan hasil keduanya.
    """
    db = SessionLocal()
    
    try:
        # Step 1: Upload ke Supabase
        print(f"[Process] Step 1: Upload for document {document_id}")
        bucket_name = os.getenv("BUCKET_NAME")
        file_path = f"uploads/{document_id}/{filename}"
        
        try:
            res = utils.supabase.storage.from_(bucket_name).upload(
                path=file_path,
                file=contents,
                file_options={"content-type": content_type, "x-upsert": "true"}
            )
            upload_success = True
            uploaded_path = res.path
            print(f"[Process] Upload SUCCESS")
        except Exception as upload_err:
            upload_success = False
            uploaded_path = None
            print(f"[Process] Upload FAILED: {upload_err}")
        
        # Step 2: Ingest ke Vector Store
        print(f"[Process] Step 2: Ingest for document {document_id}")
        
        try:
            if is_update:
                print(f"Deleting document {document_id}")
                doc_manager.delete_document(str(document_id))
            
            print(f"Upserting document {document_id}")
            doc_manager.upsert_document(str(document_id), text, {"filename": filename})
            
            ingest_success = True
            print(f"[Process] Ingest SUCCESS")
        except Exception as ingest_err:
            ingest_success = False
            print(f"[Process] Ingest FAILED: {ingest_err}")
        
        # Step 3: Determine final status dan update DB SEKALI
        if upload_success and ingest_success:
            final_status = "SUCCESS"
        else:
            final_status = "FAILED"
        
        record = db.query(models.Document).filter_by(id=document_id).first()
        if record:
            record.status = final_status
            if uploaded_path:
                record.file_path = uploaded_path
            db.commit()
        
        print(f"[Process] Final status for document {document_id}: {final_status}")
        
        # Return hasil untuk tracking
        return {
            "document_id": document_id,
            "status": final_status,
            "upload_success": upload_success,
            "ingest_success": ingest_success
        }
        
    except Exception as exc:
        # Jika ada error di luar step-step di atas
        try:
            record = db.query(models.Document).filter_by(id=document_id).first()
            if record:
                record.status = "FAILED"
                db.commit()
        except:
            db.rollback()
        
        raise self.retry(exc=exc)
    finally:
        db.close()
