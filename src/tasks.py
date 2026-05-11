from celery import Celery
from src.tasks_database import SessionLocal
from src.app import models
import os
from dotenv import load_dotenv
from src.app.lancedb_manager import LanceDBDocumentManager

load_dotenv()

celery_task = Celery(
    'tasks',
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_BACKEND_URL")
)

doc_manager = LanceDBDocumentManager()

@celery_task.task(bind=True, max_retries=3, default_retry_delay=30, queue="document_task")
def process_document(self, filename: str, document_id: int, title: str, description: str, content_type: str, text: str, is_update: bool = False):
    """
    Worker hanya handle vector ingestion.
    Upload sudah dilakukan di API.
    """
    db = SessionLocal()
    
    try:
        print(f"[Process] Ingest for document {document_id}")
        
        try:
            if is_update:
                print(f"[Process] Deleting document {document_id}")
                doc_manager.delete_document(str(document_id))
            
            print(f"[Process] Upserting document {document_id}")
            doc_manager.upsert_document(doc_id=str(document_id), text=text, metadata={"filename": filename})
            
            final_status = "SUCCESS"
            print(f"[Process] Ingest SUCCESS")
            
        except Exception as ingest_err:
            final_status = "FAILED"
            print(f"[Process] Ingest FAILED: {ingest_err}")
        
        # Update DB status saja (file_path sudah di-set di API)
        record = db.query(models.Document).filter_by(id=document_id).first()
        if record:
            record.status = final_status
            db.commit()
        
        print(f"[Process] Final status for document {document_id}: {final_status}")
        
        return {
            "document_id": document_id,
            "status": final_status
        }
        
    except Exception as exc:
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