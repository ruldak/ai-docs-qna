from celery import Celery
from src.tasks_database import SessionLocal
from src.app import models
import os
from dotenv import load_dotenv
from .app import utils
from .app.rag import get_index

load_dotenv()

celery_task = Celery(
    'tasks',
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_BACKEND_URL")
)

@celery_task.task(bind=True, max_retries=3, default_retry_delay=30, queue="io_task")
def insert_to_vector(self, contents: str, filename: str, document_id: int, title: str, description: str, content_type: str):
    """Sebuah tugas untuk memasukan document ke vector database."""
    print("Menjalankan tugas: upload dokumen ke supabase")
    
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

        record = db.query(models.Document).filter_by(id=document_id).first()
        if not record:
            record = models.Document(title=title, description=description, chunk_count=0, status="PENDING")
            db.add(record)
    
        record.status = "SUCCESS"
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
