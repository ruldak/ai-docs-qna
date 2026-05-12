from celery import Celery
from src.tasks_database import SessionLocal
from src.app import models
import os
from dotenv import load_dotenv
from src.app.lancedb_manager import LanceDBDocumentManager
from src.app.evaluator import LlamaIndexEvaluator
from src.app.utils import get_query_engine

load_dotenv()

celery_task = Celery(
    'tasks',
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_BACKEND_URL")
)

doc_manager = LanceDBDocumentManager()

# Reuse LLM instance dari query engine — konsisten dengan stack LlamaIndex
query_engine = get_query_engine()
evaluator = LlamaIndexEvaluator(llm=query_engine.llm)


@celery_task.task(bind=True, max_retries=3, default_retry_delay=30, queue="document_task")
def process_document(self, filename: str, document_id: int, title: str, description: str, content_type: str, text: str, is_update: bool = False):
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


@celery_task.task(bind=True, max_retries=2, default_retry_delay=10, queue="evaluation_task")
def evaluate_chat_message(self, evaluation_id: int, question: str):
    """
    Background task: Evaluasi chat message dengan LlamaIndex Groq LLM-as-a-Judge.
    'question' di-pass langsung dari API supaya worker tidak perlu query DB 
    (menghindari race condition transaksi belum commit).
    """
    db = SessionLocal()
    
    try:
        eval_record = db.query(models.ChatEvaluation).filter_by(id=evaluation_id).first()
        if not eval_record:
            print(f"[Eval] Evaluation {evaluation_id} not found")
            return {"status": "NOT_FOUND"}
        
        message = db.query(models.ChatMessage).filter_by(id=eval_record.message_id).first()
        if not message:
            raise ValueError(f"Message for evaluation {evaluation_id} not found")
        
        answer = message.content
        contexts = eval_record.contexts or []
        
        print(f"[Eval] Running evaluation for message {message.id}")
        
        result = evaluator.evaluate(
            question=question,
            answer=answer,
            contexts=contexts
        )
        
        eval_record.faithfulness_score = result["faithfulness_score"]
        eval_record.answer_relevancy_score = result["answer_relevancy_score"]
        eval_record.faithfulness_reasoning = result["faithfulness_reasoning"]
        eval_record.relevancy_reasoning = result["answer_relevancy_reasoning"]
        eval_record.raw_eval_response = {
            "faithfulness": result["raw_faithfulness"],
            "relevancy": result["raw_relevancy"]
        }
        eval_record.status = "SUCCESS"
        eval_record.error_message = None
        
        db.commit()
        
        print(f"[Eval] SUCCESS for message {message.id} | Faithfulness: {result['faithfulness_score']} | Relevancy: {result['answer_relevancy_score']}")
        
        return {
            "evaluation_id": evaluation_id,
            "status": "SUCCESS",
            "faithfulness_score": result["faithfulness_score"],
            "answer_relevancy_score": result["answer_relevancy_score"]
        }
        
    except Exception as exc:
        db.rollback()
        
        try:
            eval_record = db.query(models.ChatEvaluation).filter_by(id=evaluation_id).first()
            if eval_record:
                eval_record.status = "FAILED"
                eval_record.error_message = str(exc)
                db.commit()
        except:
            db.rollback()
        
        print(f"[Eval] FAILED for evaluation {evaluation_id}: {exc}")
        raise self.retry(exc=exc)
        
    finally:
        db.close()