"""
Celery task definitions for background processing.
Handles document ingestion and chat evaluation.

FIX: Setiap task menggunakan fresh LanceDB connection untuk menghindari stale data.
"""

import os
import logging
from typing import Dict, Any

from celery import Celery
from sqlalchemy.orm import Session

from src.database import SyncSessionLocal
from src.app import models
from src.app.lancedb_manager import LanceDBDocumentManager
from src.app.evaluator import LlamaIndexEvaluator
from src.app.utils import get_query_engine
from src.constants import settings

logger = logging.getLogger(__name__)

# ============================================================================
# CELERY APP
# ============================================================================
celery_app = Celery(
    "tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_backend_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Jakarta",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    worker_prefetch_multiplier=1,
)

# ============================================================================
# SINGLETON MANAGERS
# ============================================================================
_evaluator: LlamaIndexEvaluator = None

def get_evaluator() -> LlamaIndexEvaluator:
    """Get or create singleton evaluator."""
    global _evaluator
    if _evaluator is None:
        query_engine = get_query_engine()
        _evaluator = LlamaIndexEvaluator(llm=query_engine.llm)
    return _evaluator


# ============================================================================
# DATABASE HELPERS
# ============================================================================

def get_db_session() -> Session:
    """Create a new sync database session."""
    return SyncSessionLocal()


def update_document_status(db: Session, document_id: int, status: str) -> bool:
    """Update document status dengan proper error handling."""
    try:
        record = db.query(models.Document).filter_by(id=document_id).first()
        if record:
            record.status = status
            db.commit()
            return True
        logger.warning(f"Document {document_id} not found for status update")
        return False
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update document {document_id} status: {e}")
        raise


# ============================================================================
# TASKS
# ============================================================================

@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="document_task",
    name="tasks.process_document"
)
def process_document(
    self,
    filename: str,
    document_id: int,
    title: str,
    description: str,
    content_type: str,
    text: str,
    is_update: bool = False
) -> Dict[str, Any]:
    """
    Process document ingestion ke LanceDB.

    FIX: Setiap task membuat instance LanceDBDocumentManager FRESH
    untuk menghindari stale cache dari task sebelumnya.
    """
    logger.info(f"[Process] Starting ingestion for document {document_id}")

    db = get_db_session()

    doc_manager = LanceDBDocumentManager()

    try:
        # Validate document exists
        doc_record = db.query(models.Document).filter_by(id=document_id).first()
        if not doc_record:
            logger.error(f"Document {document_id} not found in database")
            return {
                "document_id": document_id,
                "status": "FAILED",
                "error": "Document not found in database"
            }

        # Update status to PROCESSING
        update_document_status(db, document_id, "PROCESSING")

        try:
            if is_update:
                logger.info(f"[Process] Deleting existing document {document_id}")
                doc_manager.delete_document(str(document_id))

            logger.info(f"[Process] Upserting document {document_id}")
            doc_manager.upsert_document(
                doc_id=str(document_id),
                text=text
            )

            final_status = "SUCCESS"
            logger.info(f"[Process] Ingestion SUCCESS for document {document_id}")

        except Exception as ingest_err:
            final_status = "FAILED"
            logger.error(f"[Process] Ingestion FAILED for document {document_id}: {ingest_err}")
            raise self.retry(exc=ingest_err)

        # Update final status
        update_document_status(db, document_id, final_status)

        return {
            "document_id": document_id,
            "status": final_status
        }

    except Exception as exc:
        try:
            update_document_status(db, document_id, "FAILED")
        except Exception as update_err:
            logger.error(f"Failed to mark document {document_id} as FAILED: {update_err}")

        logger.error(f"[Process] Task failed for document {document_id}: {exc}")
        raise self.retry(exc=exc)

    finally:
        db.close()


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    queue="evaluation_task",
    name="tasks.evaluate_chat_message"
)
def evaluate_chat_message(
    self,
    evaluation_id: int,
    question: str
) -> Dict[str, Any]:
    """
    Evaluasi chat message dengan LlamaIndex Groq LLM-as-a-Judge.
    """
    logger.info(f"[Eval] Starting evaluation {evaluation_id}")

    db = get_db_session()
    evaluator = get_evaluator()

    try:
        eval_record = db.query(models.ChatEvaluation).filter_by(id=evaluation_id).first()
        if not eval_record:
            logger.warning(f"[Eval] Evaluation {evaluation_id} not found")
            return {"status": "NOT_FOUND", "evaluation_id": evaluation_id}

        message = db.query(models.ChatMessage).filter_by(id=eval_record.message_id).first()
        if not message:
            raise ValueError(f"Message for evaluation {evaluation_id} not found")

        answer = message.content
        contexts = eval_record.contexts or []

        logger.info(f"[Eval] Running evaluation for message {message.id}")

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

        logger.info(
            f"[Eval] SUCCESS for message {message.id} | "
            f"Faithfulness: {result['faithfulness_score']} | "
            f"Relevancy: {result['answer_relevancy_score']}"
        )

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
                eval_record.error_message = str(exc)[:1000]
                db.commit()
        except Exception as update_err:
            db.rollback()
            logger.error(f"Failed to mark evaluation {evaluation_id} as FAILED: {update_err}")

        logger.error(f"[Eval] FAILED for evaluation {evaluation_id}: {exc}")
        raise self.retry(exc=exc)

    finally:
        db.close()