import os
from typing import List, Optional
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from src.constants import settings
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================
LANCEDB_URI = settings.lancedb_uri
LANCEDB_TABLE = settings.lancedb_table

_embed_model: Optional[HuggingFaceInferenceAPIEmbedding] = None


def get_embed_model() -> HuggingFaceInferenceAPIEmbedding:
    """Get or create singleton embedding model."""
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name="intfloat/multilingual-e5-large",
            token=settings.hugging_face_api_key
        )
    return _embed_model


def create_vector_store() -> LanceDBVectorStore:
    """
    Create FRESH vector store instance.
    NOT a singleton — always create new to avoid stale cache.
    """
    vs = LanceDBVectorStore(
        uri=LANCEDB_URI,
        table_name=LANCEDB_TABLE,
        mode="create",
        query_type="vector",
    )

    # Fix for llama-index < 0.12 bug: _metadata_keys is None after restart
    if vs._metadata_keys is None:
        try:
            if vs._table is not None:
                schema = vs.table.schema
                metadata_keys = []
                for field in schema:
                    if field.name == "metadata" and hasattr(field.type, "__iter__"):
                        metadata_keys = [f.name for f in field.type]
                        break
                vs._metadata_keys = metadata_keys
            else:
                vs._metadata_keys = []
        except Exception:
            vs._metadata_keys = []

    return vs


# ============================================================================
# LanceDB Document Manager
# ============================================================================

class LanceDBDocumentManager:
    """
    Use LlamaIndex for orchestration, LanceDB for storage.
    No nodes in RAM — always fetch from LanceDB.
    """

    def __init__(self):
        self.vector_store = create_vector_store()
        self.embed_model = get_embed_model()

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        Settings.embed_model = self.embed_model
        Settings.llm = Groq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature
        )

    def upsert_document(self, doc_id: str, text: str) -> str:
        """
        Upsert document to LanceDB.

        Args:
            doc_id: Unique document identifier (postgres_id)
            text: Document text content
            metadata: Additional metadata

        Returns:
            The upserted doc_id
        """
        if not text or not text.strip():
            raise ValueError("Document text cannot be empty")

        if len(text) > 10_000_000:  # 10MB text limit
            raise ValueError("Document text exceeds maximum size (10MB)")

        doc = Document(
            text=text,
            id_=doc_id,
            metadata={
                "postgres_id": doc_id
            }
        )

        index = VectorStoreIndex.from_documents(
            [doc],
            storage_context=self.storage_context,
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=50)
            ],
            show_progress=True
        )

        if not self.vector_store._metadata_keys:
            self.vector_store._metadata_keys = list(doc.metadata.keys())

        # CRITICAL: Force close the connection to prevent a stale cache
        # LanceDB maintains a cache at the connection level
        try:
            if hasattr(self.vector_store, '_connection') and self.vector_store._connection:
                delattr(self.vector_store, '_connection')
        except Exception:
            pass

        logger.info(f"✅ Upserted doc_id: {doc_id} via LlamaIndex → LanceDB")
        return doc_id

    def delete_document(self, doc_id: str) -> None:
        """Delete document from LanceDB by doc_id."""
        try:
            self.vector_store.delete(ref_doc_id=doc_id)
            logger.info(f"🗑️  Deleted doc_id: {doc_id}")

            try:
                if hasattr(self.vector_store, '_connection') and self.vector_store._connection:
                    delattr(self.vector_store, '_connection')
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"⚠️  Error deleting doc_id {doc_id}: {e}")
            raise

    def delete_all(self) -> None:
        """Delete all data in the table. USE WITH CAUTION."""
        if self.vector_store._table_exists():
            self.vector_store._connection.drop_table(LANCEDB_TABLE)
            logger.warning(f"⚠️  Dropped entire table: {LANCEDB_TABLE}")