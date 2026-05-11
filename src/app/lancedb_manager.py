"""
LlamaIndex + LanceDB integration.
LlamaIndex hanya untuk orchestration (chunking, query, LLM).
LanceDB = single source of truth (local file-based, tidak ada di RAM).
"""

import os
from typing import List, Optional
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIG
# ============================================================================
LANCEDB_URI = os.getenv("LANCEDB_URI", "./lancedb_data")
LANCEDB_TABLE = os.getenv("LANCEDB_TABLE", "documents")

# Singleton connection
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name="intfloat/multilingual-e5-large",
            token=os.getenv("HUGGING_FACE_API_KEY")
        )
    return _embed_model


def get_vector_store():
    vs = LanceDBVectorStore(
        uri=LANCEDB_URI,
        table_name=LANCEDB_TABLE,
        mode="create",
        query_type="vector",
    )
    
    # FIX untuk bug llama-index < 0.12: _metadata_keys None setelah restart
    # Error: TypeError: argument of type 'NoneType' is not iterable
    if vs._metadata_keys is None:
        try:
            # coba load dari schema table yang sudah ada
            if vs._table is not None:
                schema = vs.table.schema
                metadata_keys = []
                for field in schema:
                    if field.name == "metadata" and hasattr(field.type, "__iter__"):
                        # metadata adalah struct
                        metadata_keys = [f.name for f in field.type]
                        break
                vs._metadata_keys = metadata_keys
            else:
                vs._metadata_keys = []
        except Exception:
            # fallback aman
            vs._metadata_keys = []
    
    return vs


# ============================================================================
# LlamaIndex + LanceDB (No RAM mapping)
# ============================================================================

class LanceDBDocumentManager:
    """
    Pakai LlamaIndex untuk orchestration, LanceDB untuk storage.
    Tidak ada nodes di RAM — selalu fetch dari LanceDB.
    """

    def __init__(self):
        self.vector_store = get_vector_store()
        self.embed_model = get_embed_model()

        # Setup LlamaIndex vector store (bridge ke LanceDB)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # LlamaIndex Settings
        Settings.embed_model = self.embed_model
        Settings.llm = Groq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

    def upsert_document(self, doc_id: str, text: str, metadata: dict = None):
        doc = Document(
            text=text,
            id_=doc_id,
            metadata={
                "postgres_id": doc_id,
                **(metadata or {})
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

        # setelah upsert pertama, pastikan metadata_keys terisi
        if self.vector_store._metadata_keys is None or len(self.vector_store._metadata_keys) == 0:
            self.vector_store._metadata_keys = list(doc.metadata.keys())

        print(f"✅ Upserted doc_id: {doc_id} via LlamaIndex → LanceDB")
        return doc_id

    def delete_document(self, doc_id: str):
        """
        Hapus dari LanceDB via vector store.
        """
        self.vector_store.delete(ref_doc_id=doc_id)
        print(f"🗑️  Deleted doc_id: {doc_id}")

    def delete_all(self):
        """Hapus semua data di table"""
        if self.vector_store._table_exists():
            self.vector_store._connection.drop_table(LANCEDB_TABLE)
