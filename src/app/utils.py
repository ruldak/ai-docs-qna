"""
Utility functions: Auth, Query Engine, Supabase client.
"""

import os
import cohere
from datetime import timedelta
from typing import List, Tuple, Optional
from pathlib import Path

from fastapi_jwt import JwtAuthorizationCredentials, JwtAccessBearer, JwtRefreshBearer
from pwdlib import PasswordHash

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from supabase import create_client, Client

from src.constants import settings
from src.app.lancedb_manager import create_vector_store, get_embed_model
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PASSWORD HASHING
# ============================================================================
password_hasher = PasswordHash.recommended()  # Argon2


def get_password_hash(password: str) -> str:
    """Hash password saat registrasi."""
    return password_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifikasi password saat login."""
    return password_hasher.verify(plain_password, hashed_password)


# ============================================================================
# JWT AUTH
# ============================================================================
access_security = JwtAccessBearer(
    secret_key=settings.secret_key,
    auto_error=True,
    access_expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
)

refresh_security = JwtRefreshBearer(
    secret_key=settings.secret_key,
    auto_error=True,
    refresh_expires_delta=timedelta(days=settings.refresh_token_expire_days)
)


# ============================================================================
# QUERY ENGINE
# ============================================================================

class LlamaIndexQueryEngine:
    """
    Query engine untuk RAG menggunakan LlamaIndex + LanceDB.
    """

    def __init__(self):
        self.embed_model = get_embed_model()

        self.llm = Groq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature
        )

        Settings.llm = self.llm

        # Cohere client untuk reranking (jika diperlukan nanti)
        self.cohere_client = cohere.Client(settings.cohere_api_key)

    async def query(self, query: str, document_id: int, top_k: int = 5) -> str:
        """
        Query via LlamaIndex retriever. Return string jawaban saja.

        Args:
            query: Pertanyaan user
            document_id: ID dokumen untuk filter
            top_k: Jumlah top results

        Returns:
            String jawaban
        """
        response = await self._execute_query(query, document_id, top_k)
        if response is None:
            return "Maaf, tidak dapat menemukan informasi yang relevan untuk pertanyaan Anda."
        return response.response

    async def query_with_sources(
        self, 
        query: str, 
        document_id: int, 
        top_k: int = 5
    ) -> Tuple[str, List[str]]:
        """
        Query + return (jawaban, list_contexts).
        Contexts diambil dari source_nodes LlamaIndex.

        Args:
            query: Pertanyaan user
            document_id: ID dokumen untuk filter
            top_k: Jumlah top results

        Returns:
            Tuple (jawaban, list konteks)
        """
        response = await self._execute_query(query, document_id, top_k)

        if response is None:
            return (
                "Maaf, tidak dapat menemukan informasi yang relevan untuk pertanyaan Anda.",
                []
            )

        # Ekstrak teks dari source nodes sebagai contexts
        contexts = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for node in response.source_nodes:
                contexts.append(node.get_content())

        return response.response, contexts

    async def _execute_query(self, query: str, document_id: int, top_k: int = 5):
        """
        Internal: execute query engine dan return response object.

        Args:
            query: Pertanyaan user
            document_id: ID dokumen untuk filter
            top_k: Jumlah top results

        Returns:
            Response object atau None jika error/empty
        """
        vector_store = create_vector_store()

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model
        )

        # FIX KRITIS: Gunakan "postgres_id" bukan "doc_id"
        # Karena di lancedb_manager.py metadata key adalah "postgres_id"
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="postgres_id", value=str(document_id))]
        )

        query_engine = index.as_query_engine(
            filters=filters,
            similarity_top_k=top_k * 3,  # Ambil lebih banyak untuk diversity
            response_mode="tree_summarize",
            verbose=True
        )

        try:
            response = await query_engine.aquery(query)
            return response
        except Exception as e:
            logger.warning(f"Query error for doc_id={document_id}: {e}")
            return None


# Singleton query engine
_query_engine: Optional[LlamaIndexQueryEngine] = None


def get_query_engine() -> LlamaIndexQueryEngine:
    """Get or create singleton query engine."""
    global _query_engine
    if _query_engine is None:
        _query_engine = LlamaIndexQueryEngine()
    return _query_engine


# ============================================================================
# SUPABASE CLIENT
# ============================================================================

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Get or create singleton Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key
        )
    return _supabase_client


# Legacy compatibility
supabase = get_supabase_client()


# ============================================================================
# SYSTEM PROMPT LOADER
# ============================================================================

def load_system_prompt(document_id: int) -> str:
    """
    Load system prompt dari file dan format dengan document_id.

    Args:
        document_id: ID dokumen yang sedang di-query

    Returns:
        Formatted system prompt string
    """
    prompt_file_path = Path(__file__).parent / "system_prompt.txt"

    if not prompt_file_path.exists():
        # Fallback prompt jika file tidak ada
        return f"""Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen.
Dokumen ID: {document_id}

Instruksi:
1. Jawab berdasarkan informasi dari dokumen yang diberikan
2. Jika informasi tidak cukup, jelaskan dengan jujur
3. Gunakan Bahasa Indonesia yang baik dan benar
4. Berikan jawaban yang detail dan komprehensif
"""

    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    return prompt_template.format(document_id=document_id)