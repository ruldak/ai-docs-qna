from pwdlib import PasswordHash
import os
from dotenv import load_dotenv
from . import constants
from datetime import timedelta
import asyncio
import re

load_dotenv()

password_hash = PasswordHash.recommended() # Gunakan Argon2

def get_password_hash(password: str) -> str:
    """Hash password saat registrasi."""
    return password_hash.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Cocokkan password saat login."""
    return password_hash.verify(plain_password, hashed_password)


from fastapi_jwt import JwtAuthorizationCredentials, JwtAccessBearer, JwtRefreshBearer

secret_key = os.getenv("SECRET_KEY")
access_security = JwtAccessBearer(secret_key=secret_key, auto_error=True, access_expires_delta=timedelta(minutes=60))
refresh_security = JwtRefreshBearer(secret_key=secret_key, auto_error=True, refresh_expires_delta=timedelta(days=7))

# ---------- Query Engine ----------
import cohere
from typing import List
from .lancedb_manager import get_vector_store, get_embed_model
from llama_index.llms.groq import Groq
from typing import List, Dict, Optional
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding

class LlamaIndexQueryEngine:
    """
    LlamaIndex query engine dengan LanceDB sebagai storage.
    Tidak ada mapping di RAM — selalu fetch fresh dari LanceDB.
    """

    def __init__(self):
        self.vector_store = get_vector_store()
        self.embed_model = get_embed_model()

        # LLM
        self.llm = Groq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0
        )

        Settings.llm = self.llm

        # Cohere
        self.api_key = os.getenv("COHERE_API_KEY")
        self.client = cohere.Client(self.api_key)

    async def query(self, query: str, document_id: int, top_k: int = 5) -> str:
        """
        Query via LlamaIndex retriever → Cohere rerank.
        """
        # Build index dari vector store (tidak di RAM, fetch on query)
        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )

        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="doc_id", value=str(document_id))]
        )

        # Retriever — fetch dari LanceDB
        query_engine = index.as_query_engine(
            filters=filters,
            similarity_top_k=top_k * 3,
            response_mode="tree_summarize",
            verbose=True
        )

        try:
            response = await query_engine.aquery(query)
            nodes = response.source_nodes or []
        except Warning as w:
            print(f"[LanceDB] empty result for doc_id={document_id}: {w}")
            nodes = []
        except Exception as e:
            print(f"[Query] error: {e}")
            nodes = []

        if not nodes:
            return "Empty response"

        return response.response

def get_query_engine():
    return LlamaIndexQueryEngine()

# ----------------- Supabase Configuration --------------------
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be filled in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
