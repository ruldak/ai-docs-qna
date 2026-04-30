from pwdlib import PasswordHash
import os
from dotenv import load_dotenv
from . import constants
from datetime import timedelta

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

# ---------- tools ----------
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.core.tools import QueryEngineTool
import chromadb
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.groq import Groq
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.core import StorageContext
from . import models
from sqlalchemy import select
from .rag import get_vector_store, get_embed_model
from llama_index.postprocessor.cohere_rerank import CohereRerank

class QueryTools:
    vector_store = get_vector_store()
    embed_model = get_embed_model()

    def llm(self, temperature):
        return Groq(model="openai/gpt-oss-120b", temperature=temperature, api_key=os.getenv("GROQ_API_KEY"))

    async def query_documents(self, message: str, document_id: int):
        """Answer questions based on specific documents."""
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm(0.1)

        index = VectorStoreIndex.from_vector_store(self.vector_store, embed_model=Settings.embed_model)

        filters = MetadataFilters(
            filters=[
                ExactMatchFilter(key="doc_id", value=str(document_id))
            ]
        )

        reranker = CohereRerank(
            api_key=os.environ["COHERE_API_KEY"],
            top_n=2,
            model="rerank-multilingual-v3.0"
        )

        query_engine = index.as_query_engine(filters=filters, similarity_top_k=10, node_postprocessors=[reranker])

        result = await query_engine.aquery(message)

        return result

# ----------------- Supabase Configuration --------------------
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be filled in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)