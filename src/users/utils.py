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

# ---------- ReActAgent tools ----------
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.core.tools import QueryEngineTool
import chromadb
from llama_index.llms.openrouter import OpenRouter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.groq import Groq
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from . import models
from sqlalchemy import select

class QueryTools:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.tenant = os.getenv("TENANT")
        self.database = "fastapi_qna_legal_documents"
        self.open_router_api_key = os.getenv("OPEN_ROUTER_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGING_FACE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

    def llm(self, temperature):
        return Groq(model="openai/gpt-oss-120b", temperature=temperature, api_key=self.groq_api_key)

    def client(self):
        return chromadb.CloudClient(
            api_key=self.api_key,
            tenant=self.tenant,
            database=self.database
        )

    async def query_documents(self, message: str, document_id: int):
        """Answer questions based on specific documents."""
        client = self.client()
        Settings.llm = self.llm(0.2)

        embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name="intfloat/multilingual-e5-large",
            token=self.huggingface_api_key
        )

        collection = client.get_or_create_collection("legal_documents")
        vstore = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vstore, embed_model=embed_model)

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="id", value=document_id, operator=FilterOperator.EQ)
            ]
        )

        query_engine = index.as_query_engine(filters=filters, similarity_top_k=2)

        response = await query_engine.aquery(message)

        return response