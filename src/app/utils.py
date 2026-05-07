from pwdlib import PasswordHash
import os
from dotenv import load_dotenv
from . import constants
from datetime import timedelta
import asyncio

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
from .pinecone import get_pinecone_index, get_embed_model, NAMESPACE
from llama_index.llms.groq import Groq

class PineconeQueryEngine:
    """
    LlamaIndex hanya untuk query & LLM orchestration.
    Data selalu fresh dari Pinecone!
    """
    
    def __init__(self):
        self.index = get_pinecone_index()
        self.embed_model = get_embed_model()
        self.llm = Groq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.namespace = NAMESPACE

        self.api_key = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY required")
        
        # Cohere Python SDK
        try:
            self.client = cohere.Client(self.api_key)
        except ImportError:
            raise ImportError("pip install cohere")
    
    async def query(self, query: str, document_id: int, top_k_per_query: int = 5) -> dict:
        queries = await asyncio.to_thread(
            self.expand,
            query
        )

        print(f"🔍 Query variations: {queries}")
        
        # 2. Search setiap variation
        all_results = []
        for q in queries:
            embedding = await self.embed_model._aget_text_embedding(q)
            
            results = self.index.query(
                namespace=self.namespace,
                vector=embedding,
                top_k=top_k_per_query,
                filter={"doc_id": str(document_id)},
                include_metadata=True
            )
            
            all_results.append(results.matches)
        
        if not all_results:
            return {"answer": "Tidak menemukan informasi relevan.", "sources": []}
        
        # Extract text dari metadata
        contexts = []
        sources = []
        for r in all_results:
            text = r[0]["metadata"].get("text", "")
            contexts.append(text)
            sources.append({
                "doc_id": r[0]["metadata"].get("doc_id"),
                "score": r[0]["score"],
                "text_preview": text[:200] + "..." if len(text) > 200 else text
            })


        results = self.rerank(query=query, documents=contexts)
        return {
            "contexts": results,
            "sources": sources
        }

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 3,
        model: str = "rerank-multilingual-v3.0"
    ) -> List[dict]:
        """
        Rerank dokumen dengan Cohere.
        
        Args:
            query: Pertanyaan user
            documents: List text dokumen (dari Pinecone)
            top_n: Berapa hasil terbaik yang diambil
            model: Cohere rerank model
        
        Returns:
            List hasil rerank dengan relevance_score
        """
        if not documents:
            return []
        
        response = self.client.rerank(
            model=model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            return_documents=True
        )
        
        results = []
        for r in response.results:
            results.append({
                "text": r.document.text,
                "index": r.index,  # index asli di input documents
                "relevance_score": r.relevance_score,
            })
        
        return results

    def expand(self, query: str) -> List[str]:
        """
        Generate multiple query variations untuk better recall.
        
        Returns: List query strings untuk di-embed dan search
        """
        # Prompt untuk expand query
        prompt = f"""Anda adalah asisten pencarian dokumen hukum Indonesia.
Buat 3 variasi pertanyaan berikut untuk pencarian yang lebih baik.
Variasi harus mencakup sinonim, istilah formal, dan nomor pasal/ayat jika ada.

Pertanyaan asli: {query}

Format output (satu per baris):
1. [variasi 1]
2. [variasi 2]
3. [variasi 3]

Jangan tambahkan penjelasan, hanya list variasi."""

        response = self.llm.complete(prompt)
        text = str(response)
        
        # Parse hasil
        variations = [query]  # Original query selalu include
        
        for line in text.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                cleaned = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                if cleaned and cleaned != query:
                    variations.append(cleaned)
        
        # Deduplicate and limit
        seen = set()
        unique = []
        for v in variations:
            if v.lower() not in seen:
                seen.add(v.lower())
                unique.append(v)
        
        return unique[:4]

# ----------------- Supabase Configuration --------------------
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be filled in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)