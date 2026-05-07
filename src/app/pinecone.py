import os
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import Document, Settings
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIG
# ============================================================================
PINECONE_DIMENSION = 1024  # e5-large
NAMESPACE = os.getenv("NAMESPACE", "__default__")

# Singleton connections
_embed_model = None
_pc_client = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name="intfloat/multilingual-e5-large",
            token=os.getenv("HUGGING_FACE_API_KEY")
        )
    return _embed_model


def get_pinecone_index():
    global _pc_client
    if _pc_client is None:
        _pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if index_name not in _pc_client.list_indexes().names():
        _pc_client.create_index(
            name=index_name,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1")
            )
        )
    
    return _pc_client.Index(index_name)


# ============================================================================
# CELERY WORKER: Insert & Delete via Pinecone API Langsung
# ============================================================================

class PineconeDocumentManager:
    """
    Kelola dokumen langsung via Pinecone API.
    Tidak pakai LlamaIndex storage sama sekali!
    """
    
    def __init__(self):
        self.index = get_pinecone_index()
        self.embed_model = get_embed_model()
        self.splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        self.namespace = NAMESPACE
    
    def upsert_document(self, doc_id: str, text: str, metadata: dict = None):
        """
        Insert/update dokumen ke Pinecone.
        1. Hapus yang lama
        2. Chunk text
        3. Embed setiap chunk
        4. Upsert ke Pinecone dengan metadata text
        """
        
        # 1. Chunk text
        nodes = self.splitter.get_nodes_from_documents([Document(text=text)])
        
        # 2. Prepare vectors
        vectors = []
        for i, node in enumerate(nodes):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embed_model.get_text_embedding(node.text)
            
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": node.text,
                    "total_chunks": len(nodes),
                    **(metadata or {})
                }
            })
        
        # 3. Upsert ke Pinecone
        self.index.upsert(namespace=self.namespace, vectors=vectors)
        print(f"✅ Upserted {len(vectors)} chunks for doc_id: {doc_id}")
        return doc_id
    
    def delete_document(self, doc_id: str):
        """
        Hapus semua chunk milik doc_id dari Pinecone.
        """
        # Pinecone delete by metadata filter
        self.index.delete(
            namespace=self.namespace,
            filter={"doc_id": {"$eq": doc_id}}
        )
        print(f"🗑️  Deleted doc_id: {doc_id}")
    
    def delete_all(self):
        """Hapus semua data di namespace"""
        self.index.delete(delete_all=True, namespace=self.namespace)