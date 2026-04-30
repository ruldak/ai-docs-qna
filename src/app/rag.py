import os
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

# Singleton for Index and Embed Model
_embed_model = None
_pipeline = None
_vector_store = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name="intfloat/multilingual-e5-large",
            token=os.getenv("HUGGING_FACE_API_KEY")
        )
    return _embed_model

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(os.getenv("COLLECTION_NAME"))
        _vector_store = ChromaVectorStore(chroma_collection=collection)
    return _vector_store

def get_ingestion_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=20),
                get_embed_model(),
            ],
            vector_store=get_vector_store(),
            # 🔥 Kunci untuk production: Docstore untuk incremental update!
            # docstore=SimpleDocumentStore(), # Mulai dengan yang sederhana dulu
            # cache=IngestionCache(cache=RedisCache.from_host_and_port(...)) # Nanti untuk scaling
        )
    return _pipeline