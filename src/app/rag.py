import os
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()

# Singleton untuk Index dan Embed Model
_index = None
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name="intfloat/multilingual-e5-large",
            token=os.getenv("HUGGING_FACE_API_KEY")
        )
    return _embed_model

def get_index():
    global _index
    if _index is None:
        # Setup Chroma
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(os.getenv("COLLECTION_NAME"))
        vector_store = ChromaVectorStore(chroma_collection=collection)
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Set global embed model agar tidak perlu passing terus menerus
        Settings.embed_model = get_embed_model()
        Settings.transformations = [SentenceSplitter(chunk_size=512)]
        Settings.llm = Groq(model="openai/gpt-oss-120b", temperature=0.2, api_key=os.getenv("GROQ_API_KEY"))

        # Load index dari vector store yang sudah ada
        _index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context,
            embed_model=Settings.embed_model,
        )
    return _index