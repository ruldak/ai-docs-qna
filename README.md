# Legal Document Q&A System (FastAPI + RAG)

An advanced AI-powered Q&A system designed for analyzing and interacting with legal documents using Retrieval-Augmented Generation (RAG). This system features query expansion, re-ranking, and asynchronous document processing.

## 🚀 Key Features

- **Advanced RAG Engine**: 
    - **Query Expansion**: Generates multiple variations of user queries for better recall using Groq.
    - **Re-ranking**: Utilizes Cohere's multilingual re-ranker to improve precision.
    - **Vector Search**: High-performance similarity search powered by **Pinecone**.
- **Asynchronous Processing**: Background document ingestion and cloud storage (Supabase) via **Celery** and **Redis**.
- **Document Management**: Supports **PDF** and **DOCX** formats with automatic text extraction.
- **Multi-session Chat**: Organized conversation history with persistent storage in PostgreSQL.
- **Secure Authentication**: JWT-based authentication with password hashing (Argon2).
- **Modern Tech Stack**: Built with FastAPI, SQLModel, and LlamaIndex.

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Asynchronous)
- **AI Orchestration**: [LlamaIndex](https://www.llamaindex.ai/)
- **Vector Database**: [Pinecone](https://www.pinecone.io/)
- **LLM**: Groq (`llama-3.3-70b-versatile`)
- **Embeddings**: HuggingFace Inference API (`intfloat/multilingual-e5-large`)
- **Reranker**: Cohere Rerank (`rerank-multilingual-v3.0`)
- **Database**: PostgreSQL with [SQLModel](https://sqlmodel.tiangolo.com/) & [Alembic](https://alembic.sqlalchemy.org/)
- **Task Queue**: [Celery](https://docs.celeryq.dev/) (Redis Broker)
- **Cloud Storage**: [Supabase Storage](https://supabase.com/storage)

## 🏗️ Project Structure

```text
├── alembic/              # Database migrations
├── src/
│   ├── main.py           # Application entry point
│   ├── api.py            # APIRouter configuration
│   ├── tasks.py          # Celery task definitions (Document processing)
│   ├── database.py       # SQLAlchemy/SQLModel configuration
│   └── app/
│       ├── models.py     # SQLModel database models
│       ├── schemas.py    # Pydantic schemas (Request/Response)
│       ├── views.py      # API endpoint logic (Auth, Docs, Chat)
│       ├── pinecone.py   # Pinecone vector store manager
│       ├── utils.py      # Auth, Query Engine, and Supabase utilities
│       └── constants.py  # System constants
└── storage/              # Local storage for document indexing
```

## 📋 Prerequisites

Ensure you have the following installed:
- Python 3.10+
- PostgreSQL
- Redis (for Celery broker)
- API Keys for: **Groq**, **HuggingFace**, **Cohere**, **Pinecone**, and **Supabase**.

## ⚙️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ruldak/ai-docs-qna.git
   cd ai-docs-qna
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Copy `.env.example` to `.env` and provide your credentials:
   ```bash
   cp .env.example .env
   ```

4. **Configure Alembic Database URL**:
   Edit the `alembic.ini` file and set the SQLAlchemy URL:
   ```ini
   # sqlalchemy.url = postgresql://db_username:db_password@db_host/db_name
   sqlalchemy.url = postgresql://your_username:your_password@your_host/your_database
   ```
   Replace `your_username`, `your_password`, `your_host`, and `your_database` with your actual PostgreSQL credentials.

5. **Run Migrations**:
   ```bash
   alembic upgrade head
   ```

## 🏃‍♂️ Running the Application

1. **Start FastAPI Server**:
   ```bash
   uvicorn src.main:app --reload
   ```

2. **Start Celery Worker**:
   ```bash
   celery -A src.tasks.celery_task worker --queues=document_task --pool=threads --concurrency=3 --loglevel=info
   ```

## 📡 API Endpoints Summary

### Authentication
- `POST /api/auth/register`: Register a new account.
- `POST /api/auth/login`: Authenticate and receive JWT tokens.
- `GET /api/auth/me`: Retrieve current user profile.

### Document Management
- `GET /api/documents`: List user's documents.
- `POST /api/documents`: Upload and index a new document (triggers background task).
- `GET /api/documents/{id}`: Get document details.
- `DELETE /api/documents/{id}`: Remove document and its vector embeddings.

### Chat & Q&A
- `GET /api/sessions`: List all chat sessions.
- `POST /api/sessions`: Create a new chat session.
- `POST /api/sessions/{id}/query`: Query a specific document within a session (Advanced RAG).
- `GET /api/sessions/{id}/history`: Retrieve chat history for a session.

### Tasks
- `GET /api/tasks/{id}`: Check the status of background document processing.