# Legal Document Q&A System (FastAPI + RAG)

An advanced AI-powered Q&A system designed for analyzing and interacting with legal documents using Retrieval-Augmented Generation (RAG). This system features asynchronous document processing, vector search, and automated chat evaluation.

## 🚀 Key Features

- **Advanced RAG Engine**: 
    - **Vector Search**: High-performance similarity search powered by **LanceDB** (local file-based vector store).
    - **LLM-as-a-Judge Evaluation**: Background evaluation of chat responses for faithfulness and relevancy using **LlamaIndex** and **Groq**.
- **Asynchronous Processing**: Background document ingestion and evaluation via **Celery** and **Redis**.
- **Document Management**: Supports **PDF** and **DOCX** formats with cloud storage persistence in **Supabase Storage**.
- **Multi-session Chat**: Organized conversation history with persistent storage in PostgreSQL.
- **Secure Authentication**: JWT-based authentication with password hashing (Argon2).
- **Modern Tech Stack**: Built with FastAPI, SQLModel, and LlamaIndex.

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Asynchronous)
- **AI Orchestration**: [LlamaIndex](https://www.llamaindex.ai/)
- **Vector Database**: [LanceDB](https://lancedb.github.io/lancedb/) (Embedded & Serverless)
- **LLM**: Groq (`llama-3.3-70b-versatile`)
- **Embeddings**: HuggingFace Inference API (`intfloat/multilingual-e5-large`)
- **Database**: PostgreSQL with [SQLModel](https://sqlmodel.tiangolo.com/) & [Alembic](https://alembic.sqlalchemy.org/)
- **Task Queue**: [Celery](https://docs.celeryq.dev/) (Redis Broker)
- **Cloud Storage**: [Supabase Storage](https://supabase.com/storage)

## 🏗️ Project Structure

```text
├── alembic/              # Database migrations
├── lancedb_data/         # Local vector storage (LanceDB)
├── src/
│   ├── main.py           # Application entry point
│   ├── api.py            # APIRouter configuration
│   ├── tasks.py          # Celery task definitions (Document & Eval)
│   ├── database.py       # Async SQLAlchemy/SQLModel configuration
│   └── app/
│       ├── models.py            # SQLModel database models
│       ├── schemas.py           # Pydantic schemas (Request/Response)
│       ├── views.py             # API endpoint logic (Auth, Docs, Chat)
│       ├── lancedb_manager.py   # LanceDB Vector store manager
│       ├── evaluator.py         # LLM-as-a-Judge logic
│       ├── utils.py             # Auth, Query Engine, and Supabase utilities
│       └── constants.py         # System constants
└── docker-compose.yml           # Infrastructure orchestration
```

## 📋 Prerequisites

Ensure you have the following installed:
- Docker and Docker Compose
- API Keys for: **Groq**, **HuggingFace**, and **Supabase**.

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ruldak/ai-docs-qna.git
   cd ai-docs-qna
   ```

2. **Environment Variables**:
   Copy `.env.example` to `.env` and provide your credentials:
   ```bash
   cp .env.example .env
   ```
   Required keys:
   - `GROQ_API_KEY`
   - `HUGGING_FACE_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `BUCKET_NAME` (Supabase bucket)
   - `SECRET_KEY` (For JWT)

3. **Running with Docker (Recommended)**:
   The easiest way to run the entire stack (FastAPI, Postgres, Redis, Celery Workers, Migrations) is using Docker Compose:
   ```bash
   docker-compose up --build
   ```
   This will automatically:
   - Start PostgreSQL and Redis.
   - Run database migrations.
   - Start the FastAPI application on `http://localhost:8000`.
   - Start two Celery workers (one for document ingestion, one for evaluation).

## 🏃‍♂️ Manual Running (Development)

If you prefer to run services manually:

1. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Alembic (Database URL)**:
   Edit the `alembic.ini` file and set the SQLAlchemy URL to point to your local PostgreSQL instance:
   ```ini
   # sqlalchemy.url = postgresql://db_username:db_password@db_host/db_name
   sqlalchemy.url = postgresql://your_username:your_password@localhost/your_database
   ```

3. **Run Migrations**:
   ```bash
   alembic upgrade head
   ```

4. **Start FastAPI Server**:
   ```bash
   uvicorn src.main:app --reload
   ```

5. **Start Celery Workers**:
   ```bash
   # Ingest Worker
   celery -A src.tasks.celery_task worker --queues=document_task --pool=threads --concurrency=3 --loglevel=info

   # Evaluation Worker
   celery -A src.tasks.celery_task worker --queues=evaluation_task --pool=threads --concurrency=3 --loglevel=info
   ```

## 📡 API Endpoints Summary

### Authentication
- `POST /api/auth/register`: Register a new account.
- `POST /api/auth/login`: Authenticate and receive JWT tokens.

### Document Management
- `GET /api/documents`: List user's documents.
- `POST /api/documents`: Upload and index a new document (triggers background task).
- `DELETE /api/documents/{id}`: Remove document and its vector embeddings.

### Chat & Q&A
- `POST /api/sessions`: Create a new chat session.
- `POST /api/sessions/{id}/query`: Query a specific document within a session.
- `GET /api/sessions/{id}/history`: Retrieve chat history for a session.

### Tasks & Evaluation
- `GET /api/tasks/{id}`: Check document processing status.
- Evaluations are performed automatically in the background after each query.