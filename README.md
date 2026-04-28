# AI Document Q&A System

An intelligent AI-powered Q&A system designed for analyzing and interacting with various document types using RAG (Retrieval-Augmented Generation) techniques.

## 🚀 Key Features

- **Document Management**: Support for PDF, DOCX, and plain text formats.
- **RAG Engine**: Utilizes LlamaIndex and ChromaDB for precise information retrieval.
- **AI Agent**: Intelligent agent capable of context-aware reasoning and autonomous tool usage.
- **Asynchronous Processing**: Background file uploads to cloud storage via Celery for a responsive user experience.
- **Session Management**: Organized conversation history and multi-session support.
- **Security**: Secure authentication via JWT (JSON Web Token) with role-based access control.

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Asynchronous)
- **AI Framework**: [LlamaIndex](https://www.llamaindex.ai/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (Persistent Client)
- **LLM**: Groq (`openai/gpt-oss-120b`)
- **Embeddings**: HuggingFace Inference API (`intfloat/multilingual-e5-large`)
- **Database**: PostgreSQL with [SQLModel](https://sqlmodel.tiangolo.com/) & [Alembic](https://alembic.sqlalchemy.org/)
- **Task Queue**: [Celery](https://docs.celeryq.dev/) (Redis Broker)
- **Cloud Storage**: [Supabase Storage](https://supabase.com/storage)

## 🏗️ Architecture & Data Flow

1.  **Ingestion**:
    - Documents are uploaded via the API.
    - Text is extracted using `PyMuPDF` (PDF) or `docx2txt` (Word).
    - Content is split into chunks using `SentenceSplitter` (chunk size: 512).
    - Chunks are embedded and stored in **ChromaDB**.
2.  **Background Processing**:
    - Raw files are asynchronously uploaded to **Supabase Storage** using **Celery Workers**.
3.  **Retrieval & Query**:
    - User queries trigger a *similarity search* in ChromaDB.
    - Relevant context is passed to the **Groq LLM** along with chat history.
    - The AI Agent generates responses based on the retrieved document context.

## 📂 Project Structure

```text
├── alembic/              # Database migrations
├── src/
│   ├── main.py           # Application entry point
│   ├── api.py            # APIRouter configuration
│   ├── tasks.py          # Celery task definitions
│   ├── database.py       # SQLAlchemy/SQLModel configuration
│   └── app/
│       ├── models.py     # Database models (ORM)
│       ├── schemas.py    # Pydantic schemas (Request/Response)
│       ├── views.py      # API endpoint logic
│       ├── rag.py        # LlamaIndex & AI model configuration
│       └── utils.py      # Helpers & 3rd-party integrations (Supabase, JWT)
└── chroma_db/            # Local persistent vector store
```

## 📋 Prerequisites

Before you begin, ensure you have:
- Python 3.10+
- PostgreSQL
- Redis (as Celery broker)
- API Keys for: Groq, HuggingFace, and Supabase.

## ⚙️ Installation

1. **Clone & Setup**:
   ```bash
   git clone https://github.com/ruldak/ai-docs-qna.git
   cd ai-docs-qna
   python -m venv venv
   source venv/bin/activate # or venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in your credentials.

3. **Database Migration**:
   Configure `sqlalchemy.url` in `alembic.ini`, then run:
   ```bash
   alembic upgrade head
   ```

## 🏃‍♂️ Running the Application

1. **FastAPI Server**:
   ```bash
   uvicorn src.main:app --reload
   ```

2. **Celery Worker**:
   ```bash
   celery -A src.tasks.celery_task worker --queues=io_task --pool=threads --loglevel=info
   ```

## 📡 Primary API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Register a new user |
| `POST` | `/api/auth/login` | Login and receive a JWT token |
| `GET` | `/api/auth/me` | Get current user information |
| `GET` | `/api/documents` | List all documents belonging to the user |
| `POST` | `/api/documents` | Upload and index a new document |
| `PUT` | `/api/documents/{id}` | Update document metadata or re-upload file |
| `DELETE` | `/api/documents/{id}` | Delete a document and its vector index |
| `GET` | `/api/sessions` | List all chat sessions |
| `POST` | `/api/sessions` | Create a new chat session |
| `POST` | `/api/sessions/{id}/query` | Send a query/chat in a specific session |
| `GET` | `/api/sessions/{id}/history` | Get chat history for a specific session |
| `GET` | `/api/tasks/{id}` | Check status of background tasks |
