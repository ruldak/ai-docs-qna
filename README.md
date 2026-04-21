# Document Q&A with AI

Sistem tanya jawab cerdas berbasis AI yang dirancang khusus untuk menganalisis dan berinteraksi dengan berbagai jenis dokumen menggunakan teknik RAG (*Retrieval-Augmented Generation*).

## 🚀 Fitur Utama

- **Manajemen Dokumen**: Unggah dokumen dalam format PDF, DOCX, atau teks biasa.
- **RAG Engine**: Menggunakan LlamaIndex dan ChromaDB untuk pencarian informasi yang akurat dalam dokumen.
- **AI Agent**: Agent cerdas yang mampu memahami konteks percakapan dan menggunakan tools secara mandiri.
- **Pemrosesan Asinkron**: Indexing dokumen dilakukan di latar belakang menggunakan Celery untuk performa yang responsif.
- **Manajemen Sesi**: Mendukung riwayat percakapan yang terorganisir per sesi.
- **Keamanan**: Autentikasi pengguna menggunakan JWT (JSON Web Token).

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **AI Framework**: [LlamaIndex](https://www.llamaindex.ai/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **LLM & Embeddings**: [Groq](https://groq.com/) & [HuggingFace Inference API](https://huggingface.co/inference-api)
- **Database**: PostgreSQL with [SQLModel](https://sqlmodel.tiangolo.com/)
- **Task Queue**: [Celery](https://docs.celeryq.dev/) (dengan Redis sebagai Broker)
- **Cloud Storage**: [Supabase Storage](https://supabase.com/storage)

## 📋 Prasyarat

Sebelum memulai, pastikan Anda memiliki:
- Python 3.10+
- PostgreSQL
- Redis (untuk Celery broker)
- Akun dan API Key untuk:
  - [Groq Cloud](https://console.groq.com/)
  - [HuggingFace](https://huggingface.co/settings/tokens)
  - [Supabase](https://supabase.com/)
  - [ChromaDB Cloud](https://www.trychroma.com/) (atau setup lokal)

## ⚙️ Instalasi

1. **Clone repository**:
   ```bash
   git clone <repository-url>
   cd q&a-dokumen-hukum
   ```

2. **Buat Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # atau
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurasi Environment**:
   Buat file `.env` di direktori akar dan isi sesuai kebutuhan:
   ```env
   # Database
   DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname

   # AI Keys
   GROQ_API_KEY=your_groq_key
   HUGGINGFACE_API_KEY=your_hf_key

   # Vector Store (ChromaDB)
   API_KEY=your_chroma_key
   TENANT=your_tenant
   VECTOR_DATABASE_NAME=your_db_name
   COLLECTION_NAME=documents

   # Supabase
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   BUCKET_NAME=your_bucket_name

   # Celery
   CELERY_BROKER_URL=redis://localhost:6379/0
   CELERY_RESULT_BACKEND=redis://localhost:6379/0
   ```

## 🏃‍♂️ Menjalankan Aplikasi

1. **Jalankan Migrasi Database**:
   ```bash
   alembic upgrade head
   ```

2. **Jalankan FastAPI Server**:
   ```bash
   uvicorn src.main:app --reload
   ```
   Akses dokumentasi API di: `http://localhost:8000/docs`

3. **Jalankan Celery Worker**:
   Buka terminal baru dan jalankan:
   ```bash
   celery -A src.tasks.app worker --queues=io_task --pool=threads --loglevel=info
   ```

## 🛠️ Pengembangan
Proyek ini dibangun menggunakan **FastAPI Utilities CLI Tool** untuk memastikan struktur kode yang modular dan mudah dipelihara.

---
*Status: In Development*
