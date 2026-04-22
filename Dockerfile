# Gunakan image Python yang ringan
FROM python:3.11-slim

# Set environment variables untuk Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set direktori kerja
WORKDIR /app

# Install dependensi sistem yang diperlukan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode proyek
COPY . .

# Expose port untuk FastAPI
EXPOSE 8000

# Command default
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
