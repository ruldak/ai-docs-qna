import pytest
import io
from unittest.mock import patch, MagicMock
from httpx import AsyncClient
from src.app import models

@pytest.mark.asyncio
async def test_create_document(client: AsyncClient, auth_headers, db_session):
    """Test document upload with mock Supabase and Celery."""
    
    with patch("src.app.views.process_document") as mock_celery, \
         patch("src.app.utils.supabase") as mock_supabase:
        
        mock_task = MagicMock()
        mock_task.id = "fake-task-id-123"
        mock_celery.delay.return_value = mock_task
        
        mock_storage = MagicMock()
        mock_upload_response = MagicMock()
        mock_upload_response.path = "uploads/1/test.txt"
        mock_storage.upload.return_value = mock_upload_response
        mock_supabase.storage.from_.return_value = mock_storage
        
        file_content = b"Ini adalah konten dokumen dummy untuk testing."
        files = {"file": ("test.txt", io.BytesIO(file_content), "text/plain")}
        data = {"title": "Dokumen Testing", "description": "Deskripsi dummy"}
        
        response = await client.post(
            "/api/documents",
            data=data,
            files=files,
            headers=auth_headers
        )
        
        assert response.status_code == 201, f"API Error: {response.text}"
        json_resp = response.json()
        assert json_resp["title"] == "Dokumen Testing"
        assert json_resp["task_id"] == "fake-task-id-123"
        
        mock_celery.delay.assert_called_once()
        mock_storage.upload.assert_called_once()

@pytest.mark.asyncio
async def test_get_documents(client: AsyncClient, auth_headers, db_session, test_user):
    """Test retrieving list of documents."""
    doc = models.Document(
        title="Doc 1", description="Desc 1", user_id=test_user.id, status="SUCCESS"
    )
    db_session.add(doc)
    await db_session.commit()
    
    response = await client.get("/api/documents", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Doc 1"

@pytest.mark.asyncio
async def test_delete_document(client: AsyncClient, auth_headers, db_session, test_user):
    """Test document deletion with mock LanceDB and Supabase."""
    doc = models.Document(
        title="To Delete", description="Desc", user_id=test_user.id, file_path="dummy/path"
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)
    
    with patch("src.app.views.LanceDBDocumentManager") as mock_lance, \
         patch("src.app.utils.supabase") as mock_supabase:
         
        response = await client.delete(f"/api/documents/{doc.id}", headers=auth_headers)
        
        assert response.status_code == 204
        mock_lance.return_value.delete_document.assert_called_once()