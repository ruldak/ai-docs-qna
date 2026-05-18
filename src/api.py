"""
API Router configuration.
Aggregates all sub-routers.
"""

from fastapi import APIRouter
from src.app.views import router as app_router

router = APIRouter()
router.include_router(app_router)