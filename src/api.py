# Define all the router here.

from fastapi import APIRouter
from src.app.views import router as users_router

router = APIRouter()
router.include_router(users_router)
