# Define all the router here.

from fastapi import APIRouter
from .users.views import router as users_router
from .documents.views import router as documents_router

router = APIRouter()
router.include_router(documents_router)
router.include_router(users_router)
