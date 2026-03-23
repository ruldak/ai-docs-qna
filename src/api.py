# Define all the router here.

from fastapi import APIRouter
from src.users.views import router as users_router

router = APIRouter()
router.include_router(users_router)
