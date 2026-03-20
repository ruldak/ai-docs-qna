from fastapi import APIRouter, Depends, HTTPException
from . import utils, service, models, schemas
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_db
from sqlalchemy import select, update, delete

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/auth/me")
async def users(db: AsyncSession = Depends(get_db)):
    return {"message": "Hello Users"}

@router.post("/auth/register")
async def register(user: schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    try:
        get_user = await db.execute(select(models.User.email).where(
            models.User.email == user.email
        ))

        print("==========")
        print(f"get_user: {get_user}")
        print("==========")

        is_user_exist = get_user.scalars().first()

        return {"user_exist": is_user_exist}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")

@router.post("/auth/login")
async def login():
    return {"message": "Hello Login"}
