from fastapi import APIRouter, Depends, HTTPException, Security
from . import utils, service, models, schemas
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_db
from sqlalchemy import select, update, delete
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi_jwt import JwtAuthorizationCredentials
from sqlalchemy.exc import IntegrityError

router = APIRouter(prefix="/api")

@router.get("/users/me")
async def users(db: AsyncSession = Depends(get_db), credentials: JwtAuthorizationCredentials = Security(utils.access_security)):
    return {"message": credentials.subject["email"]}

@router.post("/users/register", response_model=schemas.User, status_code=201)
async def register(user: schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    try:
        get_user = await db.execute(select(models.User.email).where(
            models.User.email == user.email
        ))

        is_user_exist = get_user.scalars().first()

        # if is_user_exist:
        #     raise HTTPException(status_code=400, detail="email already taken")

        hashed_pw = utils.get_password_hash(user.password)

        user_instance = models.User(full_name=user.full_name, email=user.email, password=hashed_pw)
        db.add(user_instance)
        await db.commit()
        await db.refresh(user_instance)

        return user_instance
    except IntegrityError as e:
        await db.rollback()

        print(f"register integrity error: {e}")

        if "unique" in str(e.orig).lower():
            raise HTTPException(status_code=400, detail="email already taken")
                
        raise HTTPException(status_code=400, detail="An error occurred in the data.")
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"error: {e}")

@router.post("/users/login", response_model=schemas.LoginResponse)
async def login(user: schemas.UserLogin, db: AsyncSession = Depends(get_db)):
    try:
        get_user = await db.execute(select(models.User.password).where(
            models.User.email == user.email
        ))

        user_pw = get_user.scalars().first()

        if not user_pw:
            raise HTTPException(status_code=404, detail="user with that email not found.")
    
        if utils.verify_password(user.password, user_pw):
            access_token = utils.access_security.create_access_token(subject={"email": user.email})
            refresh_token = utils.access_security.create_refresh_token(subject={"email": user.email})
        else:
            raise HTTPException(status_code=403, detail="password incorrect")

        return {"access_token": access_token, "refresh_token": refresh_token}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {e}")
