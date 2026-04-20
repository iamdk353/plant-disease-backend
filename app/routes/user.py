from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db_models import User
from app.routes.database import get_db
from app.schemas import RegisterUserRequest, UserResponse

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=UserResponse)
async def register_user(body: RegisterUserRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user or return existing user based on firebase_uid."""
    stmt = select(User).where(User.firebase_uid == body.firebase_uid)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user:
        # User already exists, just return it
        return user

    new_user = User(
        firebase_uid=body.firebase_uid,
        email=body.email,
    )
    db.add(new_user)
    
    try:
        # commit to db because get_db does not automatically commit on exit unless explicitly told to in some cases, 
        # wait get_db DOES commit. But we need to refresh to get the ID.
        await db.flush()
        await db.refresh(new_user)
        return new_user
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not register user: {e}")
