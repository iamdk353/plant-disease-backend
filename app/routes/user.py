from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db_models import User
from app.routes.database import get_db
from app.routes.upload import upload_image_for_user
from app.schemas import RegisterUserRequest, UserResponse, UserProfileResponse

router = APIRouter(prefix="/auth", tags=["Auth"])
users_router = APIRouter(prefix="/users", tags=["Users"])


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.strip()
    return value or None


def _parse_primary_crops(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None

    raw_items = value if isinstance(value, list) else value.split(",")
    crops = [item.strip() for item in raw_items if item and item.strip()]
    return crops or None


def _get_form_value(form: Any, *keys: str) -> str | None:
    for key in keys:
        if key not in form:
            continue

        value = form.get(key)
        if isinstance(value, UploadFile):
            continue

        return str(value)

    return None


def _parse_form_number(
    form: Any,
    parser: Callable[[str], int | float],
    field_name: str,
    *keys: str,
) -> int | float | None:
    raw_value = _get_form_value(form, *keys)
    if raw_value is None:
        return None

    raw_value = raw_value.strip()
    if not raw_value:
        return None

    try:
        return parser(raw_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a valid {parser.__name__}",
        ) from exc


def _build_profile_response(user: User, request: Request) -> UserProfileResponse:
    photo_url = None
    if user.photo_object_name:
        photo_url = str(request.url_for("get_activity_image", object_name=user.photo_object_name))

    return UserProfileResponse(
        id=user.id,
        firebase_uid=user.firebase_uid,
        email=user.email,
        name=user.name,
        photo_object_name=user.photo_object_name,
        photo_url=photo_url,
        phone_number=user.phone_number,
        years_of_experience=user.years_of_experience,
        acres=user.acres,
        primary_crops=user.primary_crops,
        soil_type=user.soil_type,
    )


@router.post("/register", response_model=UserResponse)
async def register_user(body: RegisterUserRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user or return existing user based on firebase_uid."""
    stmt = select(User).where(User.firebase_uid == body.firebase_uid)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    normalized_email = _normalize_optional_text(body.email)
    normalized_name = _normalize_optional_text(body.name)
    normalized_phone_number = _normalize_optional_text(body.phone_number)
    normalized_soil_type = _normalize_optional_text(body.soil_type)
    crops_list = _parse_primary_crops(body.primary_crops)

    if user:
        updated = False

        if "email" in body.model_fields_set:
            if normalized_email and normalized_email != user.email:
                email_stmt = select(User).where(User.email == normalized_email)
                email_result = await db.execute(email_stmt)
                email_user = email_result.scalar_one_or_none()
                if email_user and email_user.id != user.id:
                    raise HTTPException(status_code=409, detail="Email already registered.")
            user.email = normalized_email
            updated = True

        if "name" in body.model_fields_set:
            user.name = normalized_name
            updated = True
        if "phone_number" in body.model_fields_set:
            user.phone_number = normalized_phone_number
            updated = True
        if "years_of_experience" in body.model_fields_set:
            user.years_of_experience = body.years_of_experience
            updated = True
        if "acres" in body.model_fields_set:
            user.acres = body.acres
            updated = True
        if "primary_crops" in body.model_fields_set:
            user.primary_crops = crops_list
            updated = True
        if "soil_type" in body.model_fields_set:
            user.soil_type = normalized_soil_type
            updated = True

        if updated:
            await db.flush()
            await db.refresh(user)

        return user

    if normalized_email:
        email_stmt = select(User).where(User.email == normalized_email)
        email_result = await db.execute(email_stmt)
        email_user = email_result.scalar_one_or_none()
        if email_user:
            raise HTTPException(status_code=409, detail="Email already registered.")

    new_user = User(
        firebase_uid=body.firebase_uid,
        email=normalized_email,
        name=normalized_name,
        phone_number=normalized_phone_number,
        years_of_experience=body.years_of_experience,
        acres=body.acres,
        primary_crops=crops_list,
        soil_type=normalized_soil_type,
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


@users_router.post("/", response_model=UserProfileResponse)
async def create_user_profile(
    request: Request,
    uid: str | None = Form(None, description="Firebase UID of the user"),
    name: str | None = Form(None, description="User display name"),
    email: str | None = Form(None, description="User email"),
    phone_number: str | None = Form(None, description="User phone number"),
    years_of_experience: int | None = Form(None, description="Years of farming experience"),
    acres: float | None = Form(None, description="Farm size in acres"),
    primary_crops: str | None = Form(None, description="Primary crops (comma-separated)"),
    soil_type: str | None = Form(None, description="Soil type"),
    photo: UploadFile | None = File(None, description="Profile photo to upload"),
    db: AsyncSession = Depends(get_db),
):
    form = await request.form()
    uid = _normalize_optional_text(uid) or _normalize_optional_text(
        _get_form_value(form, "uid", "firebase_uid", "firebaseUid")
    )
    name = _normalize_optional_text(name) if name is not None else _normalize_optional_text(
        _get_form_value(form, "name")
    )
    email = _normalize_optional_text(email) if email is not None else _normalize_optional_text(
        _get_form_value(form, "email")
    )
    phone_number = (
        _normalize_optional_text(phone_number)
        if phone_number is not None
        else _normalize_optional_text(_get_form_value(form, "phone_number", "phoneNumber"))
    )
    years_of_experience = (
        years_of_experience
        if years_of_experience is not None
        else _parse_form_number(
            form,
            int,
            "years_of_experience",
            "years_of_experience",
            "yearsOfExperience",
            "yearsOfExp",
        )
    )
    acres = (
        acres
        if acres is not None
        else _parse_form_number(form, float, "acres", "acres")
    )
    primary_crops = (
        primary_crops
        if primary_crops is not None
        else _get_form_value(form, "primary_crops", "primaryCrops")
    )
    soil_type = _normalize_optional_text(soil_type) if soil_type is not None else _normalize_optional_text(
        _get_form_value(form, "soil_type", "soilType")
    )

    if not uid or not uid.strip():
        raise HTTPException(status_code=400, detail="uid parameter is mandatory and cannot be empty")

    stmt = select(User).where(User.firebase_uid == uid)
    result = await db.execute(stmt)
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=409, detail="User already exists. Use PATCH to update profile.")

    normalized_email = email
    if normalized_email:
        email_stmt = select(User).where(User.email == normalized_email)
        email_result = await db.execute(email_stmt)
        email_user = email_result.scalar_one_or_none()
        if email_user:
            raise HTTPException(status_code=409, detail="Email already registered.")

    crops_list = _parse_primary_crops(primary_crops)

    user = User(
        firebase_uid=uid,
        email=normalized_email,
        name=name,
        phone_number=phone_number,
        years_of_experience=years_of_experience,
        acres=acres,
        primary_crops=crops_list,
        soil_type=soil_type,
    )
    db.add(user)
    await db.flush()

    if photo is not None:
        object_name, _ = await upload_image_for_user(uid=uid, file=photo, db=db, user=user)
        user.photo_object_name = object_name

    await db.flush()
    await db.refresh(user)
    return _build_profile_response(user, request)


@users_router.get("/{uid}", response_model=UserProfileResponse)
async def get_user_profile(
    uid: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    if not uid or not uid.strip():
        raise HTTPException(status_code=400, detail="uid parameter is mandatory and cannot be empty")

    stmt = select(User).where(User.firebase_uid == uid)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return _build_profile_response(user, request)


@users_router.patch("/{uid}", response_model=UserProfileResponse)
async def update_user_profile(
    uid: str,
    request: Request,
    name: str | None = Form(None, description="User display name"),
    email: str | None = Form(None, description="User email"),
    phone_number: str | None = Form(None, description="User phone number"),
    years_of_experience: int | None = Form(None, description="Years of farming experience"),
    acres: float | None = Form(None, description="Farm size in acres"),
    primary_crops: str | None = Form(None, description="Primary crops (comma-separated)"),
    soil_type: str | None = Form(None, description="Soil type"),
    photo: UploadFile | None = File(None, description="Profile photo to upload"),
    db: AsyncSession = Depends(get_db),
):
    form = await request.form()
    has_name = name is not None or "name" in form
    has_email = email is not None or "email" in form
    has_phone_number = (
        phone_number is not None or "phone_number" in form or "phoneNumber" in form
    )
    has_years_of_experience = (
        years_of_experience is not None
        or "years_of_experience" in form
        or "yearsOfExperience" in form
        or "yearsOfExp" in form
    )
    has_acres = acres is not None or "acres" in form
    has_primary_crops = (
        primary_crops is not None or "primary_crops" in form or "primaryCrops" in form
    )
    has_soil_type = soil_type is not None or "soil_type" in form or "soilType" in form

    name = _normalize_optional_text(name) if name is not None else _normalize_optional_text(
        _get_form_value(form, "name")
    )
    email = _normalize_optional_text(email) if email is not None else _normalize_optional_text(
        _get_form_value(form, "email")
    )
    phone_number = (
        _normalize_optional_text(phone_number)
        if phone_number is not None
        else _normalize_optional_text(_get_form_value(form, "phone_number", "phoneNumber"))
    )
    years_of_experience = (
        years_of_experience
        if years_of_experience is not None
        else _parse_form_number(
            form,
            int,
            "years_of_experience",
            "years_of_experience",
            "yearsOfExperience",
            "yearsOfExp",
        )
    )
    acres = (
        acres
        if acres is not None
        else _parse_form_number(form, float, "acres", "acres")
    )
    primary_crops = (
        primary_crops
        if primary_crops is not None
        else _get_form_value(form, "primary_crops", "primaryCrops")
    )
    soil_type = _normalize_optional_text(soil_type) if soil_type is not None else _normalize_optional_text(
        _get_form_value(form, "soil_type", "soilType")
    )

    if not uid or not uid.strip():
        raise HTTPException(status_code=400, detail="uid parameter is mandatory and cannot be empty")

    stmt = select(User).where(User.firebase_uid == uid)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    updated = False
    if has_name:
        user.name = name
        updated = True
    if has_email:
        if email and email != user.email:
            email_stmt = select(User).where(User.email == email)
            email_result = await db.execute(email_stmt)
            email_user = email_result.scalar_one_or_none()
            if email_user and email_user.id != user.id:
                raise HTTPException(status_code=409, detail="Email already registered.")
        user.email = email
        updated = True
    if has_phone_number:
        user.phone_number = phone_number
        updated = True
    if has_years_of_experience:
        user.years_of_experience = years_of_experience
        updated = True
    if has_acres:
        user.acres = acres
        updated = True
    if has_primary_crops:
        user.primary_crops = _parse_primary_crops(primary_crops)
        updated = True
    if has_soil_type:
        user.soil_type = soil_type
        updated = True
    if photo is not None:
        object_name, _ = await upload_image_for_user(uid=uid, file=photo, db=db, user=user)
        user.photo_object_name = object_name
        updated = True

    if not updated:
        raise HTTPException(status_code=400, detail="No update fields provided")

    await db.flush()
    await db.refresh(user)
    return _build_profile_response(user, request)
