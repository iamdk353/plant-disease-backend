import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.config import OCI_BUCKET
from app.oci_storage import get_oci_client
from app.schemas import UploadResponse, UserUploadsResponse
from app.routes.database import get_db
from app.db_models import User, Upload as DbUpload

router = APIRouter(tags=["Upload"])


async def upload_image_for_user(
    *,
    uid: str,
    file: UploadFile,
    db: AsyncSession,
    user: User | None = None,
) -> tuple[str, str]:
    if not uid or not uid.strip():
        raise HTTPException(status_code=400, detail="uid parameter is mandatory and cannot be empty")

    if not file:
        raise HTTPException(status_code=400, detail="file parameter is mandatory")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

    if user is None:
        stmt = select(User).where(User.firebase_uid == uid)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail=f"User with uid '{uid}' not found.")

    try:
        client, namespace = get_oci_client()
        file_content = await file.read()

        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        ext = Path(file.filename).suffix.lower() if file.filename else ""
        object_name = f"{uid}_{uuid.uuid4()}{ext}"

        client.put_object(
            namespace,
            OCI_BUCKET,
            object_name,
            file_content,
            content_type=file.content_type,
        )

        new_upload = DbUpload(
            user_id=user.id,
            object_name=object_name,
            bucket=OCI_BUCKET,
            original_filename=file.filename,
            content_type=file.content_type,
            file_size_bytes=len(file_content),
            status="uploaded",
        )
        db.add(new_upload)

        return object_name, OCI_BUCKET
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCI Upload failed: {exc}")


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    uid: str = Form(..., description="User ID associated with the upload required to trace the image"),
    file: UploadFile = File(..., description="Plant image to upload to OCI"),
    db: AsyncSession = Depends(get_db)
):
    """Upload image to OCI bucket and save metadata to database. Returns `object_name` to pass to `/predict`."""
    object_name, bucket = await upload_image_for_user(uid=uid, file=file, db=db)

    return UploadResponse(
        status="success",
        object_name=object_name,
        bucket=bucket,
        message=f"Successfully uploaded {object_name} to {bucket}",
    )


@router.get("/uploads/{uid}", response_model=UserUploadsResponse)
async def get_user_uploads(uid: str):
    """Fetch all images uploaded by a specific user from OCI bucket."""
    if not uid or not uid.strip():
        raise HTTPException(status_code=400, detail="uid parameter is mandatory and cannot be empty")
        
    try:
        client, namespace = get_oci_client()
        
        # Adding an underscore to ensure we don't match "user1" prefix when querying "user12"
        prefix = f"{uid}_"
        
        response = client.list_objects(
            namespace_name=namespace,
            bucket_name=OCI_BUCKET,
            prefix=prefix,
        )
        
        object_names = [obj.name for obj in response.data.objects] if response.data.objects else []
        
        return UserUploadsResponse(
            uid=uid,
            images=object_names
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch uploads for user: {exc}")
