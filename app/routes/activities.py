from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
import mimetypes

from app.oci_storage import fetch_from_oci

from app.db_models import User, Prediction
from app.routes.database import get_db
from app.schemas import ActivityResponse

router = APIRouter(prefix="/activities", tags=["Activities"])

@router.get("/image/{object_name}")
async def get_activity_image(object_name: str):
    """Fetch an image for an activity directly from OCI"""
    try:
        data = fetch_from_oci(object_name)
        mt, _ = mimetypes.guess_type(object_name)
        return Response(
            content=data,
            media_type=mt or "image/jpeg",
            headers={"ngrok-skip-browser-warning": "true"}  # ← add this
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail="Image not found")


@router.get("/", response_model=list[ActivityResponse])
async def get_user_activities(
    request: Request,
    uid: str = Query(..., description="Firebase UID of the user"),
    db: AsyncSession = Depends(get_db)
):
    """Fetch all prediction activities for a specific user"""
    # 1. Verify user exists
    user_stmt = select(User).where(User.firebase_uid == uid)
    user_res = await db.execute(user_stmt)
    user = user_res.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # 2. Fetch predictions with related upload and results
    pred_stmt = (
        select(Prediction)
        .options(
            selectinload(Prediction.upload),
            selectinload(Prediction.prediction_results)
        )
        .where(Prediction.user_id == user.id)
        .order_by(Prediction.created_at.desc())
    )
    
    pred_res = await db.execute(pred_stmt)
    predictions = pred_res.scalars().all()
    
    # 3. Format response
    activities = []
    for p in predictions:
        # Sort results by rank
        results = sorted(p.prediction_results, key=lambda r: r.rank)
        
        # Handle case where upload might be missing somehow
        image_name = p.upload.object_name if p.upload else "unknown_image"
        
        activities.append(
            ActivityResponse(
                id=p.id,
                image_name=image_name,
                image_url=str(request.url_for("get_activity_image", object_name=image_name)),
                inference_ms=p.inference_ms or 0.0,
                created_at=p.created_at,
                results=[
                    {
                        "rank": r.rank,
                        "label": r.label,
                        "confidence": r.confidence
                    } for r in results
                ]
            )
        )
        
    return activities
