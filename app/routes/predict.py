import time

import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db_models import User, Upload, Prediction as DbPrediction, PredictionResult
from app.routes.database import get_db
from app.model import leaf_gate, preprocess_image, run_inference
from app.oci_storage import fetch_from_oci
from app.schemas import BatchPredictRequest, PredictRequest, PredictResponse
from app.state import state

router = APIRouter(prefix="/predict", tags=["Inference"])


@router.post("", response_model=PredictResponse)
async def predict(body: PredictRequest, db: AsyncSession = Depends(get_db)):
    """Fetch image from OCI by `upload_id` and `uid`, return top-K disease predictions and save to DB."""
    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")

    # Fetch User
    stmt_user = select(User).where(User.firebase_uid == body.uid)
    res_user = await db.execute(stmt_user)
    user = res_user.scalar_one_or_none()
    if not user:
        raise HTTPException(404, f"User with uid '{body.uid}' not found")

    # Fetch Upload
    stmt_upload = select(Upload).where(Upload.object_name == body.object_name, Upload.user_id == user.id)
    res_upload = await db.execute(stmt_upload)
    upload = res_upload.scalar_one_or_none()
    if not upload:
        raise HTTPException(404, f"Upload with object_name '{body.object_name}' not found for this user")

    data = fetch_from_oci(upload.object_name)

    gate = leaf_gate(data)
    if gate is not None:
        is_leaf, score = gate
        if not is_leaf:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "It's not a leaf/plant image. Please upload a clear leaf photo.",
                    "score": round(score, 4),
                },
            )

    x = preprocess_image(data)

    t0 = time.perf_counter()
    predictions = run_inference(x)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    top_preds = predictions[0]

    # Save Prediction metadata to DB
    db_pred = DbPrediction(
        upload_id=upload.id,
        user_id=user.id,
        inference_ms=elapsed_ms
    )
    db.add(db_pred)
    await db.flush()  # flush to generate db_pred.id

    # Save Prediction Results to DB
    for i, pred in enumerate(top_preds):
        db_pred_result = PredictionResult(
            prediction_id=db_pred.id,
            rank=i + 1,
            label=pred.label,
            confidence=pred.confidence
        )
        db.add(db_pred_result)

    return PredictResponse(
        object_name=upload.object_name,
        top_predictions=top_preds,
        inference_ms=round(elapsed_ms, 2),
        prediction_id=db_pred.id,
    )


@router.post("/batch", response_model=list[PredictResponse])
async def predict_batch(body: BatchPredictRequest):
    """Fetch multiple images from OCI, classify in one forward pass (max 16)."""
    if state.model is None:
        raise HTTPException(503, "Model not loaded yet")
    if not body.object_names:
        raise HTTPException(422, "object_names list is empty")
    if len(body.object_names) > 16:
        raise HTTPException(422, "Max 16 object names per batch")

    # Leaf gate check first (avoid running expensive disease model on non-leaf images)
    non_leaf: list[dict] = []
    images: list[np.ndarray] = []

    for name in body.object_names:
        data = fetch_from_oci(name)
        gate = leaf_gate(data)
        if gate is not None:
            is_leaf, score = gate
            if not is_leaf:
                non_leaf.append({"object_name": name, "score": round(score, 4)})
                continue

        images.append(preprocess_image(data)[0])

    if non_leaf:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Some images are not leaf/plant. Upload only leaf images.",
                "non_leaf": non_leaf,
            },
        )

    x = np.stack(images, axis=0)

    t0 = time.perf_counter()
    all_predictions = run_inference(x)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    per_image_ms = round(elapsed_ms / len(body.object_names), 2)

    return [
        PredictResponse(
            object_name=name,
            top_predictions=preds,
            inference_ms=per_image_ms,
        )
        for name, preds in zip(body.object_names, all_predictions)
    ]