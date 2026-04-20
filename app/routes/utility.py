import tensorflow as tf
from fastapi import APIRouter, HTTPException

from app.schemas import HealthResponse
from app.state import state

router = APIRouter(tags=["Utility"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if state.model is not None else "loading",
        num_classes=len(state.idx_to_label) if state.idx_to_label else 0,
        model_load_time_s=round(state.load_time, 3) if state.load_time else None,
        gpu_available=bool(tf.config.list_physical_devices("GPU")),
    )


@router.get("/classes")
def list_classes():
    if not state.idx_to_label:
        raise HTTPException(503, "Model not loaded yet")
    return {"num_classes": len(state.idx_to_label), "classes": state.idx_to_label}