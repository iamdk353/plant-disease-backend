"""AI Agent API routes — report generation and crop planning."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.routes.database import get_db
from app.schemas import (
    AiReportRequest,
    AiReportResponse,
    CropPlanRequest,
    CropPlanResponse,
    LocationItem,
)
from app.agents.orchestrator import orchestrate_ai_report, orchestrate_crop_plan

router = APIRouter(prefix="/api/ai", tags=["AI Agents"])


@router.get("/report", response_model=AiReportResponse)
async def get_ai_report(
    prediction_id: UUID,
    lat: float | None = None,
    lon: float | None = None,
    crop: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Fetch-or-generate an AI report for a prediction.

    - If an ai_report exists for prediction_id, return it.
    - Otherwise run the AI pipeline once, persist, and return it.

    Note: GET cannot reliably accept a JSON body, so location is passed as query params.
    Example: /api/ai/report?prediction_id=...&lat=12.9&lon=77.6
    """
    try:
        location = LocationItem(lat=lat, lon=lon) if (lat is not None and lon is not None) else None
        result = await orchestrate_ai_report(
            prediction_id=prediction_id,
            db=db,
            location=location,
            crop=crop,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI report fetch failed: {e}")


@router.post("/report", response_model=AiReportResponse)
async def generate_ai_report(
    body: AiReportRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a full AI report for a prediction.

    Flow:
    1. Report Agent → structured diagnosis
    2. If diseased → Web Agent (product links) + Expert Agent (analysis)
    3. Save to ai_reports table
    """
    try:
        result = await orchestrate_ai_report(
            prediction_id=body.prediction_id,
            db=db,
            location=body.location,
            crop=body.crop,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI report generation failed: {e}")


@router.post("/crop-plan", response_model=CropPlanResponse)
async def generate_crop_plan(
    body: CropPlanRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate crop recommendations based on location, weather, soil, and market trends.

    Flow:
    1. Crop Planner Agent → recommendations
    2. Save to crop_plans table
    """
    try:
        result = await orchestrate_crop_plan(
            uid=body.uid,
            location=body.location,
            db=db,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop plan generation failed: {e}")
