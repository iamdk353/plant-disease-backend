"""Orchestrator — coordinates all agents and persists results to DB."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
import asyncio
from uuid import UUID

from app.db_models import Prediction, PredictionResult, User, AiReport, CropPlan
from app.schemas import (
    AiReportResponse,
    CropPlanResponse,
    ExpertAgentOutput,
    ReportAgentOutput,
    WebAgentOutput,
)

from app.agents.report_agent import generate_report
from app.agents.web_agent import build_search_fallback_url, fetch_info_links, fetch_product_links
from app.agents.expert_agent import analyze_crop_condition
from app.agents.crop_planner_agent import plan_crop
from app.schemas import LocationItem
from app.config import WEATHER_API_KEY
import httpx

logger = logging.getLogger(__name__)

async def get_current_weather(location: LocationItem | str | None) -> tuple[str, str]:
    """Fetch live weather and location name from OpenWeatherMap using explicit coords or a city string."""
    if not location or not WEATHER_API_KEY:
        return "Not provided", "Unknown location"
    
    if isinstance(location, str):
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
    else:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={location.lat}&lon={location.lon}&appid={WEATHER_API_KEY}&units=metric"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                desc = data["weather"][0]["description"]
                temp = data["main"]["temp"]
                humid = data["main"]["humidity"]
                loc_name = data.get("name", "Unknown location")
                return f"{temp}°C, {desc}, {humid}% humidity", loc_name
            else:
                logger.warning(f"Weather fetch failed: {resp.status_code}")
                return "Failed to fetch weather", "Unknown location"
    except Exception as e:
        logger.warning(f"Weather fetch exception: {e}")
        return "Failed to fetch weather", "Unknown location"


async def orchestrate_ai_report(
    prediction_id: UUID,
    db: AsyncSession,
    location: LocationItem | None = None,
    crop: str | None = None,
) -> AiReportResponse:
    """
    Full AI report pipeline:
    1. Fetch prediction from DB
    2. Run Report Agent
    3. If diseased → run Web Agent + Expert Agent
    4. Merge and save to ai_reports table
    """

    # ── 1. Fetch prediction + results from DB ──────────────────────
    stmt = (
        select(Prediction)
        .options(
            selectinload(Prediction.prediction_results),
            selectinload(Prediction.upload),
        )
        .where(Prediction.id == prediction_id)
    )
    result = await db.execute(stmt)
    prediction = result.scalar_one_or_none()

    if not prediction:
        raise ValueError(f"Prediction {prediction_id} not found")

    # Check if report already exists
    existing_stmt = select(AiReport).where(AiReport.prediction_id == prediction_id)
    existing_result = await db.execute(existing_stmt)
    existing_report = existing_result.scalar_one_or_none()

    if existing_report:
        # If links are mandatory and missing, backfill once.
        # We decide "healthy vs diseased" deterministically from the model label
        # (LLM can occasionally mis-set is_diseased).
        if not (existing_report.product_links or []):
            sorted_results = sorted(prediction.prediction_results, key=lambda r: r.rank)
            top = sorted_results[0] if sorted_results else None
            label = top.label if top else ""

            label_is_healthy = "healthy" in label.lower()

            # Only backfill treatment links when the prediction label is NOT healthy.
            if label and (not label_is_healthy):
                # Bring cached flags in sync with deterministic label.
                if existing_report.is_diseased is not True:
                    existing_report.is_diseased = True

                crop_name = crop
                if not crop_name:
                    parts = label.split("__")
                    crop_name = parts[0].replace("_", " ") if parts else None

                disease_name = existing_report.disease_name or label or "plant disease"

                links: list[str] = []
                try:
                    web_result = await asyncio.to_thread(fetch_product_links, disease_name)
                    links = web_result.product_links or []
                except Exception as e:
                    logger.warning(f"[Web Agent] Product link backfill failed: {e}")

                if not links:
                    try:
                        links = await asyncio.to_thread(fetch_info_links, disease_name, crop_name)
                    except Exception as e:
                        logger.warning(f"[Web Agent] Info link backfill failed: {e}")
                        links = []

                if not links:
                    links = [build_search_fallback_url(f"{crop_name or ''} {disease_name} treatment")]

                existing_report.product_links = links
                if not existing_report.disease_name:
                    existing_report.disease_name = label

                await db.flush()

        # Reconstruct expert_analysis from stored JSONB if present
        cached_expert: ExpertAgentOutput | None = None
        if existing_report.expert_analysis:
            cached_expert = ExpertAgentOutput(**existing_report.expert_analysis)

        return AiReportResponse(
            prediction_id=prediction_id,
            report=ReportAgentOutput(
                is_diseased=existing_report.is_diseased,
                disease_name=existing_report.disease_name,
                severity=existing_report.severity,
                report_text=existing_report.report_text,
                treatments=existing_report.treatments,
            ),
            expert_analysis=cached_expert,
            product_links=existing_report.product_links or [],
        )

    # Get top prediction
    sorted_results = sorted(prediction.prediction_results, key=lambda r: r.rank)
    top = sorted_results[0] if sorted_results else None

    if not top:
        raise ValueError(f"No prediction results found for prediction {prediction_id}")

    label = top.label
    confidence = top.confidence

    # Infer crop name from label if not provided (labels like "Tomato__Early_blight")
    if not crop:
        parts = label.split("__")
        crop = parts[0].replace("_", " ") if parts else "Unknown crop"

    # ── 2. Run Report Agent ────────────────────────────────────────
    logger.info(f"[Report Agent] Generating report for: {label} ({confidence:.2%})")
    report: ReportAgentOutput = await asyncio.to_thread(generate_report, label, confidence)

    # Enforce "healthy vs diseased" deterministically from the model label.
    label_is_healthy = "healthy" in label.lower()
    report.is_diseased = not label_is_healthy

    if report.is_diseased and not report.disease_name:
        report.disease_name = label

    # ── 3. Conditional agents ──────────────────────────────────────
    expert_analysis: ExpertAgentOutput | None = None
    product_links: list[str] = []

    if report.is_diseased:
        disease_name = report.disease_name or label

        # Web Agent — fetch treatment links (mandatory)
        logger.info(f"[Web Agent] Searching treatment links for: {disease_name}")
        try:
            web_result: WebAgentOutput = await asyncio.to_thread(fetch_product_links, disease_name)
            product_links = web_result.product_links or []
        except Exception as e:
            logger.warning(f"[Web Agent] Product link search failed: {e}")
            product_links = []

        # Fallback: authoritative info/management links
        if not product_links:
            logger.info(f"[Web Agent] Falling back to info links for: {disease_name}")
            try:
                product_links = await asyncio.to_thread(fetch_info_links, disease_name, crop)
            except Exception as e:
                logger.warning(f"[Web Agent] Info link search failed: {e}")
                product_links = []

        # Last resort: safe search URL
        if not product_links:
            product_links = [build_search_fallback_url(f"{crop} {disease_name} treatment")]

        # Expert Agent — analyze condition
        logger.info(f"[Expert Agent] Analyzing: {crop} / {disease_name}")
        try:
            current_weather, _ = await get_current_weather(location)
            expert_analysis = await asyncio.to_thread(
                analyze_crop_condition,
                crop=crop,
                disease=disease_name,
                weather=current_weather,
            )
        except Exception as e:
            logger.warning(f"[Expert Agent] Failed: {e}")
            expert_analysis = None

    # ── 4. Save to DB ──────────────────────────────────────────────
    treatments_json = None
    if report.treatments:
        treatments_json = [t.model_dump() for t in report.treatments]

    expert_analysis_json = None
    if expert_analysis:
        expert_analysis_json = expert_analysis.model_dump()

    ai_report = AiReport(
        prediction_id=prediction_id,
        user_id=prediction.user_id,
        report_text=report.report_text,
        is_diseased=report.is_diseased,
        disease_name=report.disease_name,
        confidence_score=confidence,
        severity=report.severity,
        treatments=treatments_json,
        product_links=product_links if product_links else None,
        expert_analysis=expert_analysis_json,
    )
    db.add(ai_report)
    await db.flush()

    logger.info(f"[Orchestrator] AI report saved: {ai_report.id}")

    # ── 5. Return merged response ──────────────────────────────────
    return AiReportResponse(
        prediction_id=prediction_id,
        report=report,
        expert_analysis=expert_analysis,
        product_links=product_links,
    )


async def orchestrate_crop_plan(
    uid: str,
    location: LocationItem | str,
    db: AsyncSession,
) -> CropPlanResponse:
    """
    Crop planning pipeline:
    1. Verify user
    2. Fetch live weather & resolve location name
    3. Run Crop Planner Agent with KB context
    4. Save to crop_plans table
    """

    # ── 1. Verify user ─────────────────────────────────────────────
    stmt = select(User).where(User.firebase_uid == uid)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        raise ValueError(f"User with uid '{uid}' not found")

    # ── 2. Fetch Live Info ─────────────────────────────────────────
    current_weather, loc_name = await get_current_weather(location)

    # ── 3. Run Crop Planner Agent ──────────────────────────────────
    log_loc = location if isinstance(location, str) else f"({location.lat}, {location.lon})"
    logger.info(f"[Crop Planner] Planning for: {loc_name} {log_loc}")
    plan: CropPlanResponse = await asyncio.to_thread(
        plan_crop,
        location_context=loc_name,
        weather=current_weather,
    )

    # ── 4. Save to DB ──────────────────────────────────────────────
    crop_plan = CropPlan(
        user_id=user.id,
        location=loc_name,
        weather_summary=current_weather,
        soil_type=plan.deduced_soil_type,
        recommended_crops=[c.model_dump() for c in plan.recommended_crops],
    )
    db.add(crop_plan)
    await db.flush()

    logger.info(f"[Orchestrator] Crop plan saved: {crop_plan.id} with soil: {plan.deduced_soil_type}")

    return plan
