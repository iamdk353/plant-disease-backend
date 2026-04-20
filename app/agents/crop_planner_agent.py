"""Crop Planner Agent — suggests optimal crops based on environment and market."""

from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GOOGLE_API_KEY, GEMINI_MODEL
from app.schemas import CropPlanResponse

PLANNER_PROMPT = """You are an expert agricultural planner.

KNOWLEDGE BASE (Indian Soil/Regions):
- Red Soil: land(upland, plateau, semi-arid), regions(South KA, TN, AP, TS), water_retention(low), fertility(low), crops(ragi, millets, groundnut)
- Black Soil: land(basalt plateau, plains), regions(North KA, MH, NW TS), water_retention(high), fertility(med-high), crops(cotton, jowar)
- Laterite Soil: land(hills, high rain), regions(Western Ghats, Malnad, Highlands KL), water_retention(med), fertility(low), crops(coffee, tea, spices)
- Alluvial Soil: land(river basin/floodplain), regions(Cauvery basin/delta, River basins KA, Delta AP), water_retention(med), fertility(high), crops(rice, sugarcane)
- Coastal Soil: land(coastal plains), regions(Udupi, Mangalore, Coastal KA), water_retention(low), fertility(low), crops(coconut, cashew)
- Forest Soil: land(mountains/forest), regions(Western Ghats), water_retention(var), fertility(med), crops(spices, plantations)

Inputs:
Location: {location_context}
Weather: {weather}

Tasks:
1. Deduce 'deduced_soil_type' strictly from KB + Location.
2. Suggest 3-5 best crops for this soil + weather.
3. Brief explanation for each (1-2 sentences).

Rules:
- Prioritize KB-listed crops for the deduced soil.
- Add others if weather/commercial viability allows."""


def plan_crop(
    location_context: str,
    weather: str,
) -> CropPlanResponse:
    """Call Gemini with structured output to generate crop recommendations."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,  # Lower temperature for more deterministic soil selection
    )

    structured_llm = llm.with_structured_output(CropPlanResponse)

    result = structured_llm.invoke(
        PLANNER_PROMPT.format(
            location_context=location_context,
            weather=weather,
        )
    )

    return result
