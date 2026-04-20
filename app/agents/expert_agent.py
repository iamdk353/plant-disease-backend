"""Expert Agent — veteran agricultural analysis of disease cause and prevention."""

from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GOOGLE_API_KEY, GEMINI_MODEL
from app.schemas import ExpertAgentOutput

EXPERT_PROMPT = """You're an agricultural expert.

Inputs: Crop: {crop}, Disease: {disease}, Weather: {weather}

Explain:
1. Root causes of this disease.
2. How current weather impacts it.
3. Preventive measures (chemical & organic) moving forward.

Rules:
- Practical language for farmers.
- Reference specific weather metrics.
- Keep sections concise (2-3 sentences)."""


def analyze_crop_condition(
    crop: str,
    disease: str,
    weather: str = "Not provided",
) -> ExpertAgentOutput:
    """Call Gemini with structured output to produce expert agricultural analysis."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.4,
    )

    structured_llm = llm.with_structured_output(ExpertAgentOutput)

    result = structured_llm.invoke(
        EXPERT_PROMPT.format(crop=crop, disease=disease, weather=weather)
    )

    return result
