"""Report Agent — generates structured AI diagnosis from prediction data."""

from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GOOGLE_API_KEY, GEMINI_MODEL
from app.schemas import ReportAgentOutput

REPORT_PROMPT = """You are an AI crop disease diagnostician.

Input:
Label: {label}
Confidence: {confidence:.4f}

Tasks:
1. Diseased? (true/false). 'healthy' in label means false.
2. Name the disease (if any).
3. Assess severity (low/medium/high) based on confidence & type.
4. Report: 2-3 paragraphs farmer-friendly diagnosis.
5. Treatments: Suggest ≥3 step-by-step treatments (with dosages if applicable).

Rules:
- Simple language.
- Provide specific chemical names/dosages.
- If healthy: is_diseased=false, other fields null."""


def generate_report(label: str, confidence: float) -> ReportAgentOutput:
    """Call Gemini with structured output to generate a disease report."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )

    structured_llm = llm.with_structured_output(ReportAgentOutput)

    result = structured_llm.invoke(
        REPORT_PROMPT.format(label=label, confidence=confidence)
    )

    return result
