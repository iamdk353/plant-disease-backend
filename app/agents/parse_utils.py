"""Utility for structured output from models that lack native JSON mode (e.g. Gemma)."""

import json
import re
import logging
from typing import TypeVar, Type

from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model response, stripping markdown fences."""
    # Try ```json ... ``` first
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()

    # Fallback: find the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]

    return text.strip()


def invoke_with_schema(
    llm: ChatGoogleGenerativeAI,
    prompt: str,
    schema: Type[T],
) -> T:
    """
    Invoke an LLM that lacks JSON-mode support, appending a JSON schema
    instruction to the prompt, then parsing the response into *schema*.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    full_prompt = (
        f"{prompt}\n\n"
        "IMPORTANT: You MUST respond with ONLY a valid JSON object matching this schema:\n"
        f"```json\n{schema_json}\n```\n"
        "Do NOT include any text outside the JSON object."
    )

    response = llm.invoke(full_prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    cleaned = _extract_json(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed for model response:\n%s", raw)
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc

    return schema.model_validate(data)
