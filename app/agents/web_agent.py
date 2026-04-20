"""Web Agent — fetches treatment links for disease management.

Primary: product purchase links.
Fallback: authoritative info/management links.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from app.config import GEMINI_MODEL, GOOGLE_API_KEY, TAVILY_API_KEY
from app.schemas import WebAgentOutput


def build_search_fallback_url(query: str) -> str:
    # Always-valid URL, no hallucinated domains.
    return f"https://www.google.com/search?q={quote_plus(query)}"


def _extract_urls(obj: Any) -> list[str]:
    """Best-effort extraction of URLs from Tavily results (dict/list/string)."""
    urls: list[str] = []

    if isinstance(obj, dict):
        # Tavily commonly returns {"results": [{"url": ...}, ...]}
        results = obj.get("results") or obj.get("data") or obj.get("items")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    u = item.get("url") or item.get("link")
                    if isinstance(u, str):
                        urls.append(u)
        # Sometimes directly contains url
        u = obj.get("url")
        if isinstance(u, str):
            urls.append(u)

    elif isinstance(obj, list):
        for item in obj:
            urls.extend(_extract_urls(item))

    elif isinstance(obj, str):
        # Fallback: regex scan for URLs in text
        urls.extend(re.findall(r"https?://\S+", obj))

    # Normalize + de-dup
    cleaned: list[str] = []
    seen = set()
    for u in urls:
        u = u.strip().rstrip(")].,\"'")
        if not u.startswith("http"):
            continue
        if u in seen:
            continue
        seen.add(u)
        cleaned.append(u)

    return cleaned


def _tavily_search(query: str, *, max_results: int = 5) -> list[str]:
    if not TAVILY_API_KEY:
        return []

    search_tool = TavilySearch(
        max_results=max_results,
        topic="general",
        tavily_api_key=TAVILY_API_KEY,
    )
    res = search_tool.invoke(query)
    return _extract_urls(res)


def fetch_product_links(disease_name: str) -> WebAgentOutput:
    """Try to find purchasable treatment product links.

    - Uses Tavily search.
    - If Gemini key is available, uses LLM to filter to product URLs.
    - If not, returns best-effort URLs directly from search results.
    """

    search_query = f"buy treatment product for {disease_name} plant disease India"
    urls = _tavily_search(search_query, max_results=7)

    if not urls:
        return WebAgentOutput(product_links=[])

    if not GOOGLE_API_KEY:
        return WebAgentOutput(product_links=urls[:5])

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )

    structured_llm = llm.with_structured_output(WebAgentOutput)

    extraction_prompt = f"""Extract valid product URLs from these candidate URLs for "{disease_name}" treatments.

Candidate URLs:
{urls}

Rules:
- Only links for purchasable products (fungicides/pesticides/bio-agents)
- Prefer Indian e-commerce
- Return 1-5 URLs
- Empty list if none match"""

    try:
        return structured_llm.invoke(extraction_prompt)
    except Exception:
        # Fall back to best-effort if LLM fails
        return WebAgentOutput(product_links=urls[:5])


def fetch_info_links(disease_name: str, crop: str | None = None) -> list[str]:
    """Fallback links: authoritative info / management guides (non-product)."""
    crop_part = f"{crop} " if crop else ""

    queries = [
        f"{crop_part}{disease_name} disease management ICAR",
        f"{crop_part}{disease_name} integrated disease management India PDF",
        f"{crop_part}{disease_name} KVK advisory",
    ]

    urls: list[str] = []
    for q in queries:
        urls.extend(_tavily_search(q, max_results=5))
        if len(urls) >= 5:
            break

    # Dedupe preserve order
    out: list[str] = []
    seen = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= 5:
            break

    return out
