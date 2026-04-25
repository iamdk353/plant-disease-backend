import httpx
from fastapi import APIRouter, HTTPException

from app.config import WEATHER_API_KEY

router = APIRouter(prefix="/weather", tags=["Weather"])


@router.get("")
async def get_weather(lat: float, lon: float):
    """Fetch weather data from OpenWeatherMap API."""
    if not WEATHER_API_KEY:
        raise HTTPException(500, "WEATHER_API_KEY not configured")

    params = {
        "lat": lat,
        "lon": lon,
        "appid": WEATHER_API_KEY,
        "units": "metric",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params=params,
        )

        if response.status_code != 200:
            raise HTTPException(
                response.status_code,
                f"OpenWeatherMap API error: {response.text}",
            )

        return response.json()
