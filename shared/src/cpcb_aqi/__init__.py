"""CPCB-style AQI from PM2.5 / PM10 (CLAUDE.md §5). Shared by ml/ and backend/."""

from cpcb_aqi.aqi import (
    BANDS,
    CATEGORIES,
    CATEGORY_TO_BAND,
    CATEGORY_UPPER,
    PM10_BREAKPOINTS,
    PM25_BREAKPOINTS,
    aqi_band,
    aqi_category,
    category_to_band,
    cpcb_aqi,
    label_pm,
    pm10_subindex,
    pm25_subindex,
)

__all__ = [
    "BANDS", "CATEGORIES", "CATEGORY_TO_BAND", "CATEGORY_UPPER", "PM10_BREAKPOINTS", "PM25_BREAKPOINTS",
    "aqi_band", "aqi_category", "category_to_band", "cpcb_aqi", "label_pm", "pm10_subindex", "pm25_subindex",
]
