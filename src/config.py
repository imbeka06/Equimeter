"""Global configuration for EquiMeter AI.

Architect and Developer: IMBEKA MUSA
"""

from __future__ import annotations

PROJECT_NAME = "EquiMeter AI"
PROJECT_TAGLINE = "EPRA Hackathon 2026 Decision-Support Dashboard"
PROJECT_CREDIT = "Architect and Developer IMBEKA MUSA"
DEFAULT_RANDOM_SEED = 2026
DEFAULT_HOUSEHOLDS = 1500
NILM_HOUSEHOLDS = 120

EQUITY_TIERS = [
    "Vulnerable",
    "Low-Income",
    "Middle-Income",
    "High-Intensity Users",
]

TARIFF_BANDS = {
    "Vulnerable": "Lifeline",
    "Low-Income": "Social",
    "Middle-Income": "Standard",
    "High-Intensity Users": "Premium",
}

BASE_TARIFF_RATES = {
    "Lifeline": 9.5,
    "Social": 14.0,
    "Standard": 18.5,
    "Premium": 25.0,
}
