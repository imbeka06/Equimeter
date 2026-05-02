"""Global configuration for EquiMeter."""

from __future__ import annotations

PROJECT_NAME = "EquiMeter"
PROJECT_TAGLINE = "Electricity Tariff Decision-Support System"
PROJECT_CREDIT = "EquiMeter AI Platform"
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
