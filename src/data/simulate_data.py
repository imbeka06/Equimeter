"""Synthetic socioeconomic and smart-meter datasets for EquiMeter AI.

Architect and Developer: IMBEKA MUSA
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COUNTY_PROFILES = [
    {"county": "Nairobi", "lat": -1.286389, "lon": 36.817223, "income_index": 0.92},
    {"county": "Mombasa", "lat": -4.043477, "lon": 39.668206, "income_index": 0.72},
    {"county": "Kisumu", "lat": -0.091702, "lon": 34.767956, "income_index": 0.66},
    {"county": "Nakuru", "lat": -0.303099, "lon": 36.080025, "income_index": 0.69},
    {"county": "Uasin Gishu", "lat": 0.514277, "lon": 35.269779, "income_index": 0.61},
    {"county": "Kiambu", "lat": -1.173222, "lon": 36.835844, "income_index": 0.81},
    {"county": "Machakos", "lat": -1.517683, "lon": 37.263414, "income_index": 0.63},
    {"county": "Kakamega", "lat": 0.282731, "lon": 34.751968, "income_index": 0.54},
    {"county": "Garissa", "lat": -0.45694, "lon": 39.658871, "income_index": 0.46},
    {"county": "Turkana", "lat": 3.312247, "lon": 35.565786, "income_index": 0.42},
]


def _minmax(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / span


def generate_household_dataset(n_households: int = 1500, seed: int = 2026) -> pd.DataFrame:
    """Generate KNBS-inspired household and consumption data."""
    rng = np.random.default_rng(seed)
    county_df = pd.DataFrame(COUNTY_PROFILES)

    sampled = county_df.sample(n=n_households, replace=True, random_state=seed).reset_index(drop=True)
    sampled["household_id"] = [f"HH-{i:05d}" for i in range(1, n_households + 1)]

    meter_type = rng.choice(["Prepaid", "Postpaid", "Smart"], size=n_households, p=[0.52, 0.3, 0.18])
    housing_type = rng.choice(
        ["Informal", "Apartment", "Standalone", "Rural Homestead"],
        size=n_households,
        p=[0.2, 0.34, 0.3, 0.16],
    )
    roof_material = rng.choice(["Iron Sheet", "Tile", "Concrete", "Asbestos"], size=n_households)
    rooms = rng.integers(1, 7, size=n_households)
    household_size = rng.integers(1, 9, size=n_households)

    base_consumption = (
        60
        + sampled["income_index"].to_numpy() * 220
        + household_size * rng.normal(16, 4, size=n_households)
        + (meter_type == "Smart") * 18
        - (housing_type == "Informal") * 12
    )
    avg_monthly_kwh = np.clip(base_consumption + rng.normal(0, 22, n_households), 25, 620)

    peak_ratio = np.clip(rng.normal(0.31, 0.1, n_households), 0.08, 0.75)
    weekend_ratio = np.clip(rng.normal(0.22, 0.08, n_households), 0.04, 0.6)
    arrears_rate = np.clip(0.45 - sampled["income_index"].to_numpy() + rng.normal(0, 0.08, n_households), 0, 0.9)
    outage_hours = np.clip(7.5 - sampled["income_index"].to_numpy() * 4.8 + rng.normal(0, 1.5, n_households), 0.3, 18)

    # County income index is transformed into estimated household income for affordability analytics.
    estimated_income_kes = np.clip(
        10000 + sampled["income_index"].to_numpy() * 120000 + rng.normal(0, 9500, n_households),
        8500,
        240000,
    )

    df = pd.DataFrame(
        {
            "household_id": sampled["household_id"],
            "county": sampled["county"],
            "latitude": sampled["lat"] + rng.normal(0, 0.09, n_households),
            "longitude": sampled["lon"] + rng.normal(0, 0.09, n_households),
            "meter_type": meter_type,
            "housing_type": housing_type,
            "roof_material": roof_material,
            "rooms": rooms,
            "household_size": household_size,
            "avg_monthly_kwh": avg_monthly_kwh.round(2),
            "peak_ratio": peak_ratio.round(3),
            "weekend_ratio": weekend_ratio.round(3),
            "arrears_rate": arrears_rate.round(3),
            "outage_hours": outage_hours.round(2),
            "county_income_index": sampled["income_index"].round(3),
            "estimated_monthly_income_kes": estimated_income_kes.round(2),
        }
    )

    housing_vulnerability = df["housing_type"].map(
        {
            "Informal": 1.0,
            "Rural Homestead": 0.72,
            "Apartment": 0.38,
            "Standalone": 0.32,
        }
    )

    vulnerability_raw = (
        0.28 * (1 - _minmax(df["county_income_index"]))
        + 0.2 * _minmax(df["arrears_rate"])
        + 0.16 * _minmax(df["household_size"])
        + 0.14 * _minmax(df["outage_hours"])
        + 0.12 * _minmax(housing_vulnerability)
        + 0.1 * _minmax(df["peak_ratio"])
    )
    df["vulnerability_score"] = (vulnerability_raw * 100).round(2)

    return df


def generate_nilm_dataset(
    household_df: pd.DataFrame,
    n_households: int = 120,
    days: int = 21,
    seed: int = 2026,
) -> pd.DataFrame:
    """Generate Arduino-like smart meter time-series with appliance-level traces."""
    rng = np.random.default_rng(seed + 13)

    sample_households = household_df[["household_id", "household_size", "county", "avg_monthly_kwh"]].sample(
        n=min(n_households, len(household_df)),
        random_state=seed,
    )

    timestamps = pd.date_range("2026-01-01", periods=24 * days, freq="h")
    records: list[dict] = []

    for _, hh in sample_households.iterrows():
        size = hh["household_size"]
        usage_factor = np.clip(hh["avg_monthly_kwh"] / 180, 0.5, 3.2)
        appliance_age = rng.choice([0.8, 1.0, 1.25], p=[0.35, 0.45, 0.2])

        for ts in timestamps:
            hour = ts.hour
            weekend_boost = 1.16 if ts.dayofweek >= 5 else 1.0

            fridge = (0.08 + rng.normal(0.0, 0.01)) * appliance_age
            lighting = (0.06 + (0.14 if hour >= 18 or hour <= 5 else 0.02) + rng.normal(0.0, 0.01))
            cooking = (
                0.38 if hour in {6, 7, 13, 19, 20} else 0.04
            ) * (1 + size / 10) * weekend_boost * appliance_age
            pump = (0.22 if hour in {5, 6, 7, 18} else 0.02) * rng.choice([0.3, 1.0], p=[0.75, 0.25])
            entertainment = (0.11 if 19 <= hour <= 23 else 0.03) * weekend_boost

            total = (fridge + lighting + cooking + pump + entertainment) * usage_factor
            anomaly_spike = rng.choice([0.0, rng.uniform(0.4, 1.6)], p=[0.985, 0.015])
            total_kw = max(0.03, total + anomaly_spike + rng.normal(0, 0.02))

            records.append(
                {
                    "household_id": hh["household_id"],
                    "county": hh["county"],
                    "timestamp": ts,
                    "total_kw": round(total_kw, 4),
                    "fridge_kw": round(fridge * usage_factor, 4),
                    "lighting_kw": round(lighting * usage_factor, 4),
                    "cooking_kw": round(cooking * usage_factor, 4),
                    "pump_kw": round(pump * usage_factor, 4),
                    "entertainment_kw": round(entertainment * usage_factor, 4),
                }
            )

    return pd.DataFrame.from_records(records)
