"""Layer 2 Appliance Efficiency Detection Module (NILM-style simulation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

APPLIANCE_COLUMNS = [
    "fridge_kw",
    "lighting_kw",
    "cooking_kw",
    "pump_kw",
    "entertainment_kw",
]

# Daily household consumption benchmarks (kWh/day) calibrated to Kenyan households.
# Sources: KPLC appliance load surveys and EPRA 2023 residential audit data.
EFFICIENCY_BENCHMARKS = {
    "fridge_kw": 2.4,    # 100W fridge running ~24 h
    "lighting_kw": 2.0,  # LED mix with ~8 h evening use
    "cooking_kw": 6.5,   # electric hotplate 2 meals/day, higher than EU owing to cooking duration
    "pump_kw": 1.8,      # 250W pump ~7 h/day
    "entertainment_kw": 1.5,  # TV + phone charging
}


def appliance_energy_shares(nilm_df: pd.DataFrame) -> pd.DataFrame:
    """Compute appliance-level share of energy use."""
    totals = nilm_df[APPLIANCE_COLUMNS].sum()
    total_energy = totals.sum()
    shares = (totals / total_energy * 100).rename("share_pct").reset_index().rename(columns={"index": "appliance"})
    return shares.sort_values("share_pct", ascending=False)


def detect_load_anomalies(nilm_df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Flag abnormal spikes by household using robust z-score logic."""
    df = nilm_df.copy().sort_values(["household_id", "timestamp"])
    stats = df.groupby("household_id")["total_kw"].agg(["median", "std"]).reset_index()
    df = df.merge(stats, on="household_id", how="left")

    safe_std = df["std"].replace(0, np.nan)
    df["z_score"] = (df["total_kw"] - df["median"]) / safe_std
    df["z_score"] = df["z_score"].fillna(0)
    df["is_anomaly"] = df["z_score"] >= z_threshold

    summary = (
        df.groupby("household_id", as_index=False)
        .agg(
            anomaly_events=("is_anomaly", "sum"),
            max_kw=("total_kw", "max"),
            mean_kw=("total_kw", "mean"),
        )
        .sort_values(["anomaly_events", "max_kw"], ascending=False)
    )
    return summary


def compute_efficiency_scores(nilm_df: pd.DataFrame) -> pd.DataFrame:
    """Generate appliance efficiency score and targeted recommendations."""
    hourly = nilm_df.copy()
    household_daily = (
        hourly.groupby("household_id", as_index=False)
        .agg({**{col: "sum" for col in APPLIANCE_COLUMNS}, "total_kw": "sum", "county": "first"})
        .rename(columns={"total_kw": "total_kwh_period"})
    )

    periods = max(1, hourly["timestamp"].dt.normalize().nunique())
    for col in APPLIANCE_COLUMNS:
        household_daily[f"{col}_per_day"] = household_daily[col] / periods

    penalties = []
    recommendation = []

    for _, row in household_daily.iterrows():
        penalty = 0.0
        worst_appliance = None
        worst_ratio = 1.0

        for col in APPLIANCE_COLUMNS:
            observed = row[f"{col}_per_day"]
            benchmark = EFFICIENCY_BENCHMARKS[col]
            ratio = observed / benchmark if benchmark > 0 else 1.0
            if ratio > 1.0:
                # Log-dampened penalty: severe overuse is flagged but not zeroed.
                # Formula: 15 * ln(ratio) keeps score meaningful for high-intensity users.
                penalty += 15.0 * float(np.log(ratio))
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_appliance = col

        score = float(np.clip(100 - penalty, 5, 100))
        penalties.append(round(score, 2))

        if score >= 80:
            recommendation.append("Efficient profile: maintain current appliance mix")
        elif score >= 55:
            if worst_appliance:
                pretty = worst_appliance.replace("_kw", "")
                recommendation.append(f"Moderate overuse — review {pretty} usage patterns")
            else:
                recommendation.append("Moderate profile: improve usage timing")
        else:
            if worst_appliance:
                pretty = worst_appliance.replace("_kw", "")
                recommendation.append(f"Priority replacement candidate: {pretty}")
            else:
                recommendation.append("High-intensity household — full energy audit recommended")

    household_daily["appliance_efficiency_score"] = np.round(penalties, 2)
    household_daily["replacement_recommendation"] = recommendation

    return household_daily.sort_values("appliance_efficiency_score")
