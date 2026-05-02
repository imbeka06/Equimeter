"""Layer 1 Equity Classification Engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BASE_TARIFF_RATES, TARIFF_BANDS


@dataclass
class PolicySimulationResult:
    affordability_rate: float
    subsidy_kes: float
    utility_revenue_kes: float
    average_bill_kes: float
    affected_households: int


class EquityKMeansEngine:
    """K-Means based segmentation for vulnerability-aware tariffs."""

    numeric_features = [
        "avg_monthly_kwh",
        "household_size",
        "rooms",
        "peak_ratio",
        "weekend_ratio",
        "arrears_rate",
        "outage_hours",
        "county_income_index",
        "estimated_monthly_income_kes",
        "vulnerability_score",
    ]
    categorical_features = ["county", "meter_type", "housing_type", "roof_material"]

    def __init__(self, n_clusters: int = 4, random_state: int = 2026) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.pipeline = Pipeline(
            steps=[
                (
                    "pre",
                    ColumnTransformer(
                        transformers=[
                            ("num", StandardScaler(), self.numeric_features),
                            (
                                "cat",
                                OneHotEncoder(handle_unknown="ignore"),
                                self.categorical_features,
                            ),
                        ]
                    ),
                ),
                (
                    "kmeans",
                    KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20),
                ),
            ]
        )

    def fit_predict(self, household_df: pd.DataFrame) -> pd.DataFrame:
        df = household_df.copy()
        cluster_ids = self.pipeline.fit_predict(df)
        df["cluster_id"] = cluster_ids

        mapping = self._build_cluster_tier_mapping(df)
        df["equity_tier"] = df["cluster_id"].map(mapping)
        df["recommended_tariff_band"] = df["equity_tier"].map(TARIFF_BANDS)
        return df

    def _build_cluster_tier_mapping(self, clustered_df: pd.DataFrame) -> dict[int, str]:
        profile = (
            clustered_df.groupby("cluster_id", as_index=False)
            .agg(
                vulnerability_score=("vulnerability_score", "mean"),
                avg_monthly_kwh=("avg_monthly_kwh", "mean"),
                county_income_index=("county_income_index", "mean"),
            )
            .sort_values("cluster_id")
        )

        mapping: dict[int, str] = {}

        vulnerable_cluster = int(profile.sort_values("vulnerability_score", ascending=False).iloc[0]["cluster_id"])
        mapping[vulnerable_cluster] = "Vulnerable"

        remaining = profile[~profile["cluster_id"].isin(mapping.keys())]
        high_intensity_cluster = int(remaining.sort_values("avg_monthly_kwh", ascending=False).iloc[0]["cluster_id"])
        mapping[high_intensity_cluster] = "High-Intensity Users"

        remaining = profile[~profile["cluster_id"].isin(mapping.keys())]
        low_income_cluster = int(remaining.sort_values("county_income_index", ascending=True).iloc[0]["cluster_id"])
        mapping[low_income_cluster] = "Low-Income"

        remaining = profile[~profile["cluster_id"].isin(mapping.keys())]
        if not remaining.empty:
            mapping[int(remaining.iloc[0]["cluster_id"])] = "Middle-Income"

        return mapping


def simulate_tariff_policy(
    segmented_df: pd.DataFrame,
    tariff_rates: dict[str, float] | None = None,
    subsidy_rate: float = 0.2,
    affordability_threshold: float = 0.1,
) -> tuple[pd.DataFrame, PolicySimulationResult]:
    """Run affordability and revenue simulation for a tariff policy."""
    rates = tariff_rates or BASE_TARIFF_RATES

    sim_df = segmented_df.copy()
    sim_df["tariff_rate_kes_per_kwh"] = sim_df["recommended_tariff_band"].map(rates)
    sim_df["monthly_bill_kes"] = sim_df["avg_monthly_kwh"] * sim_df["tariff_rate_kes_per_kwh"]

    sim_df["subsidy_eligible"] = sim_df["equity_tier"].isin(["Vulnerable", "Low-Income"])
    sim_df["subsidy_kes"] = np.where(
        sim_df["subsidy_eligible"],
        sim_df["monthly_bill_kes"] * subsidy_rate,
        0,
    )
    sim_df["bill_after_subsidy_kes"] = sim_df["monthly_bill_kes"] - sim_df["subsidy_kes"]

    sim_df["affordability_ratio"] = (
        sim_df["bill_after_subsidy_kes"] / sim_df["estimated_monthly_income_kes"]
    )
    sim_df["is_affordable"] = sim_df["affordability_ratio"] <= affordability_threshold

    result = PolicySimulationResult(
        affordability_rate=float(sim_df["is_affordable"].mean() * 100),
        subsidy_kes=float(sim_df["subsidy_kes"].sum()),
        utility_revenue_kes=float(sim_df["bill_after_subsidy_kes"].sum()),
        average_bill_kes=float(sim_df["bill_after_subsidy_kes"].mean()),
        affected_households=int(sim_df["subsidy_eligible"].sum()),
    )

    return sim_df, result
