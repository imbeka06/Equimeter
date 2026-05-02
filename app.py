"""EquiMeter AI Streamlit dashboard.

Architect and Developer: IMBEKA MUSA
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    BASE_TARIFF_RATES,
    DEFAULT_HOUSEHOLDS,
    DEFAULT_RANDOM_SEED,
    NILM_HOUSEHOLDS,
    PROJECT_CREDIT,
    PROJECT_NAME,
    PROJECT_TAGLINE,
)
from src.data.simulate_data import generate_household_dataset, generate_nilm_dataset
from src.models.equity_classifier import ImbekaMusaEquityKMeansEngine, simulate_tariff_policy
from src.models.nilm_efficiency import appliance_energy_shares, compute_efficiency_scores, detect_load_anomalies

st.set_page_config(page_title=PROJECT_NAME, layout="wide")


@st.cache_data(show_spinner=False)
def build_household_data(n_households: int, seed: int) -> pd.DataFrame:
    return generate_household_dataset(n_households=n_households, seed=seed)


@st.cache_data(show_spinner=False)
def run_equity_layer(household_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    engine = ImbekaMusaEquityKMeansEngine(random_state=seed)
    return engine.fit_predict(household_df)


@st.cache_data(show_spinner=False)
def run_nilm_layer(segmented_df: pd.DataFrame, seed: int, nilm_households: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nilm_df = generate_nilm_dataset(segmented_df, n_households=nilm_households, seed=seed)
    shares_df = appliance_energy_shares(nilm_df)
    anomaly_df = detect_load_anomalies(nilm_df)
    efficiency_df = compute_efficiency_scores(nilm_df)
    return nilm_df, shares_df, anomaly_df, efficiency_df


def county_heatmap_like(county_df: pd.DataFrame) -> None:
    fig = px.scatter_geo(
        county_df,
        lat="latitude",
        lon="longitude",
        color="vulnerability_score",
        size="avg_monthly_kwh",
        hover_name="county",
        scope="africa",
        color_continuous_scale="YlOrRd",
        title="County-Level Vulnerability and Demand Hotspots",
        projection="natural earth",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), height=450)
    st.plotly_chart(fig, use_container_width=True)


def render_layer_1(segmented_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Layer 1: Equity Classification Engine")

    c1, c2, c3 = st.columns(3)
    c1.metric("Households", f"{len(segmented_df):,}")
    c2.metric("Avg Vulnerability", f"{segmented_df['vulnerability_score'].mean():.1f}")
    c3.metric("Avg Monthly kWh", f"{segmented_df['avg_monthly_kwh'].mean():.1f}")

    left, right = st.columns([1.1, 1])
    with left:
        tier_count = segmented_df["equity_tier"].value_counts().reset_index()
        tier_count.columns = ["equity_tier", "count"]
        fig_tier = px.bar(
            tier_count,
            x="equity_tier",
            y="count",
            color="equity_tier",
            title="Household Segmentation by Equity Tier",
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    with right:
        fig_dist = px.histogram(
            segmented_df,
            x="vulnerability_score",
            nbins=35,
            title="Distribution of Vulnerability Scores",
            color_discrete_sequence=["#e4572e"],
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    county_rollup = (
        segmented_df.groupby("county", as_index=False)
        .agg(
            vulnerability_score=("vulnerability_score", "mean"),
            avg_monthly_kwh=("avg_monthly_kwh", "mean"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .sort_values("vulnerability_score", ascending=False)
    )

    county_heatmap_like(county_rollup)

    st.markdown("### Policy Simulation Interface")
    with st.expander("Adjust tariff rates and subsidy policy", expanded=True):
        t1, t2, t3, t4 = st.columns(4)
        lifeline = t1.slider("Lifeline (KES/kWh)", 4.0, 18.0, float(BASE_TARIFF_RATES["Lifeline"]), 0.5)
        social = t2.slider("Social (KES/kWh)", 8.0, 24.0, float(BASE_TARIFF_RATES["Social"]), 0.5)
        standard = t3.slider("Standard (KES/kWh)", 12.0, 34.0, float(BASE_TARIFF_RATES["Standard"]), 0.5)
        premium = t4.slider("Premium (KES/kWh)", 18.0, 45.0, float(BASE_TARIFF_RATES["Premium"]), 0.5)

        s1, s2 = st.columns(2)
        subsidy_rate = s1.slider("Subsidy Rate for Eligible Households", 0.0, 0.6, 0.2, 0.05)
        affordability_threshold = s2.slider("Affordability Threshold (bill/income)", 0.03, 0.2, 0.1, 0.01)

    policy_df, outcome = simulate_tariff_policy(
        segmented_df,
        tariff_rates={
            "Lifeline": lifeline,
            "Social": social,
            "Standard": standard,
            "Premium": premium,
        },
        subsidy_rate=subsidy_rate,
        affordability_threshold=affordability_threshold,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Affordability Rate", f"{outcome.affordability_rate:.1f}%")
    m2.metric("Utility Revenue", f"KES {outcome.utility_revenue_kes:,.0f}")
    m3.metric("Total Subsidy", f"KES {outcome.subsidy_kes:,.0f}")
    m4.metric("Eligible Households", f"{outcome.affected_households:,}")

    return policy_df


def render_layer_2(nilm_df: pd.DataFrame, shares_df: pd.DataFrame, anomaly_df: pd.DataFrame, efficiency_df: pd.DataFrame) -> None:
    st.subheader("Layer 2: Appliance Efficiency Detection (NILM)")

    profile = nilm_df.copy()
    profile["hour"] = profile["timestamp"].dt.hour
    hourly_profile = profile.groupby("hour", as_index=False)["total_kw"].mean()

    l1, l2 = st.columns([1.25, 1])
    with l1:
        fig_load = px.line(
            hourly_profile,
            x="hour",
            y="total_kw",
            markers=True,
            title="Average Load Profile by Hour",
        )
        st.plotly_chart(fig_load, use_container_width=True)

    with l2:
        fig_share = px.pie(
            shares_df,
            names="appliance",
            values="share_pct",
            title="Appliance Energy Contribution",
        )
        st.plotly_chart(fig_share, use_container_width=True)

    a1, a2, a3 = st.columns(3)
    a1.metric("Monitored Households", f"{nilm_df['household_id'].nunique():,}")
    a2.metric("Anomalous Households", f"{(anomaly_df['anomaly_events'] > 0).sum():,}")
    a3.metric("Avg Efficiency Score", f"{efficiency_df['appliance_efficiency_score'].mean():.1f}")

    st.markdown("### Abnormal Consumption Detection")
    st.dataframe(anomaly_df.head(20), use_container_width=True)

    st.markdown("### Efficiency Benchmarks and Targeted Replacements")
    st.dataframe(
        efficiency_df[["household_id", "county", "appliance_efficiency_score", "replacement_recommendation"]].head(25),
        use_container_width=True,
    )


def render_unified_view(policy_df: pd.DataFrame, efficiency_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Unified Decision-Support View")

    merged = policy_df.merge(
        efficiency_df[["household_id", "appliance_efficiency_score", "replacement_recommendation"]],
        on="household_id",
        how="left",
    )

    summary = (
        merged.groupby(["county", "equity_tier"], as_index=False)
        .agg(
            households=("household_id", "count"),
            avg_bill_kes=("bill_after_subsidy_kes", "mean"),
            avg_vulnerability=("vulnerability_score", "mean"),
            avg_efficiency=("appliance_efficiency_score", "mean"),
            affordability_rate=("is_affordable", "mean"),
        )
        .sort_values(["county", "equity_tier"])
    )
    summary["affordability_rate"] = summary["affordability_rate"] * 100

    st.dataframe(summary, use_container_width=True)

    fig = px.scatter(
        summary,
        x="avg_vulnerability",
        y="avg_efficiency",
        size="households",
        color="equity_tier",
        hover_name="county",
        title="County-Tier Balance: Vulnerability vs Appliance Efficiency",
    )
    st.plotly_chart(fig, use_container_width=True)

    return merged


def render_export(policy_df: pd.DataFrame, nilm_df: pd.DataFrame, merged_df: pd.DataFrame) -> None:
    st.subheader("Exportable Regulatory Datasets")

    st.download_button(
        label="Download Layer 1 Classification CSV",
        data=policy_df.to_csv(index=False).encode("utf-8"),
        file_name="equimeter_layer1_classification.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download Layer 2 NILM Analysis CSV",
        data=nilm_df.to_csv(index=False).encode("utf-8"),
        file_name="equimeter_layer2_nilm.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download Unified Decision Dataset CSV",
        data=merged_df.to_csv(index=False).encode("utf-8"),
        file_name="equimeter_unified_dashboard.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title(f"{PROJECT_NAME} | {PROJECT_TAGLINE}")
    st.caption(PROJECT_CREDIT)

    st.sidebar.header("Simulation Controls")
    seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=999999, value=DEFAULT_RANDOM_SEED, step=1)
    n_households = st.sidebar.slider("Number of Households", 500, 6000, DEFAULT_HOUSEHOLDS, 100)
    nilm_households = st.sidebar.slider("NILM Monitored Households", 30, 300, NILM_HOUSEHOLDS, 10)

    household_df = build_household_data(n_households=n_households, seed=int(seed))
    segmented_df = run_equity_layer(household_df, seed=int(seed))
    nilm_df, shares_df, anomaly_df, efficiency_df = run_nilm_layer(segmented_df, seed=int(seed), nilm_households=nilm_households)

    tabs = st.tabs(
        [
            "Unified Dashboard",
            "Layer 1: Classification",
            "Layer 2: NILM Efficiency",
            "Export",
        ]
    )

    with tabs[1]:
        policy_df = render_layer_1(segmented_df)

    with tabs[2]:
        render_layer_2(nilm_df, shares_df, anomaly_df, efficiency_df)

    with tabs[0]:
        policy_df_for_unified, _ = simulate_tariff_policy(segmented_df)
        merged_df = render_unified_view(policy_df_for_unified, efficiency_df)

    with tabs[3]:
        policy_df_for_export, _ = simulate_tariff_policy(segmented_df)
        merged_df_for_export = policy_df_for_export.merge(
            efficiency_df[["household_id", "appliance_efficiency_score", "replacement_recommendation"]],
            on="household_id",
            how="left",
        )
        render_export(policy_df_for_export, nilm_df, merged_df_for_export)


if __name__ == "__main__":
    main()
