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
import plotly.graph_objects as go

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


TIER_COLORS = {
    "Vulnerable": "#d62728",
    "Low-Income": "#ff7f0e",
    "Middle-Income": "#1f77b4",
    "High-Intensity Users": "#2ca02c",
}


def county_vulnerability_map(county_df: pd.DataFrame) -> None:
    fig = px.scatter_geo(
        county_df,
        lat="latitude",
        lon="longitude",
        color="vulnerability_score",
        size="avg_monthly_kwh",
        hover_name="county",
        hover_data={"vulnerability_score": ":.1f", "avg_monthly_kwh": ":.1f", "latitude": False, "longitude": False},
        scope="africa",
        color_continuous_scale="YlOrRd",
        title="County-Level Vulnerability and Demand Hotspots",
        projection="natural earth",
    )
    fig.update_geos(center={"lat": 0.0, "lon": 37.9}, projection_scale=6)
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), height=450)
    st.plotly_chart(fig, use_container_width=True)


def county_nilm_map(nilm_df: pd.DataFrame, segmented_df: pd.DataFrame) -> None:
    """Scatter-geo map of average hourly load per county coloured by efficiency tier."""
    county_load = nilm_df.groupby("county", as_index=False)["total_kw"].mean().rename(columns={"total_kw": "avg_load_kw"})
    county_coords = (
        segmented_df.groupby("county", as_index=False)
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
    )
    map_df = county_load.merge(county_coords, on="county", how="left")
    fig = px.scatter_geo(
        map_df,
        lat="latitude",
        lon="longitude",
        size="avg_load_kw",
        color="avg_load_kw",
        hover_name="county",
        hover_data={"avg_load_kw": ":.3f kW", "latitude": False, "longitude": False},
        scope="africa",
        color_continuous_scale="Blues",
        title="County Average NILM Load (kW)",
        projection="natural earth",
    )
    fig.update_geos(center={"lat": 0.0, "lon": 37.9}, projection_scale=6)
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), height=420)
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

    county_vulnerability_map(county_rollup)

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


def render_layer_2(
    nilm_df: pd.DataFrame,
    shares_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    efficiency_df: pd.DataFrame,
    segmented_df: pd.DataFrame,
) -> None:
    st.subheader("Layer 2: Appliance Efficiency Detection (NILM)")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Monitored Households", f"{nilm_df['household_id'].nunique():,}")
    a2.metric("Anomalous Households", f"{(anomaly_df['anomaly_events'] > 0).sum():,}")
    a3.metric("Avg Efficiency Score", f"{efficiency_df['appliance_efficiency_score'].mean():.1f} / 100")
    a4.metric("Avg Peak Load", f"{nilm_df['total_kw'].max():.2f} kW")

    st.markdown("### Stacked Hourly Load Profile")
    profile = nilm_df.copy()
    profile["hour"] = profile["timestamp"].dt.hour
    from src.models.nilm_efficiency import APPLIANCE_COLUMNS
    hourly_stack = profile.groupby("hour", as_index=False)[APPLIANCE_COLUMNS].mean()
    fig_stack = go.Figure()
    for col in APPLIANCE_COLUMNS:
        fig_stack.add_trace(go.Scatter(
            x=hourly_stack["hour"],
            y=hourly_stack[col],
            name=col.replace("_kw", "").capitalize(),
            stackgroup="one",
            mode="none",
        ))
    fig_stack.update_layout(
        title="Stacked Average Load by Appliance (kW)",
        xaxis_title="Hour of Day",
        yaxis_title="Average kW",
        height=380,
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    l1, l2 = st.columns([1, 1])
    with l1:
        fig_share = px.pie(
            shares_df,
            names="appliance",
            values="share_pct",
            title="Appliance Energy Contribution Share",
        )
        st.plotly_chart(fig_share, use_container_width=True)

    with l2:
        fig_eff_hist = px.histogram(
            efficiency_df,
            x="appliance_efficiency_score",
            nbins=20,
            title="Distribution of Appliance Efficiency Scores",
            color_discrete_sequence=["#1f77b4"],
        )
        st.plotly_chart(fig_eff_hist, use_container_width=True)

    county_nilm_map(nilm_df, segmented_df)

    st.markdown("### Abnormal Consumption Detection")
    st.dataframe(anomaly_df.head(20), use_container_width=True)

    st.markdown("### Efficiency Benchmarks and Targeted Replacements")
    display_eff = efficiency_df[[
        "household_id", "county", "appliance_efficiency_score", "replacement_recommendation"
    ]].sort_values("appliance_efficiency_score").head(30)
    st.dataframe(display_eff, use_container_width=True)

    # --- Household drill-down ---
    st.markdown("### Household Drill-Down")
    monitored_ids = sorted(efficiency_df["household_id"].unique().tolist())
    selected_hh = st.selectbox("Select a monitored household", monitored_ids)
    if selected_hh:
        hh_nilm = nilm_df[nilm_df["household_id"] == selected_hh].copy()
        hh_eff = efficiency_df[efficiency_df["household_id"] == selected_hh]

        d1, d2, d3 = st.columns(3)
        d1.metric("Efficiency Score", f"{hh_eff['appliance_efficiency_score'].values[0]:.1f} / 100")
        d2.metric("Avg Load", f"{hh_nilm['total_kw'].mean():.3f} kW")
        d3.metric("Recommendation", hh_eff['replacement_recommendation'].values[0])

        hh_nilm["hour"] = hh_nilm["timestamp"].dt.hour
        hh_hourly = hh_nilm.groupby("hour", as_index=False)[APPLIANCE_COLUMNS + ["total_kw"]].mean()
        fig_hh = px.line(
            hh_hourly,
            x="hour",
            y=["total_kw"] + APPLIANCE_COLUMNS,
            markers=True,
            title=f"Load Profile: {selected_hh}",
        )
        st.plotly_chart(fig_hh, use_container_width=True)


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
    summary["affordability_rate"] = (summary["affordability_rate"] * 100).round(1)
    summary["avg_bill_kes"] = summary["avg_bill_kes"].round(2)
    summary["avg_vulnerability"] = summary["avg_vulnerability"].round(1)
    summary["avg_efficiency"] = summary["avg_efficiency"].round(1)

    st.dataframe(summary, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            summary,
            x="avg_vulnerability",
            y="avg_efficiency",
            size="households",
            color="equity_tier",
            color_discrete_map=TIER_COLORS,
            hover_name="county",
            title="County-Tier Balance: Vulnerability vs Appliance Efficiency",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig2 = px.bar(
            summary,
            x="county",
            y="affordability_rate",
            color="equity_tier",
            color_discrete_map=TIER_COLORS,
            barmode="group",
            title="Affordability Rate by County and Tier (%)",
        )
        fig2.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)

    return merged


def render_regulatory_summary(
    segmented_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    efficiency_df: pd.DataFrame,
    outcome_affordability: float,
    outcome_subsidy: float,
    outcome_revenue: float,
) -> None:
    st.subheader("Regulatory Summary Report")
    st.caption("Auto-generated policy narrative for EPRA submission")

    total_hh = len(segmented_df)
    vulnerable_n = int((segmented_df["equity_tier"] == "Vulnerable").sum())
    low_income_n = int((segmented_df["equity_tier"] == "Low-Income").sum())
    high_intensity_n = int((segmented_df["equity_tier"] == "High-Intensity Users").sum())
    avg_vuln = segmented_df["vulnerability_score"].mean()
    avg_eff = efficiency_df["appliance_efficiency_score"].mean()
    anomalous_pct = (
        efficiency_df[efficiency_df["appliance_efficiency_score"] < 55].shape[0]
        / max(1, len(efficiency_df))
        * 100
    )

    st.markdown(f"""
**EquiMeter AI — EPRA Hackathon 2026 Decision Brief**
*Architect and Developer: IMBEKA MUSA*

---

#### 1. Population Overview

The simulation models **{total_hh:,} Kenyan households** across {segmented_df['county'].nunique()} counties.
The K-Means equity classification engine assigned each household to one of four tariff tiers based on
a composite Vulnerability Score derived from income, arrears, housing type, outage exposure, and consumption patterns.

| Tier | Households | Share |
|---|---|---|
| Vulnerable | {vulnerable_n:,} | {vulnerable_n/total_hh*100:.1f}% |
| Low-Income | {low_income_n:,} | {low_income_n/total_hh*100:.1f}% |
| Middle-Income | {total_hh-vulnerable_n-low_income_n-high_intensity_n:,} | {(total_hh-vulnerable_n-low_income_n-high_intensity_n)/total_hh*100:.1f}% |
| High-Intensity Users | {high_intensity_n:,} | {high_intensity_n/total_hh*100:.1f}% |

Mean Vulnerability Score: **{avg_vuln:.1f} / 100**

---

#### 2. Tariff Policy Simulation Outcomes

Under the current tariff configuration:

- **Household Affordability Rate:** {outcome_affordability:.1f}% of households spend ≤10% of income on electricity.
- **Projected Utility Revenue:** KES {outcome_revenue:,.0f} per month across the sample.
- **Total Subsidy Disbursed:** KES {outcome_subsidy:,.0f} per month for eligible (Vulnerable + Low-Income) households.

---

#### 3. Appliance Efficiency and NILM Insights

Non-Intrusive Load Monitoring data analysed {efficiency_df['household_id'].nunique()} smart-metered households.

- Average Appliance Efficiency Score: **{avg_eff:.1f} / 100**
- Households flagged for priority replacement: **{anomalous_pct:.1f}%** of the monitored sample.
- Primary inefficiency driver: **cooking appliances** (highest share of overuse vs benchmark).

---

#### 4. Recommendations

1. Adopt a **four-band differentiated tariff** structure as modelled (Lifeline / Social / Standard / Premium).
2. Target **Vulnerable households** in Turkana and Garissa counties for lifeline connection programmes.
3. Launch a **cooking appliance replacement programme** for low-efficiency households identified by NILM.
4. Expand **smart meter rollout** to improve real-time load visibility and reduce estimated billing inaccuracy.
5. Use the EquiMeter AI Unified Dataset as input to quarterly EPRA affordability reviews.
    """)


def render_export(policy_df: pd.DataFrame, nilm_df: pd.DataFrame, merged_df: pd.DataFrame) -> None:
    st.subheader("Exportable Regulatory Datasets")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="⬇ Layer 1 Classification CSV",
            data=policy_df.to_csv(index=False).encode("utf-8"),
            file_name="equimeter_layer1_classification.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="⬇ Layer 2 NILM Analysis CSV",
            data=nilm_df.to_csv(index=False).encode("utf-8"),
            file_name="equimeter_layer2_nilm.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            label="⬇ Unified Decision Dataset CSV",
            data=merged_df.to_csv(index=False).encode("utf-8"),
            file_name="equimeter_unified_dashboard.csv",
            mime="text/csv",
            use_container_width=True,
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
            "Regulatory Summary",
            "Export",
        ]
    )

    # Compute shared policy simulation once for use across tabs
    base_policy_df, base_outcome = simulate_tariff_policy(segmented_df)

    with tabs[1]:
        policy_df = render_layer_1(segmented_df)

    with tabs[2]:
        render_layer_2(nilm_df, shares_df, anomaly_df, efficiency_df, segmented_df)

    with tabs[0]:
        merged_df = render_unified_view(base_policy_df, efficiency_df)

    with tabs[3]:
        render_regulatory_summary(
            segmented_df,
            base_policy_df,
            efficiency_df,
            outcome_affordability=base_outcome.affordability_rate,
            outcome_subsidy=base_outcome.subsidy_kes,
            outcome_revenue=base_outcome.utility_revenue_kes,
        )

    with tabs[4]:
        merged_df_for_export = base_policy_df.merge(
            efficiency_df[["household_id", "appliance_efficiency_score", "replacement_recommendation"]],
            on="household_id",
            how="left",
        )
        render_export(base_policy_df, nilm_df, merged_df_for_export)


if __name__ == "__main__":
    main()
