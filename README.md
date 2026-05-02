# EquiMeter AI

EquiMeter AI is a two-layer decision-support system for EPRA Hackathon 2026.

Architect and Developer IMBEKA MUSA

## What This Prototype Includes

- Layer 1 Equity Classification Engine using K-Means clustering for tariff-tier recommendations:
  - Vulnerable
  - Low-Income
  - Middle-Income
  - High-Intensity Users
- Layer 2 Appliance Efficiency Detection Module using NILM-style synthetic smart-meter traces.
- Unified Streamlit dashboard with:
  - County-level hotspot mapping
  - Household segmentation charts
  - Policy simulation controls for tariff/subsidy testing
  - Exportable CSV datasets for regulatory use

## Stack

- Python
- Streamlit
- pandas
- scikit-learn
- plotly

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Project Structure

- app.py: Integrated dashboard application
- src/data/simulate_data.py: KNBS-inspired socioeconomic and NILM synthetic data generation
- src/models/equity_classifier.py: K-Means household segmentation and tariff policy simulation
- src/models/nilm_efficiency.py: Appliance energy shares, anomaly detection, and efficiency scoring
- src/config.py: Project constants and tariff defaults

## Notes

- Data in this repository is simulated for prototype and policy sandbox use.
- Models and thresholds are configurable and intended for iterative calibration with real EPRA/KPLC field data.
