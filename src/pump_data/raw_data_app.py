"""Simple Streamlit app to explore raw pump telemetry.

The app loads `pump_labels.csv` and `pump_timeseries.csv` from the package
`src/pump_data` directory and lets the user pick a `pump_id` from the labels
file. Selected pump telemetry is displayed as interactive time-series plots.

Usage: `uv run python -m streamlit run src/pump_data/raw_data_app.py`
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
LABELS_PATH = ROOT / "pump_labels.csv"
TIMESERIES_PATH = ROOT / "pump_timeseries.csv"


@st.cache_data
def load_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["class_start_utc", "class_end_utc"]) 


@st.cache_data
def load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp_utc"]) 
    df = df.sort_values("timestamp_utc")
    return df


def available_metrics(df: pd.DataFrame) -> List[str]:
    candidates = [
        "Ps_kPa",
        "Pd_kPa",
        "Q_m3h",
        "I_A",
        "dP_kPa",
        "Eff",
    ]
    return [c for c in candidates if c in df.columns]


def main() -> None:
    st.title("Pump telemetry explorer")

    st.sidebar.header("Data selection")
    labels = load_labels(LABELS_PATH)
    timeseries = load_timeseries(TIMESERIES_PATH)

    pump_ids = labels["pump_id"].astype(str).unique().tolist()
    selected = st.sidebar.selectbox("Select pump_id", pump_ids)

    st.sidebar.markdown("---")
    metrics = available_metrics(timeseries)
    selected_metrics = st.sidebar.multiselect("Metrics to plot", metrics, default=metrics[:3])

    if not selected_metrics:
        st.warning("Choose at least one metric to plot.")
        return

    st.header(f"Telemetry for pump {selected}")
    pump_df = timeseries[timeseries["pump_id"].astype(str) == str(selected)].copy()
    if pump_df.empty:
        st.info("No telemetry available for the selected pump.")
        return

    pump_df = pump_df.set_index("timestamp_utc")

    for metric in selected_metrics:
        if metric not in pump_df.columns:
            continue
        fig = px.line(pump_df, x=pump_df.index, y=metric, title=metric)
        st.plotly_chart(fig, use_container_width=True)

    st.sidebar.markdown("---")
    if st.sidebar.button("Show label record"):
        rec = labels[labels["pump_id"].astype(str) == str(selected)]
        st.sidebar.write(rec)


# uv run python -m streamlit run src/pump_data/raw_data_app.py
if __name__ == "__main__":
    main()


