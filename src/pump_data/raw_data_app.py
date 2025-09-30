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
    if df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")
    else:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert("UTC")
    df = df.sort_values("timestamp_utc")
    return df


def available_metrics(df: pd.DataFrame) -> List[str]:
    candidates = [
        "Ps_kPa",
        "Pd_kPa",
        "Q_m3h",
        "I_A"
    ]
    return [c for c in candidates if c in df.columns]


def main() -> None:
    st.title("Pump telemetry explorer")

    st.sidebar.header("Data selection")
    labels = load_labels(LABELS_PATH)
    timeseries = load_timeseries(TIMESERIES_PATH)

    # Put the labels table and selection control in the sidebar so users can
    # pick devices before plotting. Use a simple st.dataframe with selection
    # enabled and return the selected rows. This avoids adding a `select`
    # column and uses the built-in selection API.
    labels_display = labels.copy()
    labels_display["pump_id"] = labels_display["pump_id"].astype(str)

    with st.sidebar:
        st.subheader("Available devices and labels")
        # Remove start/end time columns from the displayed table to keep it
        # compact; keep them in the full `labels` DataFrame for later use.
        display_labels = labels_display.drop(columns=[c for c in ("class_start_utc", "class_end_utc") if c in labels_display.columns])
        selection = st.dataframe(
            display_labels.reset_index(drop=True),
            width='stretch',
            height=300,
            key="labels_dataframe",
            on_select="rerun",
            selection_mode="multi-row",
        )

    # Derive selected pump ids from the dataframe selection.
    selected_pumps = []
    try:
        # When on_select != "ignore", st.dataframe returns a dict-like
        # DataframeState. The selected row indices live at
        # `selection.selection.rows` (attribute style) or
        # `selection["selection"]["rows"]` (mapping style).
        sel_obj = getattr(selection, "selection", None) or selection.get("selection", None)  # type: ignore[attr-defined]
        sel_rows = []
        if sel_obj is not None:
            sel_rows = getattr(sel_obj, "rows", None) or sel_obj.get("rows", [])  # type: ignore[attr-defined]

        if sel_rows:
            # Use integer positions against the *displayed* dataframe with a
            # reset index to match Streamlit's selection indices.
            selected_pumps = (
                display_labels.reset_index(drop=True)
                .iloc[sel_rows]["pump_id"].astype(str).tolist()
            )
    except Exception:
        selected_pumps = []
    # No fallback — require users to select pumps via the dataframe selection.

    # Limit to two devices maximum and notify the user if more were selected
    if len(selected_pumps) > 2:
        st.warning("More than 2 pumps selected — only the first two will be shown.")
        selected_pumps = selected_pumps[:2]

    st.sidebar.markdown("---")

    # Let users pick a date window but keep the underlying dataset intact so
    # future interactions (e.g. different pumps) stay responsive.
    default_start = pd.Timestamp("2025-01-01", tz="UTC")
    default_end = pd.Timestamp("2025-01-31", tz="UTC")
    ts_min = timeseries["timestamp_utc"].min()
    ts_max = timeseries["timestamp_utc"].max()

    if pd.isna(ts_min) or pd.isna(ts_max):
        st.error("No telemetry data available to plot.")
        return

    min_date = ts_min.tz_convert("UTC") if ts_min.tz is not None else ts_min.tz_localize("UTC")
    max_date = ts_max.tz_convert("UTC") if ts_max.tz is not None else ts_max.tz_localize("UTC")

    start_value = max(default_start, min_date)
    end_value = min(default_end, max_date)

    date_selection = st.sidebar.date_input(
        "Date range",
        value=(start_value.date(), end_value.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    if not isinstance(date_selection, (list, tuple)) or len(date_selection) != 2:
        st.warning("Select both a start and end date.")
        return

    start_date, end_date = date_selection

    if start_date > end_date:
        st.warning("Start date must be on or before end date.")
        return

    filter_start = pd.Timestamp(start_date, tz="UTC")
    filter_end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)

    metrics = available_metrics(timeseries)
    selected_metrics = st.sidebar.multiselect("Metrics to plot", metrics, default=metrics)

    if not selected_metrics:
        st.warning("Choose at least one metric to plot.")
        return

    if not selected_pumps:
        st.info("No pumps selected. Choose pumps in the table or sidebar to begin.")
        return

    # Show selected devices and their label info inside an expander so the raw
    # dataframe is hidden by default per project conventions. Drop the
    # start/end time columns from the displayed table per user preference.
    sel_labels = labels[labels["pump_id"].astype(str).isin(selected_pumps)].copy()
    with st.expander("Selected device labels (expand to view)"):
        if sel_labels.empty:
            st.info("No label records found for the selected pump(s).")
        else:
            sel_display = sel_labels.reset_index(drop=True)
            sel_display = sel_display.drop(columns=[c for c in ("class_start_utc", "class_end_utc") if c in sel_display.columns])
            st.dataframe(sel_display, width='stretch')

    # Prepare two columns for side-by-side plotting — fill with empty strings if
    # fewer than two pumps were selected.
    pumps_to_plot = selected_pumps[:2] + [""] * max(0, 2 - len(selected_pumps))

    # Use a fragment so plots can rerun independently and avoid a full-app
    # rerender when interacting with widgets inside the fragment.
    @st.fragment
    def render_metric_fragment(metric: str, pumps: list[str]):
        st.subheader(metric)
        col_left, col_right = st.columns(2)

        for col, pump_id in zip((col_left, col_right), pumps):
            if not pump_id:
                col.info("(no pump selected)")
                continue

            pump_df = timeseries[timeseries["pump_id"].astype(str) == str(pump_id)].copy()
            if pump_df.empty:
                col.info(f"No telemetry available for pump {pump_id}.")
                continue

            pump_df = pump_df.set_index("timestamp_utc")
            if metric not in pump_df.columns:
                col.info(f"Metric {metric} not available for pump {pump_id}.")
                continue

            window_df = pump_df.loc[(pump_df.index >= filter_start) & (pump_df.index < filter_end)]
            if window_df.empty:
                col.info(f"No data between {start_date} and {end_date} for pump {pump_id}.")
                continue

            # Ensure Plotly uses a dark template to match user preference.
            fig = px.line(window_df, x=window_df.index, y=metric, title=f"{metric} — {pump_id}", template="plotly_dark")
            # Provide a distinct key per metric/pump to avoid component collisions
            # when multiple plots exist on the page.
            col.plotly_chart(fig, key=f"plot-{metric}-{pump_id}", use_container_width=True)

    for metric in selected_metrics:
        render_metric_fragment(metric, pumps_to_plot)


# uv run python -m streamlit run src/pump_data/raw_data_app.py
if __name__ == "__main__":
    main()
