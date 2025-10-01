from __future__ import annotations

from datetime import date as date_type
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pump_pipeline.factories.pipeline_factory_pump import (
    make_pump_device,
    run_pump_batch,
)
from src.pump_pipeline.pipeline_objects.data_object_pump import PumpPipelineDataObject

st.set_page_config(page_title="Pump Pipeline Explorer", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "pump_data"
LABELS_PATH = DATA_ROOT / "pump_labels.csv"

NORMALIZED_METRICS: List[str] = [
    "Ps_kPa",
    "Pd_kPa",
    "Q_m3h",
    "I_A",
    "dP_kPa",
    "Eff",
]
DEFAULT_METRICS: List[str] = ["Ps_kPa", "Pd_kPa", "Q_m3h", "I_A"]
PLOT_TEMPLATE = "plotly_dark"
MAX_SELECTED_PUMPS = 2


def load_labels(path: Path) -> pd.DataFrame:
    """
    Load pump label data from a CSV, preserving leading zeros in pump_id.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with pump_id as string and parsed date columns.
    """
    df = pd.read_csv(
        path,
        dtype={"pump_id": "string"},
        parse_dates=["class_start_utc", "class_end_utc"],
    )
    return df


def run_pipeline_for_selection(
    pump_ids: Tuple[str, ...],
    start_iso: str,
    end_iso: str,
) -> Dict[str, Dict[str, Any]]:
    if not pump_ids:
        return {}

    start_ts = pd.Timestamp(start_iso)
    end_ts = pd.Timestamp(end_iso)

    devices = [
        make_pump_device(
            device_id=str(pump_id),
            start_utc=start_ts.to_pydatetime(),
            end_utc=end_ts.to_pydatetime(),
        )
        for pump_id in pump_ids
    ]

    batch_results = run_pump_batch(
        devices,
        batch_size=max(1, len(devices)),
        io_max_workers=1,
        cpu_max_workers=1,
    )

    output: Dict[str, Dict[str, Any]] = {}
    for pump_id, dobj in batch_results.device_results.items():
        try:
            normalized = dobj.get_dataset("pump_telemetry").copy()
        except KeyError:
            normalized = pd.DataFrame()

        diff = dobj.get_artifact(PumpPipelineDataObject.ARTIFACT_DIFFERENTIAL_PRESSURE)
        eff = dobj.get_artifact(PumpPipelineDataObject.ARTIFACT_EFFICIENCY)

        diff_df = diff.copy() if isinstance(diff, pd.DataFrame) else pd.DataFrame(columns=["timestamp_utc", "dP_kPa"])
        eff_df = eff.copy() if isinstance(eff, pd.DataFrame) else pd.DataFrame(columns=["timestamp_utc", "Eff"])

        output[str(pump_id)] = {
            "normalized": normalized,
            "differential_pressure": diff_df,
            "efficiency": eff_df,
            "errors": list(dobj.errors),
            "warnings": list(dobj.warnings),
        }

    return output


def _derive_date_bounds(labels: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start_candidates = labels.get("class_start_utc")
    end_candidates = labels.get("class_end_utc")

    ts_min = start_candidates.min() if start_candidates is not None else pd.NaT
    ts_max = end_candidates.max() if end_candidates is not None else pd.NaT

    if pd.isna(ts_min):
        ts_min = pd.Timestamp("2025-01-01", tz="UTC")
    if pd.isna(ts_max):
        ts_max = pd.Timestamp("2025-03-31", tz="UTC")

    if ts_min.tzinfo is None:
        ts_min = ts_min.tz_localize("UTC")
    else:
        ts_min = ts_min.tz_convert("UTC")

    if ts_max.tzinfo is None:
        ts_max = ts_max.tz_localize("UTC")
    else:
        ts_max = ts_max.tz_convert("UTC")

    return ts_min, ts_max


def _ensure_timezone(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _to_date(ts: pd.Timestamp) -> date_type:
    ts = _ensure_timezone(ts)
    return ts.date()


def main() -> None:
    st.title("Pump pipeline explorer")

    labels = load_labels(LABELS_PATH)
    ts_min, ts_max = _derive_date_bounds(labels)

    st.sidebar.header("Data selection")
    labels_display = labels.copy()
    labels_display["pump_id"] = labels_display["pump_id"].astype(str)

    with st.sidebar:
        st.subheader("Available devices and labels")
        display_labels = labels_display.drop(
            columns=[c for c in ("class_start_utc", "class_end_utc") if c in labels_display.columns]
        )
        # Display the labels in a dataframe and allow the user to select a single row.
        selection = st.dataframe(
            display_labels.reset_index(drop=True),
            width="stretch",
            height=300,
            key="pipeline_labels_dataframe",
            on_select="rerun",
            selection_mode="single-row",
        )

    selected_pumps: List[str] = []
    try:
        sel_obj = getattr(selection, "selection", None) or selection.get("selection", None)  # type: ignore[attr-defined]
        sel_rows = []
        if sel_obj is not None:
            sel_rows = getattr(sel_obj, "rows", None) or sel_obj.get("rows", [])  # type: ignore[attr-defined]
        if sel_rows:
            selected_pumps = (
                display_labels.reset_index(drop=True)
                .iloc[sel_rows]["pump_id"].astype(str).tolist()
            )
    except Exception:
        selected_pumps = []

    st.sidebar.markdown("---")

    default_start = ts_min
    default_end = min(ts_min + pd.Timedelta(days=30), ts_max)

    date_selection = st.sidebar.date_input(
        "Date range",
        value=(default_start.date(), default_end.date()),
        min_value=_to_date(ts_min),
        max_value=_to_date(ts_max),
    )

    if not isinstance(date_selection, (list, tuple)) or len(date_selection) != 2:
        st.warning("Select both a start and end date.")
        return

    start_date, end_date = tuple(sorted(date_selection))

    if start_date > end_date:
        st.warning("Start date must be on or before end date.")
        return

    st.sidebar.markdown("---")
    selected_metrics = st.sidebar.multiselect(
        "Normalized metrics to plot",
        options=NORMALIZED_METRICS,
        default=DEFAULT_METRICS,
    )

    if not selected_metrics:
        st.warning("Choose at least one metric to plot.")
        return

    if not selected_pumps:
        st.info("No pumps selected. Choose pumps in the table to begin.")
        return

    window_start = pd.Timestamp(start_date).tz_localize("UTC")
    window_end_exclusive = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
    pipeline_end = window_end_exclusive - pd.Timedelta(milliseconds=1)

    results_by_pump = run_pipeline_for_selection(
        tuple(selected_pumps),
        window_start.isoformat(),
        pipeline_end.isoformat(),
    )

    if not results_by_pump:
        st.error("No pipeline results were generated for the selected pumps.")
        return

    normalized_by_pump: Dict[str, pd.DataFrame] = {}
    diff_by_pump: Dict[str, pd.DataFrame] = {}
    eff_by_pump: Dict[str, pd.DataFrame] = {}

    for pump_id in selected_pumps:
        pump_result = results_by_pump.get(pump_id)
        if pump_result is None:
            st.error(f"Pipeline did not return results for pump {pump_id}.")
            continue

        normalized = pump_result["normalized"].copy()
        if "timestamp_utc" not in normalized.columns:
            normalized["timestamp_utc"] = pd.NaT
        normalized["timestamp_utc"] = pd.to_datetime(normalized["timestamp_utc"], utc=True, errors="coerce")
        normalized = normalized.loc[
            (normalized["timestamp_utc"] >= window_start)
            & (normalized["timestamp_utc"] < window_end_exclusive)
        ].reset_index(drop=True)
        normalized_by_pump[pump_id] = normalized

        diff_df = pump_result["differential_pressure"].copy()
        if "timestamp_utc" not in diff_df.columns:
            diff_df["timestamp_utc"] = pd.NaT
        if not diff_df.empty:
            diff_df["timestamp_utc"] = pd.to_datetime(diff_df["timestamp_utc"], utc=True, errors="coerce")
            diff_df = diff_df.loc[
                (diff_df["timestamp_utc"] >= window_start)
                & (diff_df["timestamp_utc"] < window_end_exclusive)
            ].reset_index(drop=True)
        diff_by_pump[pump_id] = diff_df

        eff_df = pump_result["efficiency"].copy()
        if "timestamp_utc" not in eff_df.columns:
            eff_df["timestamp_utc"] = pd.NaT
        if not eff_df.empty:
            eff_df["timestamp_utc"] = pd.to_datetime(eff_df["timestamp_utc"], utc=True, errors="coerce")
            eff_df = eff_df.loc[
                (eff_df["timestamp_utc"] >= window_start)
                & (eff_df["timestamp_utc"] < window_end_exclusive)
            ].reset_index(drop=True)
        eff_by_pump[pump_id] = eff_df


    # For a single selected pump render full-width plots
    pump_to_plot = selected_pumps[0] if selected_pumps else ""

    @st.fragment
    def render_normalized_metric(metric: str, pump_id: str) -> None:
        st.subheader(f"Normalized metric: {metric}")
        if not pump_id:
            st.info("(no pump selected)")
            return
        df = normalized_by_pump.get(pump_id)
        if df is None or df.empty:
            st.info(f"No data for pump {pump_id} in the selected window.")
            return
        if metric not in df.columns:
            st.info(f"Metric {metric} not found for pump {pump_id}.")
            return
        fig = px.line(
            df,
            x="timestamp_utc",
            y=metric,
            title=f"{metric} — {pump_id}",
            template=PLOT_TEMPLATE,
        )
        st.plotly_chart(
            fig,
            key=f"normalized-{metric}-{pump_id}",
            use_container_width=True,
            config={"responsive": True},
        )

    for metric in selected_metrics:
        render_normalized_metric(metric, pump_to_plot)

    st.markdown("---")
    st.subheader("Derived artifacts")

    @st.fragment
    def render_artifact(
        title: str,
        df_map: Dict[str, pd.DataFrame],
        metric: str,
        pump_id: str,
        plot_key: str,
    ) -> None:
        st.markdown(f"**{title}**")
        if not pump_id:
            st.info("(no pump selected)")
            return
        df = df_map.get(pump_id)
        if df is None or df.empty:
            st.info(f"No {title.lower()} data for pump {pump_id}.")
            return
        # use a scatter plot for artifacts and color by recipe when available
        if "recipe" in df.columns:
            fig = px.scatter(
                df,
                x="timestamp_utc",
                y=metric,
                color="recipe",
                title=f"{title} — {pump_id}",
                template=PLOT_TEMPLATE,
            )
        else:
            fig = px.scatter(
                df,
                x="timestamp_utc",
                y=metric,
                title=f"{title} — {pump_id}",
                template=PLOT_TEMPLATE,
            )
        st.plotly_chart(
            fig,
            key=f"{plot_key}-{pump_id}",
            use_container_width=True,
            config={"responsive": True},
        )
        with st.expander(f"{title} — pump {pump_id}"):
            st.dataframe(df.head(10))

    render_artifact(
        title="Differential pressure",
        df_map=diff_by_pump,
        metric="dP_kPa",
        pump_id=pump_to_plot,
        plot_key="artifact-dp",
    )

    render_artifact(
        title="Efficiency",
        df_map=eff_by_pump,
        metric="Eff",
        pump_id=pump_to_plot,
        plot_key="artifact-eff",
    )


# uv run python -m streamlit run src/pump_pipeline/streamlit_app.py
if __name__ == "__main__":
    main()
