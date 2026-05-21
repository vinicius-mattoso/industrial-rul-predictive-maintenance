from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from run_app import DATA_SERVICE, FEATURE_SENSORS


st.set_page_config(
    page_title="Industrial Health Analytics",
    page_icon="",
    layout="wide",
)


STATUS_COLORS = {
    "Healthy": "#1f9d68",
    "Warning": "#d98b16",
    "Critical": "#d84444",
}
ZONE_COLORS = {
    "Healthy": "rgba(31, 157, 104, 0.14)",
    "Warning": "rgba(217, 139, 22, 0.14)",
    "Critical": "rgba(216, 68, 68, 0.16)",
}


@st.cache_data(show_spinner=False)
def get_engine_ids() -> list[int]:
    return DATA_SERVICE.engine_ids()


@st.cache_data(show_spinner=False)
def get_engine_frame(engine_id: int) -> pd.DataFrame:
    return DATA_SERVICE.engine_frame(engine_id)


def status_color(status: str) -> str:
    return STATUS_COLORS.get(status, "#667386")


def sensor_envelope(engine_df: pd.DataFrame, sensor: str) -> dict[str, float]:
    healthy_reference = engine_df[engine_df["health_state_rule"] == "Healthy"][sensor]
    if healthy_reference.empty:
        healthy_reference = engine_df[sensor]
    return {
        "mean": float(healthy_reference.mean()),
        "p05": float(healthy_reference.quantile(0.05)),
        "p95": float(healthy_reference.quantile(0.95)),
        "min": float(healthy_reference.min()),
        "max": float(healthy_reference.max()),
    }


def zone_segments(engine_df: pd.DataFrame) -> list[dict[str, int | str]]:
    ordered = engine_df.sort_values("time_in_cycles").copy()
    ordered["ui_status"] = ordered["health_state_rule"].replace({"Degrading": "Warning"})
    ordered["segment"] = (ordered["ui_status"] != ordered["ui_status"].shift()).cumsum()
    segments = []
    for _, segment_df in ordered.groupby("segment"):
        segments.append(
            {
                "status": str(segment_df["ui_status"].iloc[0]),
                "start": int(segment_df["time_in_cycles"].min()),
                "end": int(segment_df["time_in_cycles"].max()),
            }
        )
    return segments


def plot_sensor_operational_envelope(engine_df: pd.DataFrame, sensor: str, current_cycle: int) -> go.Figure:
    envelope = sensor_envelope(engine_df, sensor)
    fig = go.Figure()

    shown_statuses = set()
    for segment in zone_segments(engine_df):
        status = str(segment["status"])
        show_legend = status not in shown_statuses
        shown_statuses.add(status)
        fig.add_vrect(
            x0=segment["start"],
            x1=segment["end"],
            fillcolor=ZONE_COLORS.get(status, "rgba(102, 115, 134, 0.10)"),
            line_width=0,
            layer="below",
            name=f"{status} zone",
            showlegend=show_legend,
        )

    fig.add_trace(
        go.Scatter(
            x=engine_df["time_in_cycles"],
            y=engine_df[sensor],
            mode="lines",
            name=f"Engine {int(engine_df['unit_number'].iloc[0])} - {sensor}",
            line=dict(color="#111111", width=2),
        )
    )

    reference_lines = [
        ("Healthy mean", envelope["mean"], "#1d2bff", "solid", 3),
        ("Healthy P05", envelope["p05"], "#ff9900", "dash", 2),
        ("Healthy P95", envelope["p95"], "#ff9900", "dash", 2),
        ("Healthy min", envelope["min"], "#7a7a7a", "dashdot", 2),
        ("Healthy max", envelope["max"], "#7a7a7a", "dashdot", 2),
    ]
    for label, value, color, dash, width in reference_lines:
        fig.add_hline(
            y=value,
            line_color=color,
            line_dash=dash,
            line_width=width,
            annotation_text=label,
            annotation_position="right",
        )

    fig.add_vline(
        x=current_cycle,
        line_width=2,
        line_dash="dot",
        line_color="#17212f",
        annotation_text="selected cycle",
        annotation_position="top",
    )

    fig.update_layout(
        title=f"{sensor} - Engine {int(engine_df['unit_number'].iloc[0])} vs own healthy operational envelope",
        height=620,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="v", x=1.02, y=1),
        xaxis_title="Cycles",
        yaxis_title=sensor,
    )
    return fig


def sensor_deviation_table(engine_df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    records = []
    for sensor in FEATURE_SENSORS:
        envelope = sensor_envelope(engine_df, sensor)
        value = float(row[sensor])
        band_width = max(envelope["p95"] - envelope["p05"], 0.001)
        if value < envelope["p05"]:
            direction = "below P05"
            deviation = (envelope["p05"] - value) / band_width
        elif value > envelope["p95"]:
            direction = "above P95"
            deviation = (value - envelope["p95"]) / band_width
        else:
            direction = "inside P05-P95"
            deviation = 0.0

        records.append(
            {
                "sensor": sensor,
                "current_value": round(value, 4),
                "healthy_mean": round(envelope["mean"], 4),
                "healthy_p05": round(envelope["p05"], 4),
                "healthy_p95": round(envelope["p95"], 4),
                "direction": direction,
                "deviation_score": round(float(deviation), 4),
            }
        )
    return pd.DataFrame(records).sort_values("deviation_score", ascending=False)


def plot_prediction_probabilities(probabilities: dict[str, float] | None) -> go.Figure | None:
    if not probabilities:
        return None
    probability_df = pd.DataFrame(
        [{"status": key, "probability": value} for key, value in probabilities.items()]
    )
    fig = px.bar(
        probability_df,
        x="status",
        y="probability",
        color="status",
        color_discrete_map=STATUS_COLORS,
        labels={"status": "Status", "probability": "Probability"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(height=300, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def plot_feature_importance() -> go.Figure | None:
    model = DATA_SERVICE.model
    if model is None or not hasattr(model, "feature_importances_") or not DATA_SERVICE.model_features:
        return None

    importance_df = pd.DataFrame(
        {
            "feature": DATA_SERVICE.model_features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).head(20)

    fig = px.bar(
        importance_df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def render_status_badge(status: str, confidence: float | None) -> None:
    confidence_text = "fallback rule" if confidence is None else f"{confidence:.1%} confidence"
    st.markdown(
        f"""
        <div style="
            border: 1px solid {status_color(status)};
            border-left: 7px solid {status_color(status)};
            border-radius: 8px;
            padding: 16px;
            background: rgba(31, 157, 104, 0.04);
        ">
            <div style="font-size: 0.82rem; color: #667386; font-weight: 700;">PREDICTED HEALTH STATUS</div>
            <div style="font-size: 2rem; font-weight: 800; color: {status_color(status)};">{status}</div>
            <div style="color: #667386;">{confidence_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("Industrial Health Status Analytics")
st.caption("Use the Engine and Sensor filters to investigate which signal is driving the health status for each asset.")

if DATA_SERVICE.model is None:
    st.warning(DATA_SERVICE.model_message)
else:
    st.success(DATA_SERVICE.model_message)

engine_ids = get_engine_ids()

with st.sidebar:
    st.header("Investigation filters")
    engine_id = st.selectbox(
        "Engine",
        options=engine_ids,
        index=0,
        format_func=lambda value: f"Engine {value:03d}",
    )

    engine_df = get_engine_frame(engine_id)
    min_cycle = int(engine_df["time_in_cycles"].min())
    max_cycle = int(engine_df["time_in_cycles"].max())
    default_cycle = min(max(85, min_cycle), max_cycle)
    default_row = DATA_SERVICE.row_at_engine_cycle(engine_id, default_cycle)
    default_critical_sensor = DATA_SERVICE.critical_sensor_for_row(default_row)

    selected_sensor = st.selectbox(
        "Sensor",
        options=FEATURE_SENSORS,
        index=FEATURE_SENSORS.index(default_critical_sensor) if default_critical_sensor in FEATURE_SENSORS else 0,
        help="Default is the sensor with the largest deviation at the initial inspection cycle.",
    )

    selected_cycle = st.slider(
        "Inspection cycle",
        min_value=min_cycle,
        max_value=max_cycle,
        value=default_cycle,
        step=1,
    )

    st.divider()
    st.metric("Total engines", len(engine_ids))
    st.metric("Model features", len(DATA_SERVICE.model_features))

row = DATA_SERVICE.row_at_engine_cycle(engine_id, selected_cycle)
prediction = DATA_SERVICE.prediction_details_for_row(row)
feature_vector = DATA_SERVICE.feature_vector_for_row(row)
critical_sensor = DATA_SERVICE.critical_sensor_for_row(row)

top_cols = st.columns([1.15, 0.85, 0.85, 0.85])
with top_cols[0]:
    render_status_badge(prediction["status"], prediction["confidence"])
with top_cols[1]:
    st.metric("RUL capped", f"{float(row['RUL_capped']):.0f} cycles")
with top_cols[2]:
    st.metric("Raw RUL", f"{float(row['RUL']):.0f} cycles")
with top_cols[3]:
    st.metric("Critical sensor", critical_sensor)

tabs = st.tabs(
    [
        "Sensor behavior",
        "Model input",
        "Prediction details",
        "Simulation",
        "Fleet overview",
    ]
)

with tabs[0]:
    st.subheader(f"Engine {engine_id:03d} - {selected_sensor} operational envelope")
    st.plotly_chart(
        plot_sensor_operational_envelope(engine_df, selected_sensor, selected_cycle),
        width="stretch",
    )

    st.subheader("Sensor deviation ranking")
    st.caption("Sensors are ranked by distance outside the engine's own healthy P05-P95 envelope.")
    st.dataframe(sensor_deviation_table(engine_df, row), width="stretch", hide_index=True)

with tabs[1]:
    st.subheader("Exact feature vector sent to the model")
    if feature_vector.empty:
        st.info("Model features are unavailable because the artifacts were not loaded.")
    else:
        feature_view = feature_vector.T.reset_index()
        feature_view.columns = ["feature", "value"]
        st.dataframe(feature_view, width="stretch", hide_index=True)

    st.subheader("Raw row context")
    context_columns = [
        "unit_number",
        "time_in_cycles",
        "RUL",
        "RUL_capped",
        "health_state_rule",
        *FEATURE_SENSORS,
    ]
    context_df = row[context_columns].astype(str).rename("value").reset_index()
    context_df.columns = ["field", "value"]
    st.dataframe(context_df, width="stretch", hide_index=True)

with tabs[2]:
    st.subheader("Prediction output")
    probability_fig = plot_prediction_probabilities(prediction["probabilities"])
    if probability_fig is None:
        st.info("Probability output is unavailable in fallback mode.")
    else:
        st.plotly_chart(probability_fig, width="stretch")

    st.json(
        {
            "model_available": DATA_SERVICE.model is not None,
            "predicted_status": prediction["status"],
            "confidence": prediction["confidence"],
            "fallback_rule_status": prediction["fallbackRuleStatus"],
            "model_message": DATA_SERVICE.model_message,
        }
    )

    st.subheader("Feature importance")
    importance_fig = plot_feature_importance()
    if importance_fig is None:
        st.info("Feature importance is available when the LightGBM model is loaded.")
    else:
        st.plotly_chart(importance_fig, width="stretch")

with tabs[3]:
    st.subheader("Simulate a new incoming equipment reading")
    st.caption(
        "Start from the selected engine/cycle, adjust the next sensor reading, and evaluate the resulting health status."
    )

    base_values = {sensor: float(row[sensor]) for sensor in FEATURE_SENSORS}
    with st.form("simulation_form"):
        sim_cols = st.columns(3)
        simulated_rul = sim_cols[0].number_input(
            "Simulated RUL",
            min_value=0.0,
            max_value=125.0,
            value=float(row["RUL_capped"]),
            step=1.0,
        )
        sim_cols[1].metric("Base engine", f"ENGINE-{engine_id:03d}")
        sim_cols[2].metric("Base cycle", int(row["time_in_cycles"]))

        st.markdown("#### Sensor values for the next reading")
        sensor_inputs: dict[str, float] = {}
        sensor_cols = st.columns(3)
        for index, sensor in enumerate(FEATURE_SENSORS):
            envelope = sensor_envelope(engine_df, sensor)
            current_value = base_values[sensor]
            lower = min(envelope["min"], current_value) - abs(current_value) * 0.03
            upper = max(envelope["max"], current_value) + abs(current_value) * 0.03
            sensor_inputs[sensor] = sensor_cols[index % 3].number_input(
                sensor,
                min_value=float(lower),
                max_value=float(upper),
                value=float(current_value),
                step=max(abs(current_value) * 0.001, 0.001),
                format="%.4f",
            )

        submitted = st.form_submit_button("Evaluate simulated reading", type="primary")

    if submitted:
        simulated_features = DATA_SERVICE.simulated_feature_vector(
            engine_id=engine_id,
            base_cycle=int(row["time_in_cycles"]),
            sensor_values=sensor_inputs,
            simulated_rul=simulated_rul,
        )
        simulated_prediction = DATA_SERVICE.prediction_details_for_features(
            simulated_features,
            fallback_rul=simulated_rul,
        )

        result_cols = st.columns([1, 1, 1])
        with result_cols[0]:
            render_status_badge(simulated_prediction["status"], simulated_prediction["confidence"])
        with result_cols[1]:
            st.metric("Fallback rule status", simulated_prediction["fallbackRuleStatus"])
        with result_cols[2]:
            st.metric("Changed sensors", sum(sensor_inputs[s] != base_values[s] for s in FEATURE_SENSORS))

        probability_fig = plot_prediction_probabilities(simulated_prediction["probabilities"])
        if probability_fig is not None:
            st.plotly_chart(probability_fig, width="stretch")
        else:
            st.info("Probability output is unavailable in fallback mode.")

        st.subheader("Simulated model input")
        simulated_feature_view = simulated_features.T.reset_index()
        simulated_feature_view.columns = ["feature", "value"]
        st.dataframe(simulated_feature_view, width="stretch", hide_index=True)

        st.subheader("Input deltas vs selected reading")
        delta_df = pd.DataFrame(
            [
                {
                    "sensor": sensor,
                    "base_value": round(base_values[sensor], 4),
                    "simulated_value": round(sensor_inputs[sensor], 4),
                    "delta": round(sensor_inputs[sensor] - base_values[sensor], 4),
                }
                for sensor in FEATURE_SENSORS
            ]
        )
        st.dataframe(delta_df, width="stretch", hide_index=True)
    else:
        st.info("Adjust the simulated values and click Evaluate simulated reading.")

with tabs[4]:
    st.subheader("Fleet health distribution at selected cycle")
    records = []
    for item_engine_id in engine_ids:
        item_row = DATA_SERVICE.row_at_engine_cycle(item_engine_id, selected_cycle)
        item_prediction = DATA_SERVICE.prediction_details_for_row(item_row)
        records.append(
            {
                "engine": f"ENGINE-{item_engine_id:03d}",
                "cycle": int(item_row["time_in_cycles"]),
                "rul": float(item_row["RUL_capped"]),
                "status": item_prediction["status"],
                "critical_sensor": DATA_SERVICE.critical_sensor_for_row(item_row),
            }
        )

    fleet_df = pd.DataFrame(records).sort_values("rul")
    status_counts = fleet_df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]

    chart_cols = st.columns([0.7, 1.3])
    with chart_cols[0]:
        st.plotly_chart(
            px.pie(
                status_counts,
                values="count",
                names="status",
                color="status",
                color_discrete_map=STATUS_COLORS,
                hole=0.45,
            ).update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20)),
            width="stretch",
        )
    with chart_cols[1]:
        st.plotly_chart(
            px.bar(
                fleet_df.head(20).sort_values("rul", ascending=True),
                x="rul",
                y="engine",
                color="status",
                color_discrete_map=STATUS_COLORS,
                orientation="h",
                labels={"rul": "RUL capped", "engine": "Engine"},
            ).update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20)),
            width="stretch",
        )

    st.dataframe(fleet_df, width="stretch", hide_index=True)
