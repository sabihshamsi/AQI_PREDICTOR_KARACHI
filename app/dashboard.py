from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

from src.config import settings

st.set_page_config(page_title="Karachi AQI Dashboard", layout="wide")
st.title("Karachi AQI Prediction Dashboard")

st.markdown("""
<style>
.aqi-good { color: #00e400; font-weight: bold; }
.aqi-moderate { color: #e6e600; font-weight: bold; }
.aqi-sensitive { color: #ff7e00; font-weight: bold; }
.aqi-unhealthy { color: #ff0000; font-weight: bold; }
.aqi-very-unhealthy { color: #8f3f97; font-weight: bold; }
.aqi-hazardous { color: #7e0023; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def call_api(path: str, params: dict | None = None):
    base = settings.prediction_api_url.replace("/predict-latest", "")
    response = requests.get(f"{base}{path}", params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def get_aqi_class(category: str) -> str:
    return {
        "Good": "aqi-good",
        "Moderate": "aqi-moderate",
        "Unhealthy for Sensitive Groups": "aqi-sensitive",
        "Unhealthy": "aqi-unhealthy",
        "Very Unhealthy": "aqi-very-unhealthy",
        "Hazardous": "aqi-hazardous",
    }.get(category, "")


def aqi_band_color(aqi: float) -> str:
    if aqi <= 50:   return "#00e400"
    if aqi <= 100:  return "#e6e600"
    if aqi <= 150:  return "#ff7e00"
    if aqi <= 200:  return "#ff0000"
    if aqi <= 300:  return "#8f3f97"
    return "#7e0023"


# ---------------------------------------------------------------------------
# Fetch data once
# ---------------------------------------------------------------------------

predict_data = None
predict_error = None
try:
    predict_data = call_api("/predict-latest")
except Exception as exc:
    predict_error = exc

hist_df = pd.DataFrame()
try:
    rows = call_api("/history", params={"limit": 14})
    hist_df = pd.DataFrame(rows)
    if not hist_df.empty:
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df = hist_df.sort_values("date").reset_index(drop=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Section 1 — Current AQI + 3-day forecast cards
# ---------------------------------------------------------------------------

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current AQI")
    try:
        if predict_data is None:
            raise RuntimeError(predict_error)
        latest = predict_data["latest_actual"]
        st.metric("Current AQI", f"{latest['aqi']}")
        st.markdown(
            f"<span class='{get_aqi_class(latest['category'])}'>{latest['category']}</span>",
            unsafe_allow_html=True,
        )
        if latest["aqi"] >= 301:
            st.error("⚠️ Hazardous — avoid all outdoor activity")
        elif latest["aqi"] >= 201:
            st.error("⚠️ Very Unhealthy — health alert for everyone")
        elif latest["aqi"] >= 151:
            st.warning("⚠️ Unhealthy — sensitive groups stay indoors")
        elif latest["aqi"] >= 101:
            st.warning("Moderate–Unhealthy for sensitive groups")
        st.caption(f"PM2.5: {latest['pm25']} µg/m³  |  {latest['date']}")
    except Exception as exc:
        st.error(f"Could not fetch current data: {exc}")

with col2:
    st.subheader("3-Day Forecast")
    st.caption("Updates every hour from the feature pipeline.")
    try:
        if predict_data is None:
            raise RuntimeError(predict_error)
        predictions = predict_data["predictions"]
        cols = st.columns(3)
        for i, pred in enumerate(predictions):
            with cols[i]:
                st.markdown(f"**{pd.to_datetime(pred['date']).strftime('%a %b %d')}**")
                st.metric("AQI", f"{pred['aqi']}")
                cat = pred["category"]
                st.markdown(
                    f"<span class='{get_aqi_class(cat)}'>{cat}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"PM2.5: {pred['pm25']} µg/m³")
    except Exception as exc:
        st.error(f"Could not fetch forecast: {exc}")


# ---------------------------------------------------------------------------
# Section 2 — Combined real-time + forecast chart (the key chart)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("AQI — Last 14 Days + 3-Day Forecast")
st.caption(
    "Solid line = historical actuals from the feature store. "
    "Dashed line = model forecast. Shaded band = AQI category zone."
)

try:
    if predict_data is None:
        raise RuntimeError(predict_error)
    if hist_df.empty:
        raise RuntimeError("No historical data returned from /history.")

    predictions = predict_data["predictions"]

    # Build forecast dataframe — connect it to the last historical point
    last_actual_row = hist_df.iloc[-1]
    forecast_rows = [
        {"date": last_actual_row["date"], "aqi": last_actual_row["aqi"], "type": "forecast"}
    ]
    for p in predictions:
        forecast_rows.append({
            "date": pd.to_datetime(p["date"]),
            "aqi": p["aqi"],
            "type": "forecast",
            "category": p["category"],
            "pm25": p["pm25"],
        })
    forecast_df = pd.DataFrame(forecast_rows)

    fig = go.Figure()

    # --- AQI category background bands ---
    band_specs = [
        (0,   50,  "rgba(0,228,0,0.07)",    "Good"),
        (50,  100, "rgba(230,230,0,0.07)",   "Moderate"),
        (100, 150, "rgba(255,126,0,0.07)",   "Sensitive"),
        (150, 200, "rgba(255,0,0,0.07)",     "Unhealthy"),
        (200, 300, "rgba(143,63,151,0.07)",  "Very Unhealthy"),
        (300, 500, "rgba(126,0,35,0.07)",    "Hazardous"),
    ]
    for y0, y1, fill, label in band_specs:
        fig.add_hrect(
            y0=y0, y1=y1,
            fillcolor=fill,
            line_width=0,
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color="rgba(120,120,120,0.7)",
        )

    # --- Historical line ---
    fig.add_trace(go.Scatter(
        x=hist_df["date"],
        y=hist_df["aqi"],
        mode="lines+markers",
        name="Historical AQI",
        line=dict(color="#378ADD", width=2.5),
        marker=dict(
            size=7,
            color=[aqi_band_color(v) for v in hist_df["aqi"]],
            line=dict(color="#378ADD", width=1.5),
        ),
        hovertemplate="<b>%{x|%b %d}</b><br>AQI: %{y}<extra>Actual</extra>",
    ))

    # --- Forecast dashed line ---
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["aqi"],
        mode="lines+markers",
        name="Forecast AQI",
        line=dict(color="#EF9F27", width=2.5, dash="dash"),
        marker=dict(
            size=9,
            symbol="diamond",
            color=[aqi_band_color(v) for v in forecast_df["aqi"]],
            line=dict(color="#EF9F27", width=1.5),
        ),
        hovertemplate="<b>%{x|%b %d}</b><br>AQI: %{y}<extra>Forecast</extra>",
    ))

    # --- Vertical "today" divider line ---
    today_str = str(last_actual_row["date"].date()) if hasattr(last_actual_row["date"], "date") else str(last_actual_row["date"])
    fig.add_vline(
        x=last_actual_row["date"].timestamp() * 1000,
        line_dash="dot",
        line_color="rgba(150,150,150,0.6)",
        line_width=1.5,
        annotation_text="Today",
        annotation_position="top",
        annotation_font_size=11,
        annotation_font_color="rgba(150,150,150,0.9)",
    )

    fig.update_layout(
        title=dict(text="Karachi AQI — Historical & Forecast", font_size=15),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(150,150,150,0.1)",
            tickformat="%b %d",
        ),
        yaxis=dict(
            title="Air Quality Index (AQI)",
            showgrid=True,
            gridcolor="rgba(150,150,150,0.1)",
            range=[0, max(
                hist_df["aqi"].max() * 1.2,
                forecast_df["aqi"].max() * 1.2,
                120,
            )],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
        ),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=120, t=60, b=10),
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast summary table below the chart
    forecast_table = pd.DataFrame([
        {
            "Date": pd.to_datetime(p["date"]).strftime("%A, %b %d"),
            "Forecast AQI": p["aqi"],
            "Category": p["category"],
            "PM2.5 (µg/m³)": p["pm25"],
        }
        for p in predictions
    ])
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

except Exception as exc:
    st.error(f"Could not render combined chart: {exc}")


# ---------------------------------------------------------------------------
# Section 3 — 90-day historical trend + distribution
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Historical AQI — Last 90 Days")

try:
    rows90 = call_api("/history", params={"limit": 90})
    df90 = pd.DataFrame(rows90)

    if not df90.empty:
        df90["date"] = pd.to_datetime(df90["date"])

        fig_hist = px.line(
            df90, x="date", y="aqi",
            title="90-Day AQI Trend",
            labels={"aqi": "Air Quality Index", "date": "Date"},
        )
        fig_hist.update_traces(mode="lines+markers", marker=dict(size=4))
        for y_val, color, label in [
            (50,  "#00e400", "Good"),
            (100, "#cccc00", "Moderate"),
            (150, "#ff7e00", "Sensitive"),
            (200, "#ff0000", "Unhealthy"),
            (300, "#8f3f97", "Very Unhealthy"),
        ]:
            fig_hist.add_hline(
                y=y_val, line_dash="dash", line_color=color,
                annotation_text=label, annotation_position="top right",
            )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Category Distribution")
            cat_counts = df90["category"].value_counts()
            fig_pie = px.pie(
                values=cat_counts.values, names=cat_counts.index,
                title="AQI Categories — Last 90 Days",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_b:
            st.subheader("PM2.5 Distribution")
            fig_hist2 = px.histogram(
                df90, x="pm25", nbins=30,
                title="PM2.5 Histogram",
                labels={"pm25": "PM2.5 (µg/m³)"},
            )
            fig_hist2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_hist2, use_container_width=True)
    else:
        st.info("No historical data available.")

except Exception as exc:
    st.error(f"Could not fetch 90-day history: {exc}")

# ---------------------------------------------------------------------------
# Section 5 — Model info + AQI scale reference
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Model Information")
if predict_data:
    st.info(
        f"**Best models per horizon:** {predict_data['model_name']}  |  "
        f"**Model version:** {predict_data['model_version']}  |  "
        f"**Feature date:** {predict_data['prediction_feature_date']}"
    )
else:
    st.info("Model information unavailable.")

st.divider()
st.subheader("AQI Scale Reference")
st.table(pd.DataFrame({
    "AQI Range":  ["0–50", "51–100", "101–150", "151–200", "201–300", "301–500"],
    "Category":   ["Good", "Moderate", "Unhealthy for Sensitive Groups",
                   "Unhealthy", "Very Unhealthy", "Hazardous"],
    "Who is affected": [
        "No health impact",
        "Unusually sensitive individuals",
        "Sensitive groups (elderly, children, asthma)",
        "Everyone may begin to experience effects",
        "Health alert — everyone affected",
        "Health emergency — entire population",
    ],
    "": ["🟢", "🟡", "🟠", "🔴", "🟣", "🟤"],
}))