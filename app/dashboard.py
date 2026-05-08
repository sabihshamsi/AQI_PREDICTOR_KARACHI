from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.config import settings

st.set_page_config(page_title="Karachi AQI Dashboard", layout="wide")
st.title("Karachi AQI Prediction Dashboard")

# Custom CSS for AQI colors
st.markdown("""
<style>
.aqi-good { color: #00e400; font-weight: bold; }
.aqi-moderate { color: #ffff00; font-weight: bold; }
.aqi-sensitive { color: #ff7e00; font-weight: bold; }
.aqi-unhealthy { color: #ff0000; font-weight: bold; }
.aqi-very-unhealthy { color: #8f3f97; font-weight: bold; }
.aqi-hazardous { color: #7e0023; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def call_api(path: str, params: dict | None = None):
    url = settings.prediction_api_url.replace("/predict-latest", path)
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_aqi_class(category: str) -> str:
    """Get CSS class for AQI category"""
    category_map = {
        "Good": "aqi-good",
        "Moderate": "aqi-moderate",
        "Unhealthy for Sensitive Groups": "aqi-sensitive",
        "Unhealthy": "aqi-unhealthy",
        "Very Unhealthy": "aqi-very-unhealthy",
        "Hazardous": "aqi-hazardous"
    }
    return category_map.get(category, "")


# Main dashboard
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Latest Actual AQI")
    try:
        data = call_api("/predict-latest")
        latest = data["latest_actual"]

        st.metric("Current AQI", f"{latest['aqi']}")
        st.markdown(f"<span class='{get_aqi_class(latest['category'])}'>{latest['category']}</span>", unsafe_allow_html=True)
        st.caption(f"PM2.5: {latest['pm25']} µg/m³ | Date: {latest['date']}")

    except Exception as exc:
        st.error(f"Could not fetch latest data: {exc}")

with col2:
    st.subheader("AQI Predictions (Next 3 Days)")
    st.caption("⚠️ Future predictions are based on current weather conditions")
    try:
        data = call_api("/predict-latest")
        predictions = data["predictions"]

        # Create cards for each prediction
        cols = st.columns(3)
        for i, pred in enumerate(predictions):
            with cols[i]:
                st.markdown(f"**{pd.to_datetime(pred['date']).strftime('%a %m/%d')}**")
                st.metric("AQI", f"{pred['aqi']}")
                st.markdown(f"<span class='{get_aqi_class(pred['category'])}'>{pred['category'][:15]}...</span>", unsafe_allow_html=True)
                st.caption(f"PM2.5: {pred['pm25']}")

    except Exception as exc:
        st.error(f"Could not fetch predictions: {exc}")

# Historical AQI Chart
st.subheader("Historical AQI (Last 90 Days)")
try:
    rows = call_api("/history", params={"limit": 90})
    df = pd.DataFrame(rows)

    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])

        # Create AQI over time chart
        fig = px.line(df, x="date", y="aqi",
                     title="Karachi AQI Trend",
                     labels={"aqi": "Air Quality Index", "date": "Date"})

        # Add color coding based on AQI categories
        fig.update_traces(mode='lines+markers')

        # Add horizontal lines for AQI breakpoints
        fig.add_hline(y=50, line_dash="dash", line_color="#00e400", annotation_text="Good")
        fig.add_hline(y=100, line_dash="dash", line_color="#ffff00", annotation_text="Moderate")
        fig.add_hline(y=150, line_dash="dash", line_color="#ff7e00", annotation_text="Unhealthy for Sensitive")
        fig.add_hline(y=200, line_dash="dash", line_color="#ff0000", annotation_text="Unhealthy")
        fig.add_hline(y=300, line_dash="dash", line_color="#8f3f97", annotation_text="Very Unhealthy")

        st.plotly_chart(fig, use_container_width=True)

        # AQI Category Distribution
        st.subheader("AQI Category Distribution")
        category_counts = df['category'].value_counts()
        fig2 = px.pie(values=category_counts.values, names=category_counts.index,
                     title="AQI Categories in Last 90 Days")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No historical data available.")

except Exception as exc:
    st.error(f"Could not fetch history: {exc}")

# Model Information
st.subheader("Model Information")
try:
    data = call_api("/predict-latest")
    st.info(f"**Model:** {data['model_name']} | **Last Updated:** Today via CI/CD pipeline")
except:
    st.info("Model information unavailable")

# AQI Scale Reference
st.subheader("AQI Scale Reference")
aqi_scale = pd.DataFrame({
    "AQI Range": ["0-50", "51-100", "101-150", "151-200", "201-300", "301-500"],
    "Category": ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"],
    "Color": ["🟢", "🟡", "🟠", "🔴", "🟣", "🟤"]
})
st.table(aqi_scale)
