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


def call_api(path: str, params: dict | None = None):
    url = settings.prediction_api_url.replace("/predict-latest", path)
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


col1, col2 = st.columns(2)

with col1:
    st.subheader("Latest Prediction")
    try:
        latest = call_api("/predict-latest")
        st.metric("Predicted PM2.5", f"{latest['prediction_pm2_5']:.2f}")
        st.caption(
            f"Model: {latest['model_name']} | Date: {latest['date']} | Actual: {latest['latest_actual_pm2_5']:.2f}"
        )
    except Exception as exc:
        st.error(f"Could not fetch latest prediction: {exc}")

with col2:
    st.subheader("Historical PM2.5 (from Feature Store)")
    try:
        rows = call_api("/history", params={"limit": 90})
        df = pd.DataFrame(rows)
        if not df.empty and "date" in df.columns and "pm2_5" in df.columns:
            fig = px.line(df, x="date", y="pm2_5", title="Last 90 Days PM2.5")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data available.")
    except Exception as exc:
        st.error(f"Could not fetch history: {exc}")
