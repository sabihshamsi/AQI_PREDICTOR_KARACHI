from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

from src.config import settings
from src.hopsworks_utils import login_to_hopsworks, read_features

app = FastAPI(title="Karachi AQI Prediction API")


def calculate_aqi(pm25: float) -> dict:
    """Calculate AQI from PM2.5 using EPA formula"""
    if pm25 <= 12.0:
        aqi = (50 / 12.0) * pm25
        category = "Good"
        color = "#00e400"
    elif pm25 <= 35.4:
        aqi = ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
        category = "Moderate"
        color = "#ffff00"
    elif pm25 <= 55.4:
        aqi = ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
        category = "Unhealthy for Sensitive Groups"
        color = "#ff7e00"
    elif pm25 <= 150.4:
        aqi = ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
        category = "Unhealthy"
        color = "#ff0000"
    elif pm25 <= 250.4:
        aqi = ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201
        category = "Very Unhealthy"
        color = "#8f3f97"
    else:
        aqi = ((500 - 301) / (500.4 - 250.5)) * (pm25 - 250.5) + 301
        category = "Hazardous"
        color = "#7e0023"

    return {
        "aqi": round(aqi),
        "category": category,
        "color": color,
        "pm25": round(pm25, 2)
    }


def load_latest_registered_model():
    project = login_to_hopsworks()
    mr = project.get_model_registry()
    model = mr.get_model(settings.model_name, version=settings.model_version)
    local_dir = Path(tempfile.mkdtemp(prefix="aqi_model_"))
    model_dir = Path(model.download(local_path=str(local_dir)))
    payload = joblib.load(model_dir / "model.joblib")
    return payload


def predict_for_date(model_payload, target_date: datetime, latest_features: pd.DataFrame):
    """Predict PM2.5 for a specific date using latest available features"""
    framework = model_payload.get("framework", "sklearn")
    feature_cols = model_payload["features"]

    # Use the most recent features as proxy for future predictions
    # In a real system, you'd have weather forecasts for future dates
    X = latest_features[feature_cols].iloc[-1:].copy()

    if framework == "sklearn":
        model = model_payload["model"]
        pred = float(model.predict(X)[0])
    elif framework == "tensorflow":
        import tensorflow as tf
        model = tf.keras.models.load_model(model_payload["model_path"])
        transformed = model_payload["scaler"].transform(model_payload["imputer"].transform(X))
        pred = float(model.predict(transformed, verbose=0).ravel()[0])
    elif framework == "pytorch":
        import torch
        import torch.nn as nn

        transformed = model_payload["scaler"].transform(model_payload["imputer"].transform(X))
        transformed = transformed.astype(np.float32)
        state_path = model_payload["model_state_path"]

        model = nn.Sequential(
            nn.Linear(transformed.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            pred = float(model(torch.from_numpy(transformed)).numpy().ravel()[0])
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    return pred


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict-latest")
def predict_latest():
    payload = load_latest_registered_model()
    model_name = payload["model_name"]

    # Get latest features
    df = read_features().sort_values("date")
    latest = df.iloc[-1:]

    # Predict for today and next 3 days
    today = datetime.now().date()
    predictions = []

    for days_ahead in range(4):  # Today + 3 days
        pred_date = today + timedelta(days=days_ahead)
        pm25_pred = predict_for_date(payload, pred_date, latest)
        aqi_info = calculate_aqi(pm25_pred)

        predictions.append({
            "date": str(pred_date),
            "pm25": aqi_info["pm25"],
            "aqi": aqi_info["aqi"],
            "category": aqi_info["category"],
            "color": aqi_info["color"]
        })

    # Get actual latest PM2.5 for comparison
    latest_actual = float(latest[settings.target_column].iloc[0])
    latest_aqi = calculate_aqi(latest_actual)

    return {
        "model_name": model_name,
        "predictions": predictions,
        "latest_actual": {
            "date": str(latest["date"].iloc[0]),
            "pm25": latest_aqi["pm25"],
            "aqi": latest_aqi["aqi"],
            "category": latest_aqi["category"],
            "color": latest_aqi["color"]
        }
    }


@app.get("/history")
def history(limit: int = 30):
    df = read_features().sort_values("date").tail(limit)

    # Add AQI calculations to historical data
    history_data = []
    for _, row in df.iterrows():
        pm25 = float(row[settings.target_column])
        aqi_info = calculate_aqi(pm25)
        history_data.append({
            "date": str(row["date"]),
            "pm25": aqi_info["pm25"],
            "aqi": aqi_info["aqi"],
            "category": aqi_info["category"],
            "color": aqi_info["color"]
        })

    return history_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
