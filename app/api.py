from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
from fastapi import FastAPI

from src.config import settings
from src.hopsworks_utils import login_to_hopsworks, read_features

app = FastAPI(title="Karachi AQI Prediction API")


def load_latest_registered_model():
    project = login_to_hopsworks()
    mr = project.get_model_registry()
    model = mr.get_model(settings.model_name, version=settings.model_version)
    local_dir = Path(tempfile.mkdtemp(prefix="aqi_model_"))
    model_dir = Path(model.download(local_path=str(local_dir)))
    payload = joblib.load(model_dir / "model.joblib")
    return payload


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict-latest")
def predict_latest():
    payload = load_latest_registered_model()
    framework = payload.get("framework", "sklearn")
    feature_cols = payload["features"]
    model_name = payload["model_name"]
    df = read_features().sort_values("date")
    latest = df.iloc[-1:]
    X = latest[feature_cols]
    if framework == "sklearn":
        model = payload["model"]
        pred = float(model.predict(X)[0])
    elif framework == "tensorflow":
        import tensorflow as tf

        model = tf.keras.models.load_model(payload["model_path"])
        transformed = payload["scaler"].transform(payload["imputer"].transform(X))
        pred = float(model.predict(transformed, verbose=0).ravel()[0])
    elif framework == "pytorch":
        import numpy as np
        import torch
        import torch.nn as nn

        transformed = payload["scaler"].transform(payload["imputer"].transform(X))
        transformed = transformed.astype(np.float32)
        state_path = payload["model_state_path"]

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
        raise ValueError(f"Unsupported framework in registry payload: {framework}")

    return {
        "model_name": model_name,
        "date": str(latest["date"].iloc[0]),
        "prediction_pm2_5": pred,
        "latest_actual_pm2_5": float(latest[settings.target_column].iloc[0]),
    }


@app.get("/history")
def history(limit: int = 30):
    df = read_features().sort_values("date").tail(limit)
    return df.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
