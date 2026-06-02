from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from time import time
from zoneinfo import ZoneInfo

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import settings
from src.hopsworks_utils import login_to_hopsworks, read_features
from src.open_meteo import KARACHI_TIMEZONE, fetch_current_open_meteo_aqi

app = FastAPI(title="Karachi AQI Prediction API")

# Reduced model TTL so a freshly trained daily model is picked up within 5 min
MODEL_CACHE_TTL_SECONDS = 300
# FIX (Bug 1): Was 3600. Must be shorter than the hourly pipeline cadence so the
# API sees fresh feature-store data within the same hour it is written.
FEATURE_CACHE_TTL_SECONDS = 300
CURRENT_AQI_CACHE_TTL_SECONDS = 3600

_model_cache: dict = {"expires_at": 0.0, "payload": None}
_features_cache: dict = {"expires_at": 0.0, "df": None}
_current_aqi_cache: dict = {"expires_at": 0.0, "payload": None}


def calculate_aqi(pm25: float) -> dict:
    """Calculate AQI from PM2.5 using EPA breakpoint formula."""
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
        "pm25": round(pm25, 2),
    }


def load_latest_registered_model():
    now = time()
    if _model_cache["payload"] is not None and _model_cache["expires_at"] > now:
        return _model_cache["payload"]

    project = login_to_hopsworks()
    mr = project.get_model_registry()
    models = mr.get_models(settings.model_name)
    if models:
        model_version = max(model.version for model in models)
    else:
        model_version = settings.model_version

    model = mr.get_model(settings.model_name, version=model_version)
    local_dir = Path(tempfile.mkdtemp(prefix="aqi_model_"))
    model_dir = Path(model.download(local_path=str(local_dir)))
    payload = joblib.load(model_dir / "model.joblib")
    payload["model_version"] = model_version

    _model_cache["payload"] = payload
    _model_cache["expires_at"] = now + MODEL_CACHE_TTL_SECONDS
    return payload


def read_features_cached() -> pd.DataFrame:
    now = time()
    if _features_cache["df"] is not None and _features_cache["expires_at"] > now:
        return _features_cache["df"].copy()

    df = read_features()
    _features_cache["df"] = df
    _features_cache["expires_at"] = now + FEATURE_CACHE_TTL_SECONDS
    return df.copy()


def predict_for_date(
    model_payload: dict,
    df: pd.DataFrame,   # FIX (Bug 2): full sorted df, not just the last row
    horizon: int,
) -> float:
    """
    Predict PM2.5 for a given forecast horizon.

    The model was trained with horizon-shifted targets: the Day-1 model saw
    row[t] → target[t+1], the Day-2 model saw row[t] → target[t+2], etc.
    At inference time we replicate that: for horizon=1 we feed the most recent
    feature row (iloc[-1]); for horizon=2 we feed iloc[-2] so the lag features
    carry the same relative offset seen during training; for horizon=3 iloc[-3].

    This means each horizon gets a *different* input row — the one whose lag
    window correctly corresponds to predicting that many days ahead — rather
    than all three horizons receiving the identical last row.
    """
    feature_cols = model_payload["features"]
    framework = model_payload.get("framework", "sklearn")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature store is missing model columns: {missing}")

    if len(df) == 0:
        raise ValueError("Feature dataframe is empty")

    # FIX: use explicit bounded indexing instead of negative slice ranges.
    # If the feature store is small, clamp to the latest available row.
    idx = len(df) - horizon
    idx = max(0, min(idx, len(df) - 1))
    X = df[feature_cols].iloc[[idx]].copy()

    if framework == "sklearn_horizons":
        horizon_models = model_payload["horizon_models"]
        horizon_payload = horizon_models.get(horizon) or horizon_models.get(str(horizon))
        if horizon_payload is None:
            raise ValueError(f"No model stored for forecast horizon {horizon}")
        return float(horizon_payload["model"].predict(X)[0])

    if framework == "sklearn":
        return float(model_payload["model"].predict(X)[0])

    raise ValueError(
        f"Unsupported framework '{framework}'. Only sklearn models are supported."
    )


def get_latest_actual(df: pd.DataFrame) -> tuple[dict, bool]:
    now = time()
    if _current_aqi_cache["payload"] is not None and _current_aqi_cache["expires_at"] > now:
        return _current_aqi_cache["payload"], True

    today = datetime.now(ZoneInfo(KARACHI_TIMEZONE)).date()
    try:
        current_aqi = fetch_current_open_meteo_aqi(str(today))
        if current_aqi:
            _current_aqi_cache["payload"] = current_aqi
            _current_aqi_cache["expires_at"] = now + CURRENT_AQI_CACHE_TTL_SECONDS
            return current_aqi, True
    except Exception:
        pass

    latest = df.sort_values("date").iloc[-1]
    latest_aqi = calculate_aqi(float(latest[settings.target_column]))
    return {
        "date": str(latest["date"]),
        "pm25": latest_aqi["pm25"],
        "aqi": latest_aqi["aqi"],
        "category": latest_aqi["category"],
        "color": latest_aqi["color"],
    }, False


def describe_model(payload: dict) -> str:
    if payload.get("framework") != "sklearn_horizons":
        return payload.get("model_name", "unknown")

    names = []
    for horizon in (1, 2, 3):
        hp = payload["horizon_models"].get(horizon) or payload["horizon_models"].get(str(horizon))
        if hp:
            names.append(f"Day {horizon}: {hp.get('model_name', 'unknown')}")
    return ", ".join(names) if names else payload.get("model_name", "unknown")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/reload-model")
def reload_model():
    """Force-expire the model cache so the next request picks up a fresh model."""
    _model_cache["expires_at"] = 0.0
    _model_cache["payload"] = None
    return {"status": "model cache cleared"}


@app.get("/predict-latest")
def predict_latest():
    payload = load_latest_registered_model()

    if payload.get("framework") != "sklearn_horizons":
        raise HTTPException(
            status_code=409,
            detail=(
                "Registered model is not a sklearn_horizons model. "
                "Run python scripts/train_model.py to register the new 3-day horizon model, "
                "then call /reload-model or wait up to 5 minutes for the cache to expire."
            ),
        )

    df = read_features_cached().sort_values("date")
    model_name = describe_model(payload)
    today = datetime.now(ZoneInfo(KARACHI_TIMEZONE)).date()
    predictions = []

    for days_ahead in range(1, 4):
        pred_date = today + timedelta(days=days_ahead)
        pm25_pred = predict_for_date(payload, df, horizon=days_ahead)  # FIX: pass full df
        aqi_info = calculate_aqi(pm25_pred)
        predictions.append({
            "date": str(pred_date),
            "pm25": aqi_info["pm25"],
            "aqi": aqi_info["aqi"],
            "category": aqi_info["category"],
            "color": aqi_info["color"],
        })

    latest_actual, used_open_meteo_current = get_latest_actual(df)

    return {
        "model_name": model_name,
        "model_version": payload["model_version"],
        "prediction_feature_date": str(df["date"].iloc[-1]),  # FIX: use df directly
        "used_open_meteo_current": used_open_meteo_current,
        "predictions": predictions,
        "latest_actual": latest_actual,
    }




    payload = load_latest_registered_model()
    if payload.get("framework") != "sklearn_horizons":
        raise HTTPException(status_code=409, detail="Model is not sklearn_horizons type.")

    df = read_features_cached().sort_values("date")
    feature_cols = payload["features"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing feature columns: {missing}")

    X_bg = df[feature_cols].dropna()
    # Use at most 200 background rows to keep SHAP fast
    X_sample = X_bg.tail(200)

    result = {}
    for horizon in (1, 2, 3):
        hp = payload["horizon_models"].get(horizon) or payload["horizon_models"].get(str(horizon))
        if hp is None:
            continue

        model_pipeline = hp["model"]
        model_name = hp.get("model_name", "")
        # Extract the final estimator from the sklearn Pipeline
        estimator = model_pipeline.named_steps.get("model", model_pipeline)

        # Apply the pipeline's imputer transform to get clean input for SHAP
        try:
            imputer = model_pipeline.named_steps["imputer"]
            X_transformed = pd.DataFrame(
                imputer.transform(X_sample),
                columns=feature_cols,
            )
        except Exception:
            X_transformed = X_sample.fillna(X_sample.median())

        try:
            if model_name == "random_forest":
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_transformed)
            else:
                # KernelExplainer works for Ridge and MLP
                explainer = shap.KernelExplainer(
                    estimator.predict, shap.sample(X_transformed, 50)
                )
                shap_values = explainer.shap_values(X_transformed, nsamples=100)

            mean_abs = dict(
                zip(feature_cols, [float(v) for v in abs(shap_values).mean(axis=0)])
            )
            # Sort descending
            result[f"horizon_{horizon}"] = dict(
                sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)
            )
        except Exception as exc:
            result[f"horizon_{horizon}"] = {"error": str(exc)}

    return result


@app.get("/history")
def history(limit: int = 30):
    df = read_features_cached().sort_values("date").tail(limit)
    history_data = []
    for _, row in df.iterrows():
        pm25 = float(row[settings.target_column])
        aqi_info = calculate_aqi(pm25)
        history_data.append({
            "date": str(row["date"]),
            "pm25": aqi_info["pm25"],
            "aqi": aqi_info["aqi"],
            "category": aqi_info["category"],
            "color": aqi_info["color"],
        })
    return history_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)