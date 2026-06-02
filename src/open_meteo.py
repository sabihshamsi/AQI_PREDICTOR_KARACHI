from __future__ import annotations

import pandas as pd
import requests

KARACHI_LATITUDE = 24.8607
KARACHI_LONGITUDE = 67.0011
KARACHI_TIMEZONE = "Asia/Karachi"

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AIR_QUALITY_VARIABLES = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
]
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]
CURRENT_AQI_VARIABLES = ["pm2_5", "us_aqi"]


def _aqi_category(us_aqi: float) -> tuple[str, str]:
    if us_aqi <= 50:
        return "Good", "#00e400"
    if us_aqi <= 100:
        return "Moderate", "#ffff00"
    if us_aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    if us_aqi <= 200:
        return "Unhealthy", "#ff0000"
    if us_aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    return "Hazardous", "#7e0023"


def _get_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _hourly_to_daily(payload: dict, variables: list[str]) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        return pd.DataFrame(columns=["date", *variables])

    df = pd.DataFrame(hourly)
    df["date"] = pd.to_datetime(df["time"]).dt.date
    daily = df.groupby("date", as_index=False)[variables].mean(numeric_only=True)
    return daily


def _weather_url(start_date: str, end_date: str) -> str:
    pd.to_datetime(start_date).date()
    pd.to_datetime(end_date).date()
    # Backfill and mixed historical ranges must not be sent to the forecast API.
    return WEATHER_ARCHIVE_URL


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar and pollutant trend features."""
    if df.empty:
        return df

    features = df.sort_values("date").copy()
    features["date"] = pd.to_datetime(features["date"])
    date = features["date"]

    features["day_of_week"] = date.dt.dayofweek
    features["day_of_month"] = date.dt.day
    features["month"] = date.dt.month
    features["day_of_year"] = date.dt.dayofyear
    features["is_weekend"] = date.dt.dayofweek.isin([5, 6]).astype(int)

    if "pm2_5" in features.columns:
        features["pm2_5_lag_1d"] = features["pm2_5"].shift(1)
        features["pm2_5_change_1d"] = features["pm2_5"].diff()
        features["pm2_5_pct_change_1d"] = features["pm2_5"].pct_change()
        features["pm2_5_rolling_mean_3d"] = (
            features["pm2_5"].rolling(window=3, min_periods=1).mean()
        )

    if "pm10" in features.columns:
        features["pm10_lag_1d"] = features["pm10"].shift(1)
        features["pm10_change_1d"] = features["pm10"].diff()

    return features.replace([float("inf"), -float("inf")], pd.NA)


def fetch_open_meteo(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily Karachi weather and air-quality features from Open-Meteo."""
    base_params = {
        "latitude": KARACHI_LATITUDE,
        "longitude": KARACHI_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": KARACHI_TIMEZONE,
    }

    air_quality_payload = _get_json(
        AIR_QUALITY_URL,
        {**base_params, "hourly": ",".join(AIR_QUALITY_VARIABLES)},
    )
    weather_payload = _get_json(
        _weather_url(start_date, end_date),
        {**base_params, "hourly": ",".join(WEATHER_VARIABLES)},
    )

    air_quality = _hourly_to_daily(air_quality_payload, AIR_QUALITY_VARIABLES)
    weather = _hourly_to_daily(weather_payload, WEATHER_VARIABLES)
    features = air_quality.merge(weather, on="date", how="left")
    features = add_feature_engineering(features)
    return features.sort_values("date").reset_index(drop=True)


def fetch_current_open_meteo_aqi(date: str) -> dict | None:
    """Fetch the latest hourly AQI value for a Karachi date from Open-Meteo."""
    payload = _get_json(
        AIR_QUALITY_URL,
        {
            "latitude": KARACHI_LATITUDE,
            "longitude": KARACHI_LONGITUDE,
            "start_date": date,
            "end_date": date,
            "timezone": KARACHI_TIMEZONE,
            "hourly": ",".join(CURRENT_AQI_VARIABLES),
        },
    )
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        return None

    df = pd.DataFrame(hourly)
    df = df.dropna(subset=["us_aqi"])
    if df.empty:
        return None

    latest = df.iloc[-1]
    category, color = _aqi_category(float(latest["us_aqi"]))
    pm25 = latest.get("pm2_5")
    return {
        "date": str(latest["time"])[:10],
        "time": str(latest["time"]),
        "pm25": round(float(pm25), 2) if pd.notna(pm25) else None,
        "aqi": round(float(latest["us_aqi"])),
        "category": category,
        "color": color,
    }
