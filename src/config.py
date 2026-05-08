import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    hopsworks_api_key: str = os.getenv("HOPSWORKS_API_KEY", "")
    hopsworks_project: str = os.getenv("HOPSWORKS_PROJECT", "")
    feature_group_name: str = os.getenv("FEATURE_GROUP_NAME", "karachi_aqi_features")
    feature_group_version: int = int(os.getenv("FEATURE_GROUP_VERSION") or "1")
    feature_group_primary_key: str = os.getenv("FEATURE_GROUP_PRIMARY_KEY", "date")
    target_column: str = os.getenv("TARGET_COLUMN", "pm2_5")
    model_name: str = os.getenv("MODEL_NAME", "karachi_aqi_model")
    model_version: int = int(os.getenv("MODEL_VERSION") or "1")
    prediction_api_url: str = os.getenv(
        "PREDICTION_API_URL", "http://localhost:8000/predict-latest"
    )


settings = Settings()


def validate_settings() -> None:
    missing = []
    if not settings.hopsworks_api_key:
        missing.append("HOPSWORKS_API_KEY")
    if not settings.hopsworks_project:
        missing.append("HOPSWORKS_PROJECT")

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
