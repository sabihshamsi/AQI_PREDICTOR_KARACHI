from __future__ import annotations

from pathlib import Path
from typing import Any

import hopsworks
import pandas as pd
from hopsworks_common.client.exceptions import ModelRegistryException

from src.config import settings, validate_settings


def _disable_model_serving_bootstrap() -> None:
    """Skip optional serving setup that is not needed by this app."""
    try:
        from hsml.core import model_serving_api
    except Exception:
        return

    model_serving_api.ModelServingApi.load_default_configuration = lambda self: None


def login_to_hopsworks():
    validate_settings()
    _disable_model_serving_bootstrap()
    return hopsworks.login(
        api_key_value=settings.hopsworks_api_key,
        project=settings.hopsworks_project,
    )


def get_or_create_feature_group(feature_store):
    return feature_store.get_or_create_feature_group(
        name=settings.feature_group_name,
        version=settings.feature_group_version,
        primary_key=["date"],
        event_time="date",
        description="Karachi AQI features from Open-Meteo",
        online_enabled=False,
    )


def insert_features(df: pd.DataFrame) -> None:
    project = login_to_hopsworks()
    fs = project.get_feature_store()
    fg = get_or_create_feature_group(fs)
    fg.insert(
        df,
        write_options={
            "wait_for_job": True,
            "overwrite": False,
        },
    )


def read_features() -> pd.DataFrame:
    project = login_to_hopsworks()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(
        name=settings.feature_group_name, version=settings.feature_group_version
    )
    return fg.read()


def register_model_artifact(model_dir: Path, metrics: dict[str, float], model_type: str) -> Any:
    project = login_to_hopsworks()
    mr = project.get_model_registry()
    models = mr.get_models(settings.model_name)

    if models:
        latest_version = max(m.version for m in models)
        model_version = latest_version + 1
    else:
        model_version = 1

    if model_version != settings.model_version:
        print(
            f"Model version {settings.model_version} already exists, registering as version {model_version} instead."
        )

    py_model = mr.python.create_model(
        name=settings.model_name,
        version=model_version,
        description="Best AQI model for Karachi trained from Feature Store data",
        metrics=metrics,
    )
    py_model.save(str(model_dir))
    return py_model
