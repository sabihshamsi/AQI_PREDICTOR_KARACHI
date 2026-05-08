from __future__ import annotations

from pathlib import Path
from typing import Any

import hopsworks
import pandas as pd
from hopsworks_common.client.exceptions import ModelRegistryException

from src.config import settings, validate_settings


def login_to_hopsworks():
    validate_settings()
    return hopsworks.login(
        api_key_value=settings.hopsworks_api_key,
        project=settings.hopsworks_project,
    )


def get_or_create_feature_group(feature_store):
    return feature_store.get_or_create_feature_group(
        name=settings.feature_group_name,
        version=settings.feature_group_version,
        primary_key=[settings.feature_group_primary_key],
        event_time=settings.feature_group_primary_key,
        description="Karachi AQI features from Open-Meteo",
    )


def insert_features(df: pd.DataFrame) -> None:
    project = login_to_hopsworks()
    fs = project.get_feature_store()
    fg = get_or_create_feature_group(fs)
    fg.insert(df, write_options={"wait_for_job": True})


def read_features() -> pd.DataFrame:
    project = login_to_hopsworks()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(
        name=settings.feature_group_name, version=settings.feature_group_version
    )
    return fg.read()


def resolve_model_version(mr, model_name: str, preferred_version: int) -> int:
    version = preferred_version
    while True:
        try:
            mr.get_model(model_name, version=version)
            version += 1
        except ModelRegistryException:
            return version


def register_model_artifact(model_dir: Path, metrics: dict[str, float], model_type: str) -> Any:
    project = login_to_hopsworks()
    mr = project.get_model_registry()
    model_version = resolve_model_version(mr, settings.model_name, settings.model_version)

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
