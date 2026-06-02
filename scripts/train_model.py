from __future__ import annotations

from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import logging

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.hopsworks_utils import read_features, register_model_artifact

logger = logging.getLogger(__name__)


def fetch_features_with_retry(max_retries: int = 3, wait_seconds: int = 60) -> pd.DataFrame:
    for attempt in range(1, max_retries + 1):
        try:
            df = read_features()
        except Exception as e:
            logger.exception("Failed to read features from Hopsworks on attempt %d/%d: %s", attempt, max_retries, e)
            if attempt == max_retries:
                raise
            logger.info("Retrying data fetch after %s seconds...", wait_seconds)
            time.sleep(wait_seconds)
            continue

        logger.info("Retrieved %d rows from feature store", len(df))
        logger.info("Features shape: %s", getattr(df, "shape", None))
        logger.info(
            "Columns: %s",
            getattr(df, "columns", None).tolist() if hasattr(df, "columns") else None,
        )

        if not df.empty:
            return df

        if attempt < max_retries:
            logger.warning(
                "Feature store returned empty dataset on attempt %d/%d. Retrying in %s seconds...",
                attempt,
                max_retries,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    return df


def build_models() -> dict[str, Pipeline]:
    """Three Scikit-learn models: linear baseline, ensemble, and MLP."""
    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300, max_depth=16, random_state=42, n_jobs=1
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        learning_rate_init=1e-3,
                        max_iter=400,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    r2 = (
        float(r2_score(y_true, y_pred))
        if len(y_true) > 1
        else 0.0
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

def add_lag_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute lag and rolling features AFTER the train/test split to prevent
    data leakage. Test lag features are computed using only the tail of the
    training set as history — never from future test rows.

    Returns updated (train_df, test_df) with lag columns added.
    """
    lag_cols = ["pm2_5_lag_1d", "pm2_5_change_1d", "pm2_5_pct_change_1d",
                "pm2_5_rolling_mean_3d", "pm10_lag_1d", "pm10_change_1d"]

    # Drop any pre-computed lag columns that arrived from the feature store
    # (they were computed on the full dataset and are therefore leaky).
    train_df = train_df.drop(columns=[c for c in lag_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in lag_cols if c in test_df.columns])

    # --- Training lag features (safe: only uses training history) ---
    if "pm2_5" in train_df.columns:
        train_df = train_df.copy()
        train_df["pm2_5_lag_1d"] = train_df["pm2_5"].shift(1)
        train_df["pm2_5_change_1d"] = train_df["pm2_5"].diff()
        train_df["pm2_5_pct_change_1d"] = train_df["pm2_5"].pct_change()
        train_df["pm2_5_rolling_mean_3d"] = (
            train_df["pm2_5"].rolling(window=3, min_periods=1).mean()
        )
    if "pm10" in train_df.columns:
        train_df["pm10_lag_1d"] = train_df["pm10"].shift(1)
        train_df["pm10_change_1d"] = train_df["pm10"].diff()

    # --- Test lag features (safe: seed the rolling window from tail of train) ---
    # We prepend the last 3 training rows as context, compute features on the
    # combined block, then strip the context rows back off.
    context_rows = 3
    context = train_df.tail(context_rows).copy()
    combined = pd.concat([context, test_df.copy()], ignore_index=True)

    if "pm2_5" in combined.columns:
        combined["pm2_5_lag_1d"] = combined["pm2_5"].shift(1)
        combined["pm2_5_change_1d"] = combined["pm2_5"].diff()
        combined["pm2_5_pct_change_1d"] = combined["pm2_5"].pct_change()
        combined["pm2_5_rolling_mean_3d"] = (
            combined["pm2_5"].rolling(window=3, min_periods=1).mean()
        )
    if "pm10" in combined.columns:
        combined["pm10_lag_1d"] = combined["pm10"].shift(1)
        combined["pm10_change_1d"] = combined["pm10"].diff()

    test_df = combined.iloc[context_rows:].reset_index(drop=True)

    return (
        train_df.replace([float("inf"), -float("inf")], pd.NA),
        test_df.replace([float("inf"), -float("inf")], pd.NA),
    )


def chronological_split(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronological train/test split with a gap equal to the forecast horizon
    to prevent boundary leakage between the horizon-shifted target and the
    lag features on either side of the split boundary.
    """
    df = df.copy()
    df["_target"] = df[target_col].shift(-horizon)
    df = df.dropna(subset=["_target"])

    n = len(df)
    test_n = max(1, int(n * test_size))
    gap = horizon  # rows to drop at the boundary

    train_end = n - test_n - gap
    if train_end < 1:
        raise ValueError(
            f"Not enough data for horizon={horizon} with test_size={test_size}. "
            f"Have {n} rows after dropping NaN targets."
        )

    train = df.iloc[:train_end].copy()
    test = df.iloc[train_end + gap:].copy()

    # Validate that the test set is not empty after applying the gap
    if len(test) == 0:
        raise ValueError(
            f"Insufficient data for horizon={horizon}. "
            f"After train/test split with gap={gap}, no test samples remain. "
            f"Total samples: {n}, train_end: {train_end}, gap: {gap}. "
            f"Consider reducing horizon or test_size."
        )

    y_train = train.pop("_target")
    y_test = test.pop("_target")

    return train, test, y_train, y_test


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    df = fetch_features_with_retry()
    if df.empty:
        logger.error("Feature store is empty after retries. Skipping training.")
        return

    if settings.target_column not in df.columns:
        raise ValueError(f"Target column '{settings.target_column}' not found in feature store")

    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 30:
        logger.warning(
            "Dataset has only %d rows. This may be insufficient for horizons 1-3 with test_size=0.2.",
            len(df),
        )

    # Base feature columns (excluding date and the raw target).
    # Lag columns will be recomputed leak-free inside the loop.
    lag_cols = {"pm2_5_lag_1d", "pm2_5_change_1d", "pm2_5_pct_change_1d",
                "pm2_5_rolling_mean_3d", "pm10_lag_1d", "pm10_change_1d"}
    base_feature_cols = [
        c for c in df.columns
        if c not in {"date", settings.target_column} and c not in lag_cols
    ]

    horizon_models: dict = {}
    horizon_metrics: dict = {}
    feature_cols = []  # Initialize before loop to avoid NameError if all horizons fail

    for horizon in (1, 2, 3):
        print(f"\n=== Training horizon {horizon} day(s) ahead ===")

        try:
            train_df, test_df, y_train, y_test = chronological_split(
                df,
                settings.target_column,
                horizon=horizon,
            )
        except ValueError as split_error:
            logger.warning(f"Skipping horizon {horizon}: {split_error}")
            continue
        except Exception as e:
            logger.exception(f"Error during split for horizon {horizon}: {e}")
            continue

        print(
            f"Horizon {horizon}: "
            f"train={len(train_df)}, "
            f"test={len(test_df)}, "
            f"y_train={len(y_train)}, "
            f"y_test={len(y_test)}"
        )

        if len(train_df) == 0 or len(test_df) == 0:
            print(
                f"Skipping horizon {horizon}: "
                f"empty train or test set."
            )
            continue

        train_df, test_df = add_lag_features(train_df, test_df)

        feature_cols = [
            c
            for c in base_feature_cols + list(lag_cols)
            if c in train_df.columns
            and c in test_df.columns
            and c not in {"date", settings.target_column}
        ]

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        if len(X_train) == 0 or len(X_test) == 0:
            print(
                f"Skipping horizon {horizon}: "
                f"X_train={X_train.shape}, "
                f"X_test={X_test.shape}"
            )
            continue

        y_test_np = y_test.to_numpy()

        best_name = ""
        best_metrics = {
            "rmse": float("inf"),
            "mae": float("inf"),
            "r2": -float("inf"),
        }
        best_model = None

        for model_name, model in build_models().items():
            try:
                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                metrics = evaluate(y_test_np, preds)

                print(
                    f"  horizon_{horizon}_{model_name}: "
                    f"{metrics}"
                )

                if metrics["rmse"]  < best_metrics["rmse"]:
                    best_name = model_name
                    best_metrics = metrics
                    best_model = model

            except Exception as model_error:
                print(
                    f"  horizon_{horizon}_{model_name} failed: "
                    f"{model_error}"
                )

        if best_model is None:
            print(
                f"No successful model for horizon "
                f"{horizon}. Skipping."
            )
            continue

        print(
            f"  Best for horizon {horizon}: "
            f"{best_name} "
            f"(RMSE={best_metrics['rmse']:.3f})"
        )

        horizon_models[horizon] = {
            "framework": "sklearn",
            "model": best_model,
            "model_name": best_name,
        }

        horizon_metrics[horizon] = best_metrics

    # Safety check: ensure at least one horizon trained successfully
    if not horizon_models:
        logger.error("No models trained successfully.")
        return

    model_dir = Path("artifacts") / "best_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "framework": "sklearn_horizons",
            "horizon_models": horizon_models,
            "features": feature_cols,
            "model_name": "direct_horizon_models",
        },
        model_dir / "model.joblib",
    )

    flat_metrics = {
        f"horizon_{horizon}_{metric}": value
        for horizon, metrics in horizon_metrics.items()
        for metric, value in metrics.items()
    }
    register_model_artifact(
        model_dir=model_dir,
        metrics=flat_metrics,
        model_type="direct_horizon_models",
    )
    print(f"\nRegistered direct horizon models with metrics: {flat_metrics}")


if __name__ == "__main__":
    main()
