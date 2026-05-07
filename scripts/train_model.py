from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.hopsworks_utils import read_features, register_model_artifact


def build_models() -> dict[str, Pipeline]:
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
                        n_estimators=300, max_depth=16, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
        # Advanced baseline neural model from sklearn (fast and easy to deploy).
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


def evaluate(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_tensorflow(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"Skipping tensorflow model: {exc}")
        return None, None

    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    preds = model.predict(X_test, verbose=0).ravel()
    return model, preds


def run_pytorch(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"Skipping pytorch model: {exc}")
        return None, None

    torch.manual_seed(42)
    x_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    x_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        out = model(x_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(x_test_t).numpy().ravel()
    return model, preds


def main() -> None:
    df = read_features()
    if settings.target_column not in df.columns:
        raise ValueError(f"Target column '{settings.target_column}' not found in feature store")

    df = df.sort_values("date")
    feature_cols = [c for c in df.columns if c not in {"date", settings.target_column}]
    X = df[feature_cols]
    y = df[settings.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    best_name = ""
    best_metrics = {"rmse": float("inf"), "mae": float("inf"), "r2": -float("inf")}
    best_model = None

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test_np, preds)
        print(f"{model_name}: {metrics}")

        if metrics["rmse"] < best_metrics["rmse"]:
            best_name = model_name
            best_metrics = metrics
            best_model = model

    # Standardized arrays for deep-learning candidates.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_arr = scaler.transform(imputer.transform(X_test))

    tf_model, tf_preds = run_tensorflow(X_train_arr, y_train_np, X_test_arr)
    if tf_model is not None:
        tf_metrics = evaluate(y_test_np, tf_preds)
        print(f"tensorflow_dense: {tf_metrics}")
        if tf_metrics["rmse"] < best_metrics["rmse"]:
            best_name = "tensorflow_dense"
            best_metrics = tf_metrics
            best_model = {"framework": "tensorflow", "model": tf_model, "imputer": imputer, "scaler": scaler}

    torch_model, torch_preds = run_pytorch(X_train_arr, y_train_np, X_test_arr)
    if torch_model is not None:
        torch_metrics = evaluate(y_test_np, torch_preds)
        print(f"pytorch_mlp: {torch_metrics}")
        if torch_metrics["rmse"] < best_metrics["rmse"]:
            best_name = "pytorch_mlp"
            best_metrics = torch_metrics
            best_model = {"framework": "pytorch", "model": torch_model, "imputer": imputer, "scaler": scaler}

    if best_model is None:
        raise RuntimeError("No model was successfully trained.")

    model_dir = Path("artifacts") / "best_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(best_model, dict) and best_model.get("framework") == "tensorflow":
        tf_dir = model_dir / "tensorflow_model"
        best_model["model"].save(tf_dir)
        joblib.dump(
            {
                "framework": "tensorflow",
                "model_path": str(tf_dir),
                "imputer": best_model["imputer"],
                "scaler": best_model["scaler"],
                "features": feature_cols,
                "model_name": best_name,
            },
            model_dir / "model.joblib",
        )
    elif isinstance(best_model, dict) and best_model.get("framework") == "pytorch":
        torch_path = model_dir / "torch_model.pt"
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"PyTorch selected but unavailable during save: {exc}") from exc
        torch.save(best_model["model"].state_dict(), torch_path)
        joblib.dump(
            {
                "framework": "pytorch",
                "model_state_path": str(torch_path),
                "imputer": best_model["imputer"],
                "scaler": best_model["scaler"],
                "features": feature_cols,
                "model_name": best_name,
            },
            model_dir / "model.joblib",
        )
    else:
        joblib.dump(
            {
                "framework": "sklearn",
                "model": best_model,
                "features": feature_cols,
                "model_name": best_name,
            },
            model_dir / "model.joblib",
        )

    register_model_artifact(
        model_dir=model_dir,
        metrics=best_metrics,
        model_type=best_name,
    )
    print(f"Registered model '{best_name}' with metrics: {best_metrics}")


if __name__ == "__main__":
    main()
