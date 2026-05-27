from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

DROP_COLUMNS = [
    "SCATS Number",
    "CD_MELWAY",
    "NB_LATITUDE",
    "NB_LONGITUDE",
    "HF VicRoads Internal",
    "VR Internal Stat",
    "VR Internal Loc",
    "NB_TYPE_SURVEY",
]
DEFAULT_DATA_FILE = "Scats Data October 2006.xls"


def _resolve_data_path(data_path: str | Path | None = None) -> Path:
    if data_path is not None:
        path = Path(data_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Could not find dataset at: {path}")
        return path

    candidates = [
        Path(__file__).with_name(DEFAULT_DATA_FILE),
        Path(__file__).resolve().parent.parent / DEFAULT_DATA_FILE,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {DEFAULT_DATA_FILE} next to GRU_model.py or in the repo root."
    )


def _require_tensorflow():
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, GRU, Input
        from tensorflow.keras.models import Sequential
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required to train the GRU model. "
            "Install it in the Anaconda environment used for the notebook."
        ) from exc

    return Sequential, Input, GRU, Dense, Dropout, EarlyStopping


def load_scats_data(data_path: str | Path | None = None) -> pd.DataFrame:
    path = _resolve_data_path(data_path)
    df = pd.read_excel(path, sheet_name="Data")

    # This workbook usually contains a metadata row above the actual header row.
    # Re-read with header=1 when the expected SCATS columns are missing.
    if "SCATS Number" not in df.columns:
        df = pd.read_excel(path, sheet_name="Data", header=1)

    return df.drop(DROP_COLUMNS, axis=1).copy()


def get_locations(data_path: str | Path | None = None) -> np.ndarray:
    return load_scats_data(data_path)["Location"].unique()


def _prepare_location_timeslot_data(
    loc_index: int,
    time_index: int,
    data_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    df = load_scats_data(data_path)
    locations = df["Location"].unique()

    if not 0 <= loc_index < len(locations):
        raise IndexError(
            f"loc_index {loc_index} is out of range for {len(locations)} locations."
        )

    df_loc = df[df["Location"] == locations[loc_index]].copy()
    df_loc["Date"] = pd.to_datetime(df_loc["Date"])
    df_loc["day"] = df_loc["Date"].dt.day
    df_loc["month"] = df_loc["Date"].dt.month
    df_loc["year"] = df_loc["Date"].dt.year
    df_loc["day_of_week"] = df_loc["Date"].dt.dayofweek

    x = df_loc[["day", "month", "year", "day_of_week"]].copy()
    y_all = df_loc.drop(
        ["day", "month", "year", "day_of_week", "Location", "Date"],
        axis=1,
    )

    if not 0 <= time_index < y_all.shape[1]:
        raise IndexError(
            f"time_index {time_index} is out of range for {y_all.shape[1]} time slots."
        )

    y = y_all.iloc[:, time_index].copy()
    x["y_lag1"] = y.shift(1)
    x["y_lag2"] = y.shift(2)

    x = x.dropna().copy()
    y = y.loc[x.index].copy()

    x.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    return x, y, locations


def build_gru_model(
    num_features: int,
    first_units: int = 64,
    second_units: int = 32,
    dropout_rate: float = 0.2,
):
    Sequential, Input, GRU, Dense, Dropout, _ = _require_tensorflow()

    model = Sequential(
        [
            Input(shape=(1, num_features)),
            GRU(first_units, return_sequences=True),
            Dropout(dropout_rate),
            GRU(second_units),
            Dropout(dropout_rate),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_gru_model(
    loc_index: int,
    time_index: int,
    data_path: str | Path | None = None,
    train_size: float = 0.8,
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 16,
    validation_split: float = 0.1,
    patience: int = 10,
    verbose: int = 0,
) -> dict[str, Any]:
    _, _, _, _, _, EarlyStopping = _require_tensorflow()
    x, y, locations = _prepare_location_timeslot_data(loc_index, time_index, data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        train_size=train_size,
        random_state=random_state,
        shuffle=False,
    )

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_train_scaled = scaler_x.fit_transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)
    y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.to_numpy().reshape(-1, 1))

    x_train_3d = x_train_scaled.reshape(x_train_scaled.shape[0], 1, x_train_scaled.shape[1])
    x_test_3d = x_test_scaled.reshape(x_test_scaled.shape[0], 1, x_test_scaled.shape[1])

    model = build_gru_model(num_features=x_train_3d.shape[2])
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train_3d,
        y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=verbose,
    )

    y_pred_scaled = model.predict(x_test_3d, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = y_test.to_numpy()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mean_true = float(np.mean(y_true))
    nrmse = float(rmse / mean_true) if mean_true else float("inf")

    return {
        "model": model,
        "history": history,
        "locations": locations,
        "location": locations[loc_index],
        "time_column": f"V{time_index:02d}",
        "x_train": x_train,
        "x_test": x_test,
        "x_test_3d": x_test_3d,
        "y_train": y_train,
        "y_test": y_test,
        "y_test_scaled": y_test_scaled,
        "y_pred": y_pred,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "rmse": rmse,
        "nrmse": nrmse,
        "data_path": str(_resolve_data_path(data_path)),
    }


def _mean_for_day_of_week(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    target_day_of_week: int,
) -> float:
    matching = y_train[x_train["day_of_week"] == target_day_of_week]
    if matching.empty:
        return float(np.mean(y_train))
    return float(np.mean(matching))


def build_prediction_features(
    predict_day: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    today = datetime.strptime(predict_day, "%m/%d/%Y")
    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)

    return pd.DataFrame(
        [
            {
                "day": today.day,
                "month": today.month,
                "year": today.year,
                "day_of_week": today.weekday(),
                "y_lag1": _mean_for_day_of_week(y_train, x_train, yesterday.weekday()),
                "y_lag2": _mean_for_day_of_week(y_train, x_train, day_before.weekday()),
            }
        ]
    )


def predict_gru_with_results(
    results: dict[str, Any],
    predict_day: str,
) -> np.ndarray:
    x_new = build_prediction_features(
        predict_day=predict_day,
        x_train=results["x_train"],
        y_train=results["y_train"],
    )
    x_new_scaled = results["scaler_x"].transform(x_new)
    x_new_3d = x_new_scaled.reshape(1, 1, x_new_scaled.shape[1])

    pred_scaled = results["model"].predict(x_new_3d, verbose=0)
    return results["scaler_y"].inverse_transform(pred_scaled).flatten()


def traffic_gru(
    predict_day: str,
    loc_index: int,
    time_index: int,
    data_path: str | Path | None = None,
    train_size: float = 0.8,
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 16,
    validation_split: float = 0.1,
    patience: int = 10,
    verbose: int = 0,
) -> np.ndarray:
    results = train_gru_model(
        loc_index=loc_index,
        time_index=time_index,
        data_path=data_path,
        train_size=train_size,
        random_state=random_state,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        patience=patience,
        verbose=verbose,
    )

    return predict_gru_with_results(results, predict_day)


def save_gru_plots(
    results: dict[str, Any],
    output_dir: str | Path | None = None,
    prefix: str = "gru",
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_path = Path(__file__).with_name("Visualization graphs")
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_plot = output_path / f"{prefix}_prediction_plot.png"
    training_plot = output_path / f"{prefix}_training_loss.png"

    plt.figure(figsize=(12, 10))
    plt.plot(range(len(results["y_test"])), results["y_test"].to_numpy(), label="True traffic")
    plt.plot(range(len(results["y_pred"])), results["y_pred"], label="Predicted traffic")
    plt.title(
        f"GRU - Traffic Volume Prediction ({results['location']}, {results['time_column']})"
    )
    plt.xlabel("Sample index")
    plt.ylabel("Traffic volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(prediction_plot, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(results["history"].history["loss"], label="Train loss")
    if "val_loss" in results["history"].history:
        plt.plot(results["history"].history["val_loss"], label="Val loss")
    plt.title("GRU Training History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(training_plot, dpi=150)
    plt.close()

    return {
        "prediction_plot": str(prediction_plot),
        "training_plot": str(training_plot),
    }


if __name__ == "__main__":
    result = traffic_gru("10/15/2006", loc_index=1, time_index=0)
    print(f"\nPredicted traffic volume for 10/15/2006: {result[0]:.1f}")
