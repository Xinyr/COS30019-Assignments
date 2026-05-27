from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
VALUE_COLUMNS = [f"V{i:02d}" for i in range(96)]
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
        f"Could not find {DEFAULT_DATA_FILE} next to RandomForest_model.py or in the repo root."
    )


def load_scats_data(data_path: str | Path | None = None) -> pd.DataFrame:
    path = _resolve_data_path(data_path)
    df = pd.read_excel(path, sheet_name="Data")

    if "SCATS Number" not in df.columns:
        df = pd.read_excel(path, sheet_name="Data", header=1)

    return df.drop(DROP_COLUMNS, axis=1).copy()


def get_locations(data_path: str | Path | None = None) -> np.ndarray:
    return load_scats_data(data_path)["Location"].unique()


def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=["Location", "Date"],
        value_vars=VALUE_COLUMNS,
        var_name="time_column",
        value_name="traffic_volume",
    ).copy()

    long_df["Date"] = pd.to_datetime(long_df["Date"])
    long_df["date_only"] = long_df["Date"].dt.normalize()
    long_df["time_index"] = long_df["time_column"].str[1:].astype(int)
    long_df["hour"] = long_df["time_index"] // 4
    long_df["minute"] = (long_df["time_index"] % 4) * 15
    long_df["day"] = long_df["date_only"].dt.day
    long_df["month"] = long_df["date_only"].dt.month
    long_df["year"] = long_df["date_only"].dt.year
    long_df["day_of_week"] = long_df["date_only"].dt.dayofweek
    long_df["is_weekend"] = (long_df["day_of_week"] >= 5).astype(int)

    long_df.sort_values(["Location", "time_index", "date_only"], inplace=True)
    long_df["lag1"] = long_df.groupby(["Location", "time_index"])["traffic_volume"].shift(1)
    long_df["lag2"] = long_df.groupby(["Location", "time_index"])["traffic_volume"].shift(2)
    long_df.dropna(inplace=True)
    long_df.reset_index(drop=True, inplace=True)
    return long_df


def _split_by_date(long_df: pd.DataFrame, train_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(long_df["date_only"].unique()))
    if len(unique_dates) < 2:
        raise ValueError("Not enough unique dates to create a train/test split.")

    split_index = int(len(unique_dates) * train_size)
    split_index = max(1, min(split_index, len(unique_dates) - 1))

    train_dates = set(unique_dates[:split_index])
    test_dates = set(unique_dates[split_index:])

    train_df = long_df[long_df["date_only"].isin(train_dates)].copy()
    test_df = long_df[long_df["date_only"].isin(test_dates)].copy()
    return train_df, test_df


def _build_pipeline(
    n_estimators: int = 60,
    max_depth: int | None = 14,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    feature_columns = [
        "Location",
        "time_index",
        "hour",
        "minute",
        "day",
        "month",
        "year",
        "day_of_week",
        "is_weekend",
        "lag1",
        "lag2",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("location", OneHotEncoder(handle_unknown="ignore"), ["Location"]),
            (
                "numeric",
                "passthrough",
                [col for col in feature_columns if col != "Location"],
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _subset_metrics(df: pd.DataFrame, pred_col: str = "predicted_volume") -> tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(df["traffic_volume"], df[pred_col])))
    mean_true = float(df["traffic_volume"].mean())
    nrmse = float(rmse / mean_true) if mean_true else float("inf")
    return rmse, nrmse


def train_random_forest_model(
    data_path: str | Path | None = None,
    train_size: float = 0.8,
    random_state: int = 42,
    n_estimators: int = 60,
    max_depth: int | None = 14,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
    eval_loc_index: int | None = None,
    eval_time_index: int | None = None,
    allowed_locations: Iterable[str] | None = None,
) -> dict[str, Any]:
    wide_df = load_scats_data(data_path)
    if allowed_locations is not None:
        allowed_locations = set(allowed_locations)
        wide_df = wide_df[wide_df["Location"].isin(allowed_locations)].copy()
        if wide_df.empty:
            raise ValueError("allowed_locations filtered out all rows for RandomForest training.")
    locations = wide_df["Location"].unique()
    long_df = _wide_to_long(wide_df)
    train_df, test_df = _split_by_date(long_df, train_size=train_size)

    feature_columns = [
        "Location",
        "time_index",
        "hour",
        "minute",
        "day",
        "month",
        "year",
        "day_of_week",
        "is_weekend",
        "lag1",
        "lag2",
    ]

    pipeline = _build_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipeline.fit(train_df[feature_columns], train_df["traffic_volume"])
    y_pred = pipeline.predict(test_df[feature_columns])

    test_results = test_df.copy()
    test_results["predicted_volume"] = y_pred
    rmse, nrmse = _subset_metrics(test_results)

    subset_results = None
    if eval_loc_index is not None and eval_time_index is not None:
        if not 0 <= eval_loc_index < len(locations):
            raise IndexError(
                f"eval_loc_index {eval_loc_index} is out of range for {len(locations)} locations."
            )
        eval_location = locations[eval_loc_index]
        subset_results = test_results[
            (test_results["Location"] == eval_location)
            & (test_results["time_index"] == eval_time_index)
        ].copy()
        if not subset_results.empty:
            subset_rmse, subset_nrmse = _subset_metrics(subset_results)
        else:
            subset_rmse, subset_nrmse = None, None
    else:
        eval_location = None
        subset_rmse, subset_nrmse = None, None

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_importances = pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": feature_importances,
        }
    ).sort_values("importance", ascending=False)

    return {
        "pipeline": pipeline,
        "locations": locations,
        "wide_df": wide_df,
        "train_df": train_df,
        "test_df": test_results,
        "rmse": rmse,
        "nrmse": nrmse,
        "feature_importances": importance_df,
        "data_path": str(_resolve_data_path(data_path)),
        "allowed_locations": sorted(allowed_locations) if allowed_locations else None,
        "eval_location": eval_location,
        "eval_time_index": eval_time_index,
        "eval_subset": subset_results,
        "eval_rmse": subset_rmse,
        "eval_nrmse": subset_nrmse,
    }


def _mean_lag_for_prediction(
    train_df: pd.DataFrame,
    location: str,
    time_index: int,
    target_day_of_week: int,
) -> float:
    mask = (
        (train_df["Location"] == location)
        & (train_df["time_index"] == time_index)
        & (train_df["day_of_week"] == target_day_of_week)
    )
    matching = train_df.loc[mask, "traffic_volume"]
    if not matching.empty:
        return float(matching.mean())

    fallback = train_df.loc[
        (train_df["Location"] == location) & (train_df["time_index"] == time_index),
        "traffic_volume",
    ]
    if not fallback.empty:
        return float(fallback.mean())

    return float(train_df["traffic_volume"].mean())


def _build_prediction_row(
    results: dict[str, Any],
    predict_day: str,
    loc_index: int,
    time_index: int,
) -> pd.DataFrame:
    locations = results["locations"]
    if not 0 <= loc_index < len(locations):
        raise IndexError(
            f"loc_index {loc_index} is out of range for {len(locations)} locations."
        )

    location = locations[loc_index]
    target_date = datetime.strptime(predict_day, "%m/%d/%Y")
    yesterday = target_date - timedelta(days=1)
    day_before = target_date - timedelta(days=2)

    return pd.DataFrame(
        [
            {
                "Location": location,
                "time_index": time_index,
                "hour": time_index // 4,
                "minute": (time_index % 4) * 15,
                "day": target_date.day,
                "month": target_date.month,
                "year": target_date.year,
                "day_of_week": target_date.weekday(),
                "is_weekend": int(target_date.weekday() >= 5),
                "lag1": _mean_lag_for_prediction(
                    results["train_df"], location, time_index, yesterday.weekday()
                ),
                "lag2": _mean_lag_for_prediction(
                    results["train_df"], location, time_index, day_before.weekday()
                ),
            }
        ]
    )


def predict_random_forest_with_results(
    results: dict[str, Any],
    predict_day: str,
    loc_index: int,
    time_index: int,
) -> np.ndarray:
    prediction_row = _build_prediction_row(results, predict_day, loc_index, time_index)
    prediction = results["pipeline"].predict(prediction_row)
    return np.asarray(prediction)


def traffic_random_forest(
    predict_day: str,
    loc_index: int,
    time_index: int,
    data_path: str | Path | None = None,
    train_size: float = 0.8,
    random_state: int = 42,
    n_estimators: int = 60,
    max_depth: int | None = 14,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
    allowed_locations: Iterable[str] | None = None,
) -> np.ndarray:
    results = train_random_forest_model(
        data_path=data_path,
        train_size=train_size,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        allowed_locations=allowed_locations,
    )
    return predict_random_forest_with_results(results, predict_day, loc_index, time_index)


def save_random_forest_plots(
    results: dict[str, Any],
    output_dir: str | Path | None = None,
    prefix: str = "rf",
    max_points: int = 300,
) -> dict[str, str]:
    if output_dir is None:
        output_path = Path(__file__).with_name("Visualization graphs")
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_plot = output_path / f"{prefix}_prediction_plot.png"
    importance_plot = output_path / f"{prefix}_feature_importance.png"

    plot_df = results["eval_subset"]
    if plot_df is None or plot_df.empty:
        plot_df = results["test_df"].head(max_points).copy()
        title_suffix = "Global Test Sample"
    else:
        plot_df = plot_df.sort_values("date_only").head(max_points).copy()
        title_suffix = f"{results['eval_location']} / V{results['eval_time_index']:02d}"

    plt.figure(figsize=(12, 10))
    plt.plot(
        range(len(plot_df)),
        plot_df["traffic_volume"].to_numpy(),
        label="True traffic",
    )
    plt.plot(
        range(len(plot_df)),
        plot_df["predicted_volume"].to_numpy(),
        label="Predicted traffic",
    )
    plt.title(f"Random Forest - Traffic Volume Prediction ({title_suffix})")
    plt.xlabel("Sample index")
    plt.ylabel("Traffic volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(prediction_plot, dpi=150)
    plt.close()

    top_features = results["feature_importances"].head(15).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(importance_plot, dpi=150)
    plt.close()

    return {
        "prediction_plot": str(prediction_plot),
        "feature_importance_plot": str(importance_plot),
    }


if __name__ == "__main__":
    results = train_random_forest_model(eval_loc_index=1, eval_time_index=0)
    prediction = traffic_random_forest("10/15/2006", loc_index=1, time_index=0)
    print(f"Global RMSE: {results['rmse']:.2f}")
    print(f"Global NRMSE: {results['nrmse']:.2f}")
    if results["eval_rmse"] is not None:
        print(f"Focused RMSE: {results['eval_rmse']:.2f}")
        print(f"Focused NRMSE: {results['eval_nrmse']:.2f}")
    print(f"Predicted traffic volume for 10/15/2006: {prediction[0]:.1f}")
