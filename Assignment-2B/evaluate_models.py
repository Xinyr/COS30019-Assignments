from __future__ import annotations

import argparse
import csv
from pathlib import Path

from GRU import save_gru_plots, train_gru_model
from LSTM import save_lstm_plots, train_lstm_model
from RandomForest import save_random_forest_plots, train_random_forest_model


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Scats Data October 2006.xls"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and compare LSTM, GRU, and Random Forest.")
    parser.add_argument("--loc-index", type=int, default=1, help="Location index from the SCATS workbook.")
    parser.add_argument("--time-index", type=int, default=0, help="15-minute slot index (0-95).")
    parser.add_argument("--epochs", type=int, default=60, help="Epoch count for LSTM and GRU.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"), help="Folder for CSV and plots.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rf_results = train_random_forest_model(
        data_path=DATA_PATH,
        eval_loc_index=args.loc_index,
        eval_time_index=args.time_index,
    )
    lstm_results = train_lstm_model(
        loc_index=args.loc_index,
        time_index=args.time_index,
        data_path=DATA_PATH,
        epochs=args.epochs,
    )
    gru_results = train_gru_model(
        loc_index=args.loc_index,
        time_index=args.time_index,
        data_path=DATA_PATH,
        epochs=args.epochs,
    )

    rf_plots = save_random_forest_plots(rf_results, output_dir=output_dir, prefix="rf")
    lstm_plots = save_lstm_plots(lstm_results, output_dir=output_dir, prefix="lstm")
    gru_plots = save_gru_plots(gru_results, output_dir=output_dir, prefix="gru")

    rows = [
        {
            "model": "random_forest",
            "location": rf_results["eval_location"] or "global",
            "time_column": f"V{args.time_index:02d}",
            "rmse": rf_results["eval_rmse"] if rf_results["eval_rmse"] is not None else rf_results["rmse"],
            "nrmse": rf_results["eval_nrmse"] if rf_results["eval_nrmse"] is not None else rf_results["nrmse"],
            "prediction_plot": rf_plots["prediction_plot"],
        },
        {
            "model": "lstm",
            "location": lstm_results["location"],
            "time_column": lstm_results["time_column"],
            "rmse": lstm_results["rmse"],
            "nrmse": lstm_results["nrmse"],
            "prediction_plot": lstm_plots["prediction_plot"],
        },
        {
            "model": "gru",
            "location": gru_results["location"],
            "time_column": gru_results["time_column"],
            "rmse": gru_results["rmse"],
            "nrmse": gru_results["nrmse"],
            "prediction_plot": gru_plots["prediction_plot"],
        },
    ]

    csv_path = output_dir / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved model comparison to {csv_path}")
    for row in rows:
        print(
            f"{row['model']}: location={row['location']} {row['time_column']} "
            f"RMSE={row['rmse']:.2f} NRMSE={row['nrmse']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
