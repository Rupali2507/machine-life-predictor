"""Test higher training weights for near-failure FD001 examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/train/train_FD001.txt"
REPORT_DIR = ROOT / "reports"
SEEDS = (42, 21, 7, 100, 2026)
CUTOFFS = (0.60, 0.70, 0.80, 0.90)
WEIGHTS = (1.0, 2.0, 4.0)
RUL_CAP = 100
WEIGHT_RUL_THRESHOLD = 40
FEATURES = [
    "cycle", "setting_1", "setting_2", "sensor_2", "sensor_3",
    "sensor_4", "sensor_6", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]


def make_model(name: str, seed: int):
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    return HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )


def predict_validation(frame, engine_ids, model, cutoff, seed, model_name, weight):
    rows = []
    for engine_id in engine_ids:
        engine = frame.loc[frame.engine_id == engine_id].sort_values("cycle")
        count = max(1, int(np.floor(len(engine) * cutoff)))
        row = engine.iloc[count - 1]
        prediction = max(0.0, float(model.predict(row[FEATURES].to_frame().T)[0]))
        rows.append({
            "seed": seed,
            "model": model_name,
            "near_failure_weight": weight,
            "cutoff": cutoff,
            "engine_id": int(engine_id),
            "true_RUL": float(row.RUL),
            "predicted_RUL": prediction,
            "error": prediction - float(row.RUL),
        })
    return rows


def main() -> None:
    columns = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    columns += [f"sensor_{i}" for i in range(1, 22)]
    frame = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=columns)
    frame["RUL"] = frame.groupby("engine_id")["cycle"].transform("max") - frame.cycle
    ids = frame.engine_id.unique()
    outputs = []
    total_fits = len(SEEDS) * 2 * len(WEIGHTS)
    fit_no = 0
    print(
        f"Comparing weights {WEIGHTS} for training RUL <= {WEIGHT_RUL_THRESHOLD}: "
        f"{total_fits} fits, {total_fits * len(CUTOFFS)} cutoff evaluations.",
        flush=True,
    )

    for seed in SEEDS:
        train_ids, validation_ids = train_test_split(ids, test_size=0.20, random_state=seed)
        train = frame.loc[frame.engine_id.isin(train_ids)]
        validation = frame.loc[frame.engine_id.isin(validation_ids)]
        targets = train.RUL.clip(upper=RUL_CAP)
        for model_name in ("ExtraTrees", "HistGradientBoosting"):
            for weight in WEIGHTS:
                fit_no += 1
                sample_weight = np.where(train.RUL.to_numpy() <= WEIGHT_RUL_THRESHOLD, weight, 1.0)
                print(
                    f"[fit {fit_no}/{total_fits}] seed={seed}, model={model_name}, "
                    f"near_failure_weight={weight:g}",
                    flush=True,
                )
                model = make_model(model_name, seed)
                model.fit(train[FEATURES], targets, sample_weight=sample_weight)
                for cutoff in CUTOFFS:
                    print(f"  evaluating cutoff={cutoff:.0%}", flush=True)
                    outputs.extend(
                        predict_validation(
                            validation, validation_ids, model, cutoff, seed,
                            model_name, weight,
                        )
                    )

    predictions = pd.DataFrame(outputs)
    predictions["absolute_error"] = predictions.error.abs()
    predictions["actual_rul_band"] = pd.cut(
        predictions.true_RUL,
        bins=[-np.inf, 20, 40, 80, 150, np.inf],
        labels=["0-20", "21-40", "41-80", "81-150", ">150"],
    )
    split_rows = []
    for keys, group in predictions.groupby(
        ["seed", "model", "near_failure_weight", "cutoff"], observed=True
    ):
        seed, model_name, weight, cutoff = keys
        split_rows.append({
            "seed": seed,
            "model": model_name,
            "near_failure_weight": weight,
            "cutoff": cutoff,
            "MAE": mean_absolute_error(group.true_RUL, group.predicted_RUL),
            "RMSE": np.sqrt(mean_squared_error(group.true_RUL, group.predicted_RUL)),
            "bias": group.error.mean(),
        })
    metrics = pd.DataFrame(split_rows)
    summary = (
        metrics.groupby(["model", "near_failure_weight", "cutoff"], observed=True)
        .agg(
            mean_MAE=("MAE", "mean"), std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"), std_RMSE=("RMSE", "std"),
            mean_bias=("bias", "mean"), std_bias=("bias", "std"),
        )
        .reset_index()
    )
    near_failure = (
        predictions.loc[predictions.actual_rul_band == "21-40"]
        .groupby(["model", "near_failure_weight", "cutoff"], observed=True)
        .agg(
            engine_fold_predictions=("engine_id", "size"),
            MAE=("absolute_error", "mean"),
            mean_bias=("error", "mean"),
            overestimate_rate=("error", lambda error: (error > 0).mean()),
        )
        .reset_index()
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(REPORT_DIR / "fd001_near_failure_weight_predictions.csv", index=False)
    metrics.to_csv(REPORT_DIR / "fd001_near_failure_weight_metrics_by_split.csv", index=False)
    summary.to_csv(REPORT_DIR / "fd001_near_failure_weight_summary.csv", index=False)
    near_failure.to_csv(REPORT_DIR / "fd001_near_failure_weight_21_40_band.csv", index=False)
    print("\nOverall summary:")
    print(summary.round(2).to_string(index=False))
    print("\n21-40 actual RUL band:")
    print(near_failure.round(2).to_string(index=False))
    print(f"\nReports written under {REPORT_DIR}")


if __name__ == "__main__":
    main()
