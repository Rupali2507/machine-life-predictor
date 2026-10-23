"""Evaluate causal sensor trend features against the existing FD001 features."""

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
RUL_CAP = 100
WINDOW = 20
BASE_FEATURES = [
    "cycle", "setting_1", "setting_2", "sensor_2", "sensor_3",
    "sensor_4", "sensor_6", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]
TREND_FEATURES = [f"sensor_{sensor}_slope_{WINDOW}" for sensor in (9, 14)]


def slope(values: np.ndarray) -> float:
    """OLS slope against equally spaced cycle indices for one rolling window."""
    n = len(values)
    if n < 5:
        return 0.0
    x = np.arange(n, dtype=float)
    centered_x = x - x.mean()
    return float(np.dot(centered_x, values - values.mean()) / np.dot(centered_x, centered_x))


def add_causal_trends(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["engine_id", "cycle"]).copy()
    for sensor in (9, 14):
        name = f"sensor_{sensor}_slope_{WINDOW}"
        frame[name] = frame.groupby("engine_id")[f"sensor_{sensor}"].transform(
            lambda series: series.rolling(WINDOW, min_periods=5).apply(slope, raw=True)
        )
        frame[name] = frame[name].fillna(0.0)
    return frame


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


def evaluate_cutoff(frame, engine_ids, model, features, cutoff, seed, model_name, variant):
    rows = []
    for engine_id in engine_ids:
        engine = frame.loc[frame.engine_id == engine_id].sort_values("cycle")
        count = max(1, int(np.floor(len(engine) * cutoff)))
        row = engine.iloc[count - 1]
        prediction = max(0.0, float(model.predict(row[features].to_frame().T)[0]))
        rows.append({
            "seed": seed,
            "model": model_name,
            "features": variant,
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
    frame = add_causal_trends(frame)
    ids = frame.engine_id.unique()
    prediction_rows = []
    total_fits = len(SEEDS) * 2 * 2
    fit_no = 0
    print(
        f"Comparing baseline and causal {WINDOW}-cycle sensor slopes: "
        f"{total_fits} fits, {total_fits * len(CUTOFFS)} cutoff evaluations.",
        flush=True,
    )

    for seed in SEEDS:
        train_ids, validation_ids = train_test_split(ids, test_size=0.20, random_state=seed)
        train = frame.loc[frame.engine_id.isin(train_ids)]
        validation = frame.loc[frame.engine_id.isin(validation_ids)]
        for model_name in ("ExtraTrees", "HistGradientBoosting"):
            for variant, features in (
                ("baseline", BASE_FEATURES),
                ("plus_sensor_trends", BASE_FEATURES + TREND_FEATURES),
            ):
                fit_no += 1
                print(
                    f"[fit {fit_no}/{total_fits}] seed={seed}, model={model_name}, "
                    f"features={variant}", flush=True
                )
                model = make_model(model_name, seed)
                model.fit(train[features], train.RUL.clip(upper=RUL_CAP))
                for cutoff in CUTOFFS:
                    print(f"  evaluating cutoff={cutoff:.0%}", flush=True)
                    prediction_rows.extend(
                        evaluate_cutoff(
                            validation, validation_ids, model, features, cutoff,
                            seed, model_name, variant,
                        )
                    )

    predictions = pd.DataFrame(prediction_rows)
    predictions["absolute_error"] = predictions.error.abs()
    predictions["actual_rul_band"] = pd.cut(
        predictions.true_RUL,
        bins=[-np.inf, 20, 40, 80, 150, np.inf],
        labels=["0-20", "21-40", "41-80", "81-150", ">150"],
    )
    metric_rows = []
    for keys, group in predictions.groupby(
        ["seed", "model", "features", "cutoff"], observed=True
    ):
        seed, model_name, variant, cutoff = keys
        metric_rows.append({
            "seed": seed,
            "model": model_name,
            "features": variant,
            "cutoff": cutoff,
            "MAE": mean_absolute_error(group.true_RUL, group.predicted_RUL),
            "RMSE": np.sqrt(mean_squared_error(group.true_RUL, group.predicted_RUL)),
            "bias": group.error.mean(),
        })
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["model", "features", "cutoff"], observed=True)
        .agg(
            mean_MAE=("MAE", "mean"), std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"), std_RMSE=("RMSE", "std"),
            mean_bias=("bias", "mean"), std_bias=("bias", "std"),
        )
        .reset_index()
    )
    band = (
        predictions.loc[predictions.actual_rul_band == "21-40"]
        .groupby(["model", "features", "cutoff"], observed=True)
        .agg(
            engine_fold_predictions=("engine_id", "size"),
            MAE=("absolute_error", "mean"),
            mean_bias=("error", "mean"),
            overestimate_rate=("error", lambda error: (error > 0).mean()),
        )
        .reset_index()
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(REPORT_DIR / "fd001_sensor_trend_predictions.csv", index=False)
    metrics.to_csv(REPORT_DIR / "fd001_sensor_trend_metrics_by_split.csv", index=False)
    summary.to_csv(REPORT_DIR / "fd001_sensor_trend_summary.csv", index=False)
    band.to_csv(REPORT_DIR / "fd001_sensor_trend_near_failure.csv", index=False)
    print("\nOverall summary:")
    print(summary.round(2).to_string(index=False))
    print("\n21-40 actual RUL band:")
    print(band.round(2).to_string(index=False))
    print(f"\nReports written under {REPORT_DIR}")


if __name__ == "__main__":
    main()
