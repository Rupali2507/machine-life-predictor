

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
TARGET_CAPS: tuple[int | None, ...] = (100, 150, None)
FEATURES = [
    "cycle", "setting_1", "setting_2", "sensor_2", "sensor_3",
    "sensor_4", "sensor_6", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]


def load_training_data() -> pd.DataFrame:
    columns = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    columns += [f"sensor_{i}" for i in range(1, 22)]
    frame = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=columns)
    frame["RUL"] = frame.groupby("engine_id")["cycle"].transform("max") - frame["cycle"]
    return frame


def make_model(name: str, seed: int):
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=1.0,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def predict_at_cutoff(
    frame: pd.DataFrame,
    engine_ids: np.ndarray,
    model,
    cutoff: float,
    seed: int,
    cap: int | None,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    for engine_id in engine_ids:
        trajectory = frame.loc[frame.engine_id == engine_id].sort_values("cycle")
        count = max(1, int(np.floor(len(trajectory) * cutoff)))
        row = trajectory.iloc[count - 1]
        prediction = max(0.0, float(model.predict(row[FEATURES].to_frame().T)[0]))
        rows.append(
            {
                "seed": seed,
                "model": model_name,
                "target_cap": "uncapped" if cap is None else cap,
                "cutoff": cutoff,
                "engine_id": int(engine_id),
                "observed_rows": count,
                "cutoff_cycle": int(row.cycle),
                "true_RUL": float(row.RUL),
                "predicted_RUL": prediction,
                "error": prediction - float(row.RUL),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"FD001 training data not found: {DATA_PATH}")
    frame = load_training_data()
    engine_ids = frame.engine_id.unique()
    outputs = []
    total_fits = len(SEEDS) * len(TARGET_CAPS) * 2
    total_evaluations = total_fits * len(CUTOFFS)
    fit_number = 0
    evaluation_number = 0
    print(
        f"Starting FD001 comparison: {len(SEEDS)} splits, "
        f"{len(TARGET_CAPS)} target settings, 2 models, "
        f"{len(CUTOFFS)} cutoffs ({total_fits} fits, "
        f"{total_evaluations} evaluations).",
        flush=True,
    )

    for seed in SEEDS:
        train_ids, validation_ids = train_test_split(
            engine_ids, test_size=0.20, random_state=seed
        )
        train = frame.loc[frame.engine_id.isin(train_ids)]
        validation = frame.loc[frame.engine_id.isin(validation_ids)]
        assert set(train_ids).isdisjoint(validation_ids)

        for cap in TARGET_CAPS:
            target = train.RUL if cap is None else train.RUL.clip(upper=cap)
            for model_name in ("HistGradientBoosting", "ExtraTrees"):
                fit_number += 1
                cap_label = "uncapped" if cap is None else str(cap)
                print(
                    f"[fit {fit_number}/{total_fits}] seed={seed}, "
                    f"target_cap={cap_label}, model={model_name}: "
                    f"training on {len(train_ids)} engines...",
                    flush=True,
                )
                model = make_model(model_name, seed)
                model.fit(train[FEATURES], target)
                for cutoff in CUTOFFS:
                    evaluation_number += 1
                    print(
                        f"  [evaluation {evaluation_number}/{total_evaluations}] "
                        f"cutoff={cutoff:.0%}, "
                        f"validating on {len(validation_ids)} engines...",
                        flush=True,
                    )
                    outputs.append(
                        predict_at_cutoff(
                            validation,
                            validation_ids,
                            model,
                            cutoff,
                            seed,
                            cap,
                            model_name,
                        )
                    )

    predictions = pd.concat(outputs, ignore_index=True)
    predictions["absolute_error"] = predictions.error.abs()
    predictions["actual_rul_band"] = pd.cut(
        predictions.true_RUL,
        bins=[-np.inf, 20, 40, 80, 150, np.inf],
        labels=["0-20", "21-40", "41-80", "81-150", ">150"],
    )

    metric_rows = []
    for keys, group in predictions.groupby(
        ["seed", "model", "target_cap", "cutoff"], observed=True
    ):
        seed, model_name, cap, cutoff = keys
        metric_rows.append(
            {
                "seed": seed,
                "model": model_name,
                "target_cap": cap,
                "cutoff": cutoff,
                "MAE": mean_absolute_error(group.true_RUL, group.predicted_RUL),
                "RMSE": np.sqrt(
                    mean_squared_error(group.true_RUL, group.predicted_RUL)
                ),
                "bias": group.error.mean(),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["model", "target_cap", "cutoff"], observed=True)
        .agg(
            mean_MAE=("MAE", "mean"),
            std_MAE=("MAE", "std"),
            mean_RMSE=("RMSE", "mean"),
            std_RMSE=("RMSE", "std"),
            mean_bias=("bias", "mean"),
            std_bias=("bias", "std"),
        )
        .reset_index()
    )
    by_rul = (
        predictions.groupby(
            ["model", "target_cap", "cutoff", "actual_rul_band"], observed=True
        )
        .agg(
            engines=("engine_id", "size"),
            MAE=("absolute_error", "mean"),
            mean_bias=("error", "mean"),
            overestimate_rate=("error", lambda errors: (errors > 0).mean()),
        )
        .reset_index()
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(REPORT_DIR / "fd001_experiment_predictions.csv", index=False)
    metrics.to_csv(REPORT_DIR / "fd001_experiment_metrics_by_split.csv", index=False)
    summary.to_csv(REPORT_DIR / "fd001_experiment_summary.csv", index=False)
    by_rul.to_csv(REPORT_DIR / "fd001_experiment_errors_by_actual_rul.csv", index=False)
    print("All experiments complete. Summary metrics:", flush=True)
    print(summary.round(2).to_string(index=False))
    print(f"\nReports written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
