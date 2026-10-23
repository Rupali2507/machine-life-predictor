"""Compare observed sensor summaries for overestimated and accurate engines.

Uses the 80% cutoff, ExtraTrees with a 100-cycle target cap, and engines with
21-40 cycles of actual RUL. Each engine is summarized once after averaging its
prediction errors across the validation splits in which it appears.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "reports/fd001_experiment_predictions.csv"
DATA_PATH = ROOT / "data/raw/train/train_FD001.txt"
REPORT_DIR = ROOT / "reports"
SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_9",
    "sensor_11", "sensor_12", "sensor_14", "sensor_15",
]
N_ENGINES = 5


def main() -> None:
    columns = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    columns += [f"sensor_{i}" for i in range(1, 22)]
    data = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=columns)
    predictions = pd.read_csv(PREDICTIONS)

    selected = predictions.loc[
        (predictions["model"] == "ExtraTrees")
        & (predictions["target_cap"].astype(str) == "100")
        & (predictions["cutoff"] == 0.8)
    ].copy()
    engine_errors = (
        selected.groupby("engine_id", as_index=False)
        .agg(
            true_RUL=("true_RUL", "mean"),
            mean_prediction=("predicted_RUL", "mean"),
            mean_error=("error", "mean"),
            split_appearances=("seed", "nunique"),
        )
    )
    in_band = engine_errors.loc[engine_errors.true_RUL.between(21, 40)].copy()
    if len(in_band) < 2 * N_ENGINES:
        raise ValueError(f"Expected at least {2 * N_ENGINES} engines in band; found {len(in_band)}")

    bad = in_band.nlargest(N_ENGINES, "mean_error").assign(group="largest_overestimates")
    controls = (
        in_band.loc[~in_band.engine_id.isin(bad.engine_id)]
        .assign(abs_error=lambda frame: frame.mean_error.abs())
        .nsmallest(N_ENGINES, "abs_error")
        .drop(columns="abs_error")
        .assign(group="most_accurate_controls")
    )
    chosen = pd.concat([bad, controls], ignore_index=True)

    feature_rows = []
    for row in chosen.itertuples(index=False):
        trajectory = data.loc[data.engine_id == row.engine_id].sort_values("cycle")
        n_observed = max(1, int(np.floor(len(trajectory) * 0.8)))
        observed = trajectory.iloc[:n_observed]
        result = {
            "engine_id": int(row.engine_id),
            "group": row.group,
            "true_RUL": row.true_RUL,
            "mean_prediction": row.mean_prediction,
            "mean_error": row.mean_error,
            "split_appearances": int(row.split_appearances),
            "observed_cycles": n_observed,
            "last_observed_cycle": int(observed.cycle.iloc[-1]),
        }
        x = np.arange(n_observed, dtype=float)
        for sensor in SENSORS:
            values = observed[sensor].to_numpy(dtype=float)
            result[f"{sensor}_mean"] = float(values.mean())
            result[f"{sensor}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            result[f"{sensor}_last"] = float(values[-1])
            result[f"{sensor}_trend_per_cycle"] = (
                float(np.polyfit(x, values, 1)[0]) if len(values) > 1 else 0.0
            )
        feature_rows.append(result)

    detail = pd.DataFrame(feature_rows)
    stat_columns = [
        f"{sensor}_{stat}"
        for sensor in SENSORS
        for stat in ("mean", "std", "last", "trend_per_cycle")
    ]
    summary_rows = []
    for feature in stat_columns:
        group_means = detail.groupby("group")[feature].mean()
        over = float(group_means["largest_overestimates"])
        control = float(group_means["most_accurate_controls"])
        pooled = float(detail[feature].std(ddof=1))
        summary_rows.append(
            {
                "feature_summary": feature,
                "overestimate_group_mean": over,
                "control_group_mean": control,
                "difference_over_minus_control": over - control,
                "standardized_difference_overall_sd": (over - control) / pooled if pooled else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = REPORT_DIR / "fd001_sensor_pattern_engine_comparison.csv"
    summary_path = REPORT_DIR / "fd001_sensor_pattern_group_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Largest overestimates:")
    print(bad[["engine_id", "true_RUL", "mean_prediction", "mean_error", "split_appearances"]].round(2).to_string(index=False))
    print("\nMost accurate controls in the same actual-RUL band:")
    print(controls[["engine_id", "true_RUL", "mean_prediction", "mean_error", "split_appearances"]].round(2).to_string(index=False))
    print("\nLargest absolute sensor-summary differences (standardized):")
    print(
        summary.assign(
            abs_effect=lambda frame: frame.standardized_difference_overall_sd.abs()
        )
        .nlargest(12, "abs_effect")[[
            "feature_summary", "overestimate_group_mean", "control_group_mean",
            "difference_over_minus_control", "standardized_difference_overall_sd",
        ]]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nWrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
