"""Plot sensor 9 and 14 trajectories for the selected FD001 engines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/train/train_FD001.txt"
COMPARISON_PATH = ROOT / "reports/fd001_sensor_pattern_engine_comparison.csv"
FIGURE_DIR = ROOT / "reports/figures"
SENSORS = ("sensor_9", "sensor_14")
GROUP_ORDER = ("largest_overestimates", "most_accurate_controls")
GROUP_LABELS = {
    "largest_overestimates": "Largest overestimates",
    "most_accurate_controls": "Most accurate controls",
}


def main() -> None:
    columns = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    columns += [f"sensor_{i}" for i in range(1, 22)]
    data = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=columns)
    comparison = pd.read_csv(COMPARISON_PATH)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for sensor in SENSORS:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)
        for ax, group in zip(axes, GROUP_ORDER):
            engines = comparison.loc[comparison.group == group].sort_values("engine_id")
            for item in engines.itertuples(index=False):
                trajectory = data.loc[data.engine_id == item.engine_id].sort_values("cycle")
                n_observed = max(1, int(item.observed_cycles))
                cutoff_cycle = int(trajectory.iloc[n_observed - 1].cycle)
                ax.plot(
                    trajectory.cycle,
                    trajectory[sensor],
                    marker=".",
                    markersize=2.5,
                    linewidth=1.1,
                    alpha=0.8,
                    label=f"Engine {int(item.engine_id)} (RUL {int(item.true_RUL)})",
                )
                ax.axvline(cutoff_cycle, color="0.35", linestyle="--", linewidth=0.8, alpha=0.45)
            ax.set_title(GROUP_LABELS[group])
            ax.set_ylabel(sensor.replace("_", " ").title())
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, ncol=2, loc="best")
        axes[-1].set_xlabel("Engine cycle (dashed line marks 80% cutoff per engine)")
        fig.suptitle(f"FD001 {sensor.replace('_', ' ').title()} trajectories")
        fig.tight_layout()
        output = FIGURE_DIR / f"fd001_{sensor}_trajectory_comparison.png"
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
