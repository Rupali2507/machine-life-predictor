# Machine Life Predictor
### Predictive Maintenance for a Connected Fleet

An experimental predictive maintenance system that uses machine sensor data to estimate **Remaining Useful Life (RUL)**, quantify the uncertainty of that estimate, and turn it into maintenance triage decisions and work orders.

This project is being developed for **TELIPORT Season 4 — Case Study 1: Self-Healing Machines: Predictive Maintenance for a Connected Fleet**.

The objective is to progress from a reproducible machine-learning baseline toward an end-to-end predictive maintenance pipeline that can detect abnormal behavior, estimate machine health, prioritize maintenance actions, and support automated work-order generation.

> **Current status:** On NASA C-MAPSS FD001, the project has a leakage-safe evaluation harness, per-engine feature engineering, Random Forest / HistGradientBoosting models, quantile (interval) RUL prediction, and a rule-based triage agent that generates work orders. Interval calibration, anomaly detection, MLflow tracking, a dashboard, and other C-MAPSS subsets are still to do.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Current Implementation](#current-implementation)
- [Machine Learning Approach](#machine-learning-approach)
- [Experimental Results](#experimental-results)
- [Uncertainty Estimation and Triage](#uncertainty-estimation-and-triage)
- [Sensor Trajectory Analysis](#sensor-trajectory-analysis)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Reproducibility and Evaluation](#reproducibility-and-evaluation)
- [Current Limitations](#current-limitations)
- [Roadmap](#roadmap)
- [Technology Stack](#technology-stack)
- [References](#references)

---

## Problem Statement

Machines operating in connected fleets generate continuous sensor readings, including temperature, pressure, vibration, and other operational measurements.

Traditional maintenance strategies often depend on fixed service intervals or intervention after a failure. These approaches can lead to:

- Unexpected equipment failures and unplanned downtime.
- Maintenance schedules that do not reflect actual machine health.
- Missed opportunities to identify degradation early.
- Increased maintenance costs and inefficient resource allocation.
- Delays between identifying a potential problem and taking action.

The goal of predictive maintenance is to use operational data to estimate machine health and anticipate potential failures before they occur.

### Proposed Solution

The project aims to develop a data-driven pipeline with four major capabilities:

1. **Detect:** Identify unusual machine behavior and potential degradation. *(planned)*
2. **Predict:** Estimate Remaining Useful Life, with an uncertainty interval, using sensor measurements. *(implemented)*
3. **Prioritize:** Apply a rule-based triage agent to determine maintenance severity. *(implemented, prototype)*
4. **Automate:** Generate maintenance work orders and escalate exceptional cases for human review. *(implemented, prototype)*

The long-term objective is to minimize manual intervention while maintaining clear monitoring, evaluation, and maintenance decision logic.

```
sensor history ──► per-engine features ──► quantile RUL model ──► triage rules ──► work orders
```

---

## Project Objectives

### Completed

- Load and explore the NASA C-MAPSS FD001 dataset.
- Validate dataset dimensions, missing values, duplicate rows, and sensor behavior.
- Construct RUL targets from engine operating cycles.
- Identify and remove constant features.
- Train Random Forest and HistGradientBoosting regression baselines and compare them across seeds and observation cutoffs.
- Investigate sensor trajectories associated with large RUL prediction errors.
- Build a leakage-safe evaluation harness: engine-grouped cross-validation, deployment-style cutoffs, official test evaluation, reference baselines, the NASA score, and alert precision/recall.
- Engineer per-engine features (baseline-relative readings, rolling statistics) that substantially reduce error.
- Tune the rolling window and RUL cap with cross-validation.
- Predict RUL intervals with quantile models and use the lower bound for alerts.
- Implement a rule-based triage agent that produces prioritized work orders with a human-review flag.
- Add unit tests for the triage rules.

### Planned

- Calibrate the prediction intervals.
- Implement a dedicated anomaly detection component.
- Evaluate a sequence model (GRU / 1D-CNN) on sensor windows.
- Track experiments and model versions using MLflow.
- Add model performance monitoring and a retraining workflow.
- Build a dashboard for predictions, anomalies, and maintenance priorities.
- Evaluate generalization to additional C-MAPSS subsets.

---

## Dataset

### NASA C-MAPSS

The project uses the **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset, which contains simulated turbofan engine degradation trajectories.

Official dataset information: [NASA Prognostics Center of Excellence — Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

### Current Dataset: FD001

FD001 is the initial dataset selected for this project (single operating condition, single fault mode).

| Dataset component | Shape / count |
|---|---:|
| Training observations | 20,631 rows |
| Training engines | 100 |
| Test observations | 13,096 rows |
| Test engines | 100 |
| Test RUL labels | 100 |

The training engines are run to failure. The test engines are stopped before failure, and `RUL_FD001.txt` gives the true remaining cycles for each.

The dataset includes:

- Engine identifiers.
- Operating cycles.
- Three operational setting columns.
- Twenty-one sensor measurement columns.

### RUL Target Construction

For each engine in the training data, the RUL target is calculated as:

`RUL = Maximum cycle for the engine - Current cycle`

For example, if an engine's final observed training cycle is 200, its RUL at cycle 150 is 50 cycles.

A capped target is used for training:

`RUL_capped = min(RUL, cap)`

This limits large early-life target values. The cap was 150 in the first notebooks; the current evaluation uses **125**, which gave the best cross-validated error among 100, 125, and 150.

**Important:** The target calculation above describes the training-label construction. Test-set RUL must be evaluated using the separate provided test labels rather than treating each engine's last observed test cycle as its failure cycle.

### Data Quality Checks

Initial exploration of the training data found:

- No missing values.
- No duplicate rows.
- Multiple sensors with constant values.
- Different engine lifetimes and sensor trajectories.

Seven constant columns were excluded from the feature set:

- `setting_3`
- `sensor_1`
- `sensor_5`
- `sensor_10`
- `sensor_16`
- `sensor_18`
- `sensor_19`

This leaves 17 raw measurement columns (2 settings and 15 sensors). The first notebooks also used `cycle` as a feature (18 features); the current harness does not.

---

## Current Implementation

### 1. Data Exploration

Notebook: `notebooks/01_fd001_data_exploration.ipynb`

The exploratory analysis covers:

- Dataset loading and structure.
- Training and test dataset dimensions.
- Engine operating-cycle distributions.
- Missing-value and duplicate checks.
- RUL target construction.
- Constant-feature identification.
- Sensor distributions and trajectories.

### 2. Reproducible Baseline Pipeline

Notebook: `notebooks/02_reproducible_baseline.ipynb`

The baseline workflow includes:

1. Load the FD001 training dataset.
2. Construct the RUL target.
3. Remove constant features.
4. Split engines into training and validation groups.
5. Train regression models.
6. Generate validation predictions.
7. Calculate evaluation metrics.

The validation split is performed at the **engine level**, using an 80/20 split with `random_state=42`.

This prevents observations from the same engine from appearing in both training and validation sets.

### 3. Experiment Comparison

The project includes scripts for investigating model performance and prediction errors.

Examples visible in the project structure include:

- `src/compare_rul_experiments.py`
- `src/compare_sensor_patterns.py`
- `src/evaluate_near_failure_weights.py`
- `src/evaluate_sensor_trends.py`

These experiments investigate model configurations, sensor behavior, near-failure prediction performance, and possible sources of prediction error.

The exact outputs depend on the experiment configuration and should be interpreted using the corresponding generated reports.

### 4. Evaluation Harness

Script: `src/evaluation/cv_harness.py` (and `src/evaluation/tune_cv.py`)

- Engine-grouped 5-fold cross-validation with random deployment-style cutoffs, identical for every model.
- Evaluation on the official test set (last observed cycle of each test engine).
- Reference baselines (predict-the-mean, Ridge), Random Forest, HistGradientBoosting, and their average.
- Metrics: MAE, RMSE, bias, NASA score, and precision / recall / F1 for the alert "RUL ≤ 30".
- `tune_cv.py` selects the rolling window and RUL cap by cross-validation, scored against a fixed target so settings are comparable.

### 5. Quantile Models and Triage Agent

Scripts: `src/evaluation/quantile_triage.py`, `src/agent/triage.py`

- Quantile models predict the 10th / 25th / 50th / 90th percentile of RUL.
- The triage agent turns those percentiles into URGENT / SCHEDULE / MONITOR decisions and writes work orders (see [Uncertainty Estimation and Triage](#uncertainty-estimation-and-triage)).
- Tests: `tests/test_triage.py`.

---

## Machine Learning Approach

### Feature Engineering

The current feature set has 62 features: the 17 raw columns plus, for each of the 15 informative sensors:

- the reading relative to the engine's own first 5 cycles,
- a 10-cycle rolling mean,
- a 10-cycle rolling standard deviation.

All features use only past cycles of the same engine. Each engine starts with different wear, and a single row has no memory of the engine's trend, so these features give the models the context they were missing.

### Models

Two regression algorithms are used, plus an average of the two and two reference baselines (predict-the-mean, Ridge on standardized features).

#### Random Forest Regressor

Random Forest combines predictions from multiple decision trees.

| Parameter | Initial notebooks | Current harness |
|---|---:|---:|
| `n_estimators` | 200 | 200 |
| `min_samples_leaf` | 2 | 5 |
| `random_state` | 42 | 42 |
| `n_jobs` | -1 | -1 |

#### HistGradientBoosting Regressor

HistGradientBoosting builds an ensemble of decision trees sequentially, with each stage improving on the previous ensemble.

| Parameter | Initial notebooks | Current harness |
|---|---:|---:|
| `max_iter` | 200 | 300 |
| `learning_rate` | 0.1 | 0.05 |
| `max_leaf_nodes` | 15 | 15 |
| `l2_regularization` | 1.0 | 1.0 |
| `early_stopping` | automatic | off |
| `random_state` | 42 | 42 |

In the initial notebooks, the automatic early stopping stopped the fit at 100 iterations. Scikit-learn's built-in early stopping holds out a random 10% of rows (not whole engines), so rows from the same engine can fall on both sides; the current harness turns it off.

#### Quantile models

HistGradientBoosting with `loss="quantile"` at the 10th, 25th, 50th, and 90th percentiles (same hyperparameters as above). Predictions are sorted to prevent quantile crossing and clipped to the RUL cap.

### Evaluation Metrics

**Mean Absolute Error (MAE)**

Measures the average absolute difference between predicted and actual RUL. Lower MAE indicates more accurate predictions on average.

**Root Mean Squared Error (RMSE)**

Penalizes large prediction errors more heavily than MAE. Lower RMSE indicates fewer or smaller large errors.

**Prediction Bias**

Defined as:

`Bias = Mean(Predicted RUL - Actual RUL)`

Interpretation:

- Positive bias: the model overestimates RUL on average.
- Negative bias: the model underestimates RUL on average.
- Bias near zero: predictions are approximately unbiased on average.

For predictive maintenance, overestimating RUL can be particularly concerning because it may suggest that a machine has more operating life remaining than it actually does.

**NASA score**

An asymmetric score from the C-MAPSS literature that penalizes late predictions (overestimates) more than early ones. Lower is better.

**Alert precision / recall / F1**

Treating "RUL ≤ 30 cycles" as the event to catch: recall is the share of truly urgent engines that were flagged, precision is the share of flagged engines that were truly urgent.

---

## Experimental Results

### Current evaluation: grouped CV and official test

Engine-grouped 5-fold CV (random cutoffs at least 30 cycles into each validation engine; mean ± std over folds):

| Model | MAE | RMSE | Bias | Recall (RUL ≤ 30) |
|---|---:|---:|---:|---:|
| Predict the mean | 38.41 ± 2.56 | 44.20 ± 2.52 | +12.86 | 0.00 |
| Ridge | 13.64 ± 1.17 | 17.23 ± 1.28 | +4.02 | 0.73 |
| HistGradientBoosting | 11.03 ± 1.71 | 15.37 ± 2.64 | +2.60 | 0.88 |
| Random Forest | 11.51 ± 1.69 | 15.88 ± 2.45 | +3.21 | 0.84 |
| Avg(RF+HGB) | 11.15 ± 1.70 | 15.43 ± 2.57 | +2.92 | 0.87 |

Official test set (last observed cycle of each of the 100 test engines). `_raw` columns compare against the uncapped true RUL, which is the form used in the literature:

| Model | MAE_raw | RMSE | RMSE_raw | NASA_raw | Recall (≤ 30) | Precision (≤ 30) |
|---|---:|---:|---:|---:|---:|---:|
| Predict the mean | 35.90 | 41.94 | 43.07 | 33,629 | 0.00 | 0.00 |
| Ridge | 14.40 | 16.73 | 17.96 | 516 | 0.48 | 1.00 |
| HistGradientBoosting | 10.90 | 13.92 | 14.90 | 326 | 0.80 | 0.95 |
| Random Forest | 10.96 | 13.78 | 14.66 | 331 | 0.68 | 0.94 |
| Avg(RF+HGB) | 10.74 | 13.57 | 14.51 | 308 | 0.64 | 0.94 |

### Interpretation

- Random Forest, HistGradientBoosting, and their average are statistically indistinguishable here: the differences are well inside the fold-to-fold standard deviation, and their ordering changes between runs.
- Most of the improvement over the initial raw-feature baselines (test RMSE about 17 → about 14) came from the per-engine features, not from the choice of model. Ridge on the same features already reaches RMSE about 17 versus 44 for predicting the mean.
- Tuning the rolling window (5 / 10 / 30) and RUL cap (100 / 125 / 150) by CV selected window 10 and cap 125. Windows 5 and 10 were tied, and MAE was nearly identical across caps, so the cap result partly reflects aligning training with the scoring target.
- Alerts based on the median prediction miss urgent engines on the test set (recall 0.64–0.80). The interval-based alerts below address this.

### Earlier results: raw-feature baselines (notebooks)

These come from the first notebooks (raw sensor features, RUL cap 150, engine-level 80/20 splits, one prediction per validation engine at a fixed fraction of its life). They are kept for reference; they use 20 validation engines per split and are not comparable to the tables above.

#### Five-Split Baseline Evaluation

The baseline models were evaluated across five engine-level train/validation splits using random seeds 42, 21, 7, 100, and 2026, at the 80% observation cutoff.

| Model | Mean MAE | MAE Std. Dev. | Mean RMSE | RMSE Std. Dev. | Mean Bias |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 14.11 | 2.00 | 18.17 | 2.69 | +8.06 |
| Random Forest | 14.62 | 1.99 | 19.08 | 3.30 | +9.69 |

*RUL errors and bias are measured in operating cycles.*

Both models showed a clear positive bias (overestimating RUL). The 0.5-cycle MAE gap between them was inside the fold-to-fold variation, which the later cross-validation confirmed.

#### Observation Cutoff Experiment

The HistGradientBoosting baseline was also evaluated at different observation cutoffs. The cutoff determines how much of each engine's trajectory is available before making an RUL prediction.

| Observation cutoff | Mean MAE | MAE Std. Dev. | Mean RMSE | Mean Bias |
|---|---:|---:|---:|---:|
| 60% | 19.09 | 3.75 | 23.35 | +6.71 |
| 70% | 17.81 | 3.57 | 22.65 | +6.64 |
| 80% | 14.11 | 2.00 | 18.17 | +8.06 |
| 90% | 6.53 | 1.20 | 8.45 | +2.86 |

Predictions become more accurate as more of the engine trajectory is observed. However, a "percentage of lifetime" cutoff needs the failure cycle, which is not known at prediction time, and error shrinks mechanically as true RUL shrinks. This is why the current harness uses random cutoffs and the official last-cycle test instead.

---

## Uncertainty Estimation and Triage

### Interval predictions

Quantile models give a RUL range instead of a single number. Using the lower bound (10th percentile) for the alert "RUL ≤ 30" trades some precision for much higher recall, which matters because a missed failure costs more than an early inspection.

| Alert based on | CV recall / precision | Test recall / precision |
|---|---:|---:|
| Median | 0.87 / 0.95 | 0.68 / 1.00 |
| 25th percentile | 0.93 / 0.91 | 0.88 / 0.88 |
| 10th percentile (lower bound) | 0.97 / 0.81 | 1.00 / 0.89 |

The 10–90 interval covered the true RUL 69% of the time in CV (82% on the test set) against a nominal 80%, with a mean width of about 30–34 cycles. The intervals are therefore **overconfident and not yet calibrated**.

### Triage rules

`q10`, `q50`, `q90` are the predicted RUL percentiles.

| Severity | Condition | Action |
|---|---|---|
| URGENT (P1) | `q10 ≤ 30` | priority work order |
| SCHEDULE (P2) | `30 < q10 ≤ 60` | scheduled maintenance |
| MONITOR | otherwise | no work order |

A work order is marked `PENDING_REVIEW` when the interval is wider than 60 cycles (`q90 − q10 > 60`), or when `q10 ≤ 30` but `q50 > 30` (the lower bound and the median disagree). Each order contains the engine, priority, status, the RUL lower / median / upper values, the reasons, and a recommended action with a "before cycle X" deadline.

### Outcome on the 100 test engines

| Triage | true RUL ≤ 30 | true RUL > 30 |
|---|---:|---:|
| URGENT | 25 | 3 |
| SCHEDULE | 0 | 19 |
| MONITOR | 0 | 53 |

- All 25 truly urgent engines were flagged, with 3 false alarms (true RUL 34–38, within 4–8 cycles of the threshold). The 17 URGENT engines without a review flag were all truly urgent.
- 15 engines were flagged for review: 11 URGENT engines with lower-bound / median disagreement (8 truly urgent, 3 false alarms) and 4 SCHEDULE engines with wide intervals. The review rule therefore captured all three false alarms without discarding any real alert.
- This is a small test set (about 25 urgent engines), so treat it as encouraging, not conclusive. The thresholds (30 / 60 / 60 cycles) are defaults to be set with the people who act on the alerts.

---

## Sensor Trajectory Analysis

In addition to model-level metrics, the project investigates sensor trajectories for engines with large prediction errors.

The analysis compares sensor measurements across selected groups of engines, including:

- Engines with the largest RUL overestimates.
- Engines for which the model predictions were more accurate.

The purpose is to investigate whether sensor behavior provides useful clues about model failures.

### Sensor 9 Investigation

A trajectory comparison was generated for sensor 9.

The visualization compares selected engines from the largest-overestimate group against selected engines from the more-accurate group. Vertical dashed lines indicate the 80% observation cutoff used in the experiment.

The analysis is exploratory. A visual comparison alone does not establish that sensor 9 causes the prediction errors or that its behavior differs significantly between the two groups.

Further statistical comparisons are required before drawing such conclusions.

### Temporal Feature Experiment (initial, two sensors)

Temporal features were first explored for sensor 9 and sensor 11 only, including rolling means, rolling standard deviations, and first differences.

On a single 20-engine validation split, this configuration did not improve performance:

| Configuration | MAE | RMSE | Bias |
|---|---:|---:|---:|
| Baseline | 13.14 | 16.38 | +10.01 |
| Temporal features | 14.26 | 17.67 | +9.50 |

That single-split result was within noise. The later harness applied per-engine features to all informative sensors and evaluated them with grouped cross-validation, where they clearly helped (see [Experimental Results](#experimental-results)).

### Bias Correction Experiment

A calibration-based bias correction was also explored. The estimated calibration bias was approximately +20.73 cycles.

| Metric | Before correction | After correction |
|---|---:|---:|
| MAE | 17.61 | 17.41 |
| RMSE | 22.50 | 20.63 |
| Bias | +12.31 | -8.43 |

A constant shift moved the bias from positive to negative (overcorrecting) because the bias depends on the RUL level, so it was not adopted. These results come from a separate experimental setup and should not be compared directly with other tables. Interval-based alerts address the overestimation problem more directly.

---

## Project Structure

The repository is organized into notebooks, data, models, evaluation scripts, configuration, and reporting directories.

```text
machine-life-predictor/
│
├── app/
│   ├── backend/
│   └── frontend/
│
├── configs/
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   ├── test/
│   │   └── Rul/
│   └── processed/
│
├── mlruns/
│
├── models/
│   ├── fd001_random_forest_cap150.joblib
│   └── fd001_quantile_triage.joblib
│
├── notebooks/
│   ├── 01_fd001_data_exploration.ipynb
│   └── 02_reproducible_baseline.ipynb
│
├── reports/
│   ├── figures/
│   ├── fd001_baseline_metrics.json
│   ├── fd001_experiment_*.csv, fd001_near_failure_weight_*.csv,
│   │   fd001_sensor_*.csv, fd001_validation_predictions.csv
│   ├── work_orders.json
│   └── work_orders.all_engines.csv
│
├── src/
│   ├── agent/
│   │   └── triage.py              # train / run: RUL intervals -> severity -> work orders
│   ├── data/
│   ├── evaluation/
│   │   ├── cv_harness.py          # grouped CV + official test, baselines, alert metrics
│   │   ├── tune_cv.py             # tune rolling window and RUL cap
│   │   └── quantile_triage.py     # quantile models, interval coverage, triage table
│   ├── compare_rul_experiments.py
│   ├── compare_sensor_patterns.py
│   ├── evaluate_near_failure_weights.py
│   └── evaluate_sensor_trends.py
│
├── tests/
│   └── test_triage.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `app/` and `mlruns/` directories are reserved for future components, and the generated reports may vary as experiments evolve.

---

## Getting Started

### Prerequisites

- Python 3.10 or later is recommended; use a version compatible with the installed dependencies.
- Git.
- A local copy of the NASA C-MAPSS dataset.

The experiments can be run on a CPU; a GPU is not required for the current models.

### 1. Clone the Repository

```bash
git clone https://github.com/Rupali2507/machine-life-predictor.git
cd machine-life-predictor
```

### 2. Create a Virtual Environment

On Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the NASA C-MAPSS dataset from the official NASA Prognostics Center of Excellence data repository and place the FD001 files as follows:

```text
data/
└── raw/
    ├── train/train_FD001.txt
    ├── test/test_FD001.txt
    └── Rul/RUL_FD001.txt
```

**Dataset note:** The raw dataset is not required to be committed to GitHub. Download it separately and keep the local data directories excluded from version control.

### 5. Run the Notebooks

```bash
jupyter notebook
```

Run the notebooks in this order:

1. `notebooks/01_fd001_data_exploration.ipynb`
2. `notebooks/02_reproducible_baseline.ipynb`

The first notebook explores the dataset and constructs the target. The second notebook runs the initial baseline workflow.

### 6. Run the Evaluation and Triage Pipeline

Run these from the repository root:

```bash
# Evaluate baselines, RF, HGB and the average (grouped CV + official test)
python src/evaluation/cv_harness.py \
  --train data/raw/train/train_FD001.txt \
  --test  data/raw/test/test_FD001.txt \
  --rul   data/raw/Rul/RUL_FD001.txt

# Optional: tune rolling window and RUL cap with CV
python src/evaluation/tune_cv.py --train data/raw/train/train_FD001.txt

# Quantile intervals, coverage, and alert metrics
python src/evaluation/quantile_triage.py \
  --train data/raw/train/train_FD001.txt \
  --test  data/raw/test/test_FD001.txt \
  --rul   data/raw/Rul/RUL_FD001.txt

# Train the triage model, then generate work orders
python src/agent/triage.py train --train data/raw/train/train_FD001.txt \
  --out models/fd001_quantile_triage.joblib
python src/agent/triage.py run --model models/fd001_quantile_triage.joblib \
  --history data/raw/test/test_FD001.txt --rul data/raw/Rul/RUL_FD001.txt \
  --out reports/work_orders.json

# Tests
python -m pytest -q tests/test_triage.py
```

`triage.py run` writes `reports/work_orders.json` (URGENT and SCHEDULE engines) and `reports/work_orders.all_engines.csv` (every engine). Engine histories must start at cycle 1, and an upper bound of 125 means "125 or more".

### 7. Run Earlier Experiment Scripts

```bash
python src/compare_rul_experiments.py
python src/compare_sensor_patterns.py
python src/evaluate_near_failure_weights.py
python src/evaluate_sensor_trends.py
```

Run these from the repository root. Their reports are written under `reports/` according to each script's configuration.

---

## Reproducibility and Evaluation

Reproducibility is important because machine-learning performance can change depending on how engines are split into training and validation sets.

The current evaluation protocol:

- **Engine-grouped cross-validation:** rows from one engine never appear in both training and validation.
- **Deployment-style cutoffs:** each validation engine is predicted at a random cutoff at least 30 cycles in, repeated five times, and the same cutoffs are used for every model so comparisons are paired.
- **Official test set:** last observed cycle of each test engine against `RUL_FD001.txt`, reported against both capped and uncapped true RUL.
- **Model and parameter selection by cross-validation only;** the test set was not used for tuning.
- **Fixed random seeds** and reference baselines (predict-the-mean, Ridge) so improvements can be quantified.
- **Metrics:** MAE, RMSE, bias, NASA score, and alert precision / recall / F1.

The 100-engine test set is small, so differences of a fraction of a cycle between models should not be over-interpreted.

---

## Current Limitations

### 1. Dataset Coverage

FD001 is a simulated dataset with a single operating condition and fault mode. The other C-MAPSS subsets (FD002–FD004) and real fleet data have not been evaluated.

### 2. Early RUL Prediction

Performance degrades when fewer engine observations are available, and early in life the sensors show little degradation, so some error is not removable. Predictions are capped at 125 cycles, so long-horizon RUL is not estimated.

### 3. Overconfident Intervals

The 10–90 prediction interval covered the true RUL 69% of the time in CV (nominal 80%). Intervals should be calibrated before being presented as probabilistic guarantees.

### 4. Residual Positive Bias

With the current features the average bias is smaller (about +2.6 to +3.2 cycles in CV) but still positive; interval-based alerts mitigate, but do not remove, the risk of late warnings.

### 5. Small Evaluation Sets

The test set has 100 engines (about 25 urgent), so alert recall and precision move in steps and the triage outcome is encouraging rather than conclusive.

### 6. Triage Thresholds

The 30 / 60 cycle thresholds and the 60-cycle review width are defaults, not values derived from real maintenance costs or lead times.

### 7. Anomaly Detection

A dedicated anomaly detection model has not yet been established as part of the pipeline.

### 8. MLOps Integration

Experiment tracking, production monitoring, and automated retraining are part of the planned architecture. Their integration has not yet been demonstrated.

### 9. Application Interface

The backend and frontend directories are reserved for future application development. A complete operational dashboard has not yet been established.

---

## Roadmap

### Phase 1: Strengthen the Baseline — done

- [x] Consolidate experiment configurations and results.
- [x] Establish a consistent, leakage-safe model-selection protocol.
- [x] Evaluate on the official test set.
- [x] Add per-engine feature engineering.
- [x] Add automated tests (triage rules).
- [ ] Add tests for preprocessing and evaluation code.

### Phase 2: Improve Prediction and Uncertainty

- [ ] Calibrate the prediction intervals (for example conformal prediction).
- [ ] Evaluate a GRU / 1D-CNN sequence model on 30-cycle windows.
- [ ] Revisit the near-failure sample weighting experiments with the new protocol.

### Phase 3: Develop Anomaly Detection

- [ ] Select suitable sensor features and a health index from early healthy cycles.
- [ ] Implement an anomaly detection baseline.
- [ ] Evaluate anomaly scores and combine them with RUL predictions.

### Phase 4: Harden the Triage Agent

- [x] Rule-based severity from RUL intervals.
- [x] Work-order generation with a human-review flag.
- [ ] Set thresholds with maintenance stakeholders and cost assumptions.
- [ ] Add anomaly scores as an input.

### Phase 5: Add MLOps

- [ ] Track model experiments and parameters using MLflow.
- [ ] Register and version candidate models.
- [ ] Monitor model performance and prediction behavior.
- [ ] Develop a controlled retraining workflow.

### Phase 6: Build the Application

- [ ] Interface for sensor data and RUL predictions.
- [ ] Visualize intervals, anomaly scores, and machine health.
- [ ] Display maintenance priorities and work-order status.

### Phase 7: Expand Evaluation

- [ ] Evaluate additional C-MAPSS subsets.
- [ ] Test robustness under different operating conditions.
- [ ] Document performance, limitations, and failure cases.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| scikit-learn | Regression, quantile models, cross-validation, metrics |
| Jupyter Notebook | Exploration and experimentation |
| Joblib | Serialization of trained model artifacts |
| pytest | Unit tests |
| Git and GitHub | Version control and source management |
| MLflow | Planned experiment and model tracking |
| Streamlit | Planned monitoring and visualization interface |

---

## References

1. **NASA Prognostics Center of Excellence — Data Repository**  
   NASA C-MAPSS dataset and related prognostics datasets:  
   https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

2. **scikit-learn Documentation**  
   https://scikit-learn.org/

3. **MLflow Documentation**  
   https://mlflow.org/docs/latest/

---

## Project Status

The project has moved from a raw-feature baseline to an evaluated pipeline that connects sensor history, interval RUL prediction, rule-based triage, and work-order generation on FD001. The next goals are calibrated intervals, an anomaly detection component, and tracking and monitoring around the model.

**The central objective is not only to predict when a machine may fail, but to turn those predictions into timely and actionable maintenance decisions.**
