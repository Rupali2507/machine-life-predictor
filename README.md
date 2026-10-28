# Machine Life Predictor
### Predictive Maintenance for a Connected Fleet

An experimental predictive maintenance system that uses machine sensor data to estimate **Remaining Useful Life (RUL)** and investigate early indicators of machine degradation.

This project is being developed for **TELIPORT Season 4 — Case Study 1: Self-Healing Machines: Predictive Maintenance for a Connected Fleet**.

The objective is to progress from a reproducible machine-learning baseline toward an end-to-end predictive maintenance pipeline that can detect abnormal behavior, estimate machine health, prioritize maintenance actions, and support automated work-order generation.

> **Current status:** Dataset exploration, preprocessing, baseline RUL prediction models, multi-split evaluation, and sensor trajectory analysis have been completed or explored. The complete automated maintenance workflow is still under development.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Current Implementation](#current-implementation)
- [Machine Learning Approach](#machine-learning-approach)
- [Experimental Results](#experimental-results)
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

1. **Detect:** Identify unusual machine behavior and potential degradation.
2. **Predict:** Estimate Remaining Useful Life using sensor measurements.
3. **Prioritize:** Apply a rule-based triage agent to determine maintenance severity.
4. **Automate:** Generate maintenance work orders and escalate exceptional cases for human review.

The long-term objective is to minimize manual intervention while maintaining clear monitoring, evaluation, and maintenance decision logic.

---

## Project Objectives

### Completed and explored

- Load and explore the NASA C-MAPSS FD001 dataset.
- Validate dataset dimensions, missing values, duplicate rows, and sensor behavior.
- Construct RUL targets from engine operating cycles.
- Identify and remove constant features.
- Create an engine-level training and validation split.
- Train Random Forest and HistGradientBoosting regression baselines.
- Evaluate prediction errors using MAE, RMSE, and prediction bias.
- Compare model performance across multiple random seeds and observation cutoffs.
- Investigate sensor trajectories associated with large RUL prediction errors.
- Explore temporal features and bias correction as potential improvements.

### Planned

- Implement a dedicated anomaly detection component.
- Develop a rule-based maintenance triage agent.
- Integrate automatic work-order generation.
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

FD001 is the initial dataset selected for this project.

| Dataset component | Shape / count |
|---|---:|
| Training observations | 20,631 rows |
| Training engines | 100 |
| Test observations | 13,096 rows |
| Test engines | 100 |
| Test RUL labels | 100 |

The training data contains multivariate engine sensor measurements recorded over operating cycles.

The dataset includes:

- Engine identifiers.
- Operating cycles.
- Three operational setting columns.
- Twenty sensor measurement columns.

### RUL Target Construction

For each engine in the training data, the RUL target is calculated as:

`RUL = Maximum cycle for the engine - Current cycle`

For example, if an engine's final observed training cycle is 200, its RUL at cycle 150 is 50 cycles.

A capped target was also explored:

`RUL_capped = min(RUL, 150)`

This limits large early-life target values and provides an alternative regression target.

**Important:** The target calculation above describes the training-label construction. Test-set RUL must be evaluated using the separate provided test labels rather than treating each engine's last observed test cycle as its failure cycle.

### Data Quality Checks

Initial exploration of the training data found:

- No missing values.
- No duplicate rows.
- Multiple sensors with constant values.
- Different engine lifetimes and sensor trajectories.

Seven constant columns were excluded from the baseline feature set:

- `setting_3`
- `sensor_1`
- `sensor_5`
- `sensor_10`
- `sensor_16`
- `sensor_18`
- `sensor_19`

This leaves 18 features for the baseline models.

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

---

## Machine Learning Approach

### Baseline Models

Two regression algorithms have been evaluated.

#### Random Forest Regressor

Random Forest combines predictions from multiple decision trees.

The baseline configuration includes:

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `min_samples_leaf` | 2 |
| `random_state` | 42 |
| `n_jobs` | -1 |

#### HistGradientBoosting Regressor

HistGradientBoosting builds an ensemble of decision trees sequentially, with each stage improving on the previous ensemble.

The baseline configuration includes:

| Parameter | Value |
|---|---:|
| `max_iter` | 200 |
| `learning_rate` | 0.1 |
| `max_leaf_nodes` | 15 |
| `l2_regularization` | 1.0 |
| `random_state` | 42 |

The internal early-stopping mechanism stopped the baseline fit at 100 iterations in the recorded experiment.

### Evaluation Metrics

The models are evaluated using three metrics.

**Mean Absolute Error (MAE)**

Measures the average absolute difference between predicted and actual RUL.

Lower MAE indicates more accurate predictions on average.

**Root Mean Squared Error (RMSE)**

Penalizes large prediction errors more heavily than MAE.

Lower RMSE indicates fewer or smaller large errors.

**Prediction Bias**

Defined as:

`Bias = Mean(Predicted RUL - Actual RUL)`

Interpretation:

- Positive bias: the model overestimates RUL on average.
- Negative bias: the model underestimates RUL on average.
- Bias near zero: predictions are approximately unbiased on average.

For predictive maintenance, overestimating RUL can be particularly concerning because it may suggest that a machine has more operating life remaining than it actually does.

---

## Experimental Results

### Five-Split Baseline Evaluation

The baseline models were evaluated across five engine-level train/validation splits using random seeds 42, 21, 7, 100, and 2026.

The following results use the 80% observation cutoff configuration.

| Model | Mean MAE | MAE Std. Dev. | Mean RMSE | RMSE Std. Dev. | Mean Bias |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 14.11 | 2.00 | 18.17 | 2.69 | +8.06 |
| Random Forest | 14.62 | 1.99 | 19.08 | 3.30 | +9.69 |

*RUL errors and bias are measured in operating cycles.*

### Interpretation

HistGradientBoosting performed slightly better than Random Forest in this evaluation:

- Mean MAE was lower by approximately 0.51 cycles.
- Mean RMSE was lower by approximately 0.91 cycles.
- Mean positive bias was lower by approximately 1.63 cycles.

Both models exhibited positive average bias, indicating a tendency to overestimate RUL under this evaluation setup.

The differences are modest. These results support using HistGradientBoosting as the leading baseline candidate, but do not establish that it will outperform Random Forest in every setting.

### Observation Cutoff Experiment

The HistGradientBoosting baseline was also evaluated at different observation cutoffs.

The cutoff determines how much of each engine's trajectory is available before making an RUL prediction.

| Observation cutoff | Mean MAE | MAE Std. Dev. | Mean RMSE | Mean Bias |
|---|---:|---:|---:|---:|
| 60% | 19.09 | 3.75 | 23.35 | +6.71 |
| 70% | 17.81 | 3.57 | 22.65 | +6.64 |
| 80% | 14.11 | 2.00 | 18.17 | +8.06 |
| 90% | 6.53 | 1.20 | 8.45 | +2.86 |

These results show that predictions become more accurate when more of the engine trajectory is observed in this experiment.

However, this is not evidence that the model provides equally accurate early warnings. Predicting RUL with 90% of the trajectory available is a different task from predicting it with only 60% available.

The 60% and 70% settings have larger errors, highlighting the challenge of estimating RUL earlier in the degradation process.

The reported results are experimental validation results, not a claim of performance on unseen test-set engines.

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

### Temporal Feature Experiment

Temporal features were explored for sensor 9 and sensor 11, including:

- Rolling means.
- Rolling standard deviations.
- First differences.

These features were evaluated as possible ways to represent local sensor trends.

In the recorded baseline comparison, the temporal-feature configuration did not improve performance:

| Configuration | MAE | RMSE | Bias |
|---|---:|---:|---:|
| Baseline | 13.14 | 16.38 | +10.01 |
| Temporal features | 14.26 | 17.67 | +9.50 |

The temporal-feature experiment was therefore not adopted as an improvement to the baseline.

### Bias Correction Experiment

A calibration-based bias correction was also explored.

In that experiment, the estimated calibration bias was approximately +20.73 cycles.

The recorded validation results were:

| Metric | Before correction | After correction |
|---|---:|---:|
| MAE | 17.61 | 17.41 |
| RMSE | 22.50 | 20.63 |
| Bias | +12.31 | -8.43 |

The correction slightly improved MAE and RMSE in that experiment but shifted the bias from positive to negative.

Because the correction overcompensated for the original positive bias, it was not adopted as the default baseline.

These results come from a separate experimental setup and should not be directly compared with the five-split baseline table.

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
│   └── fd001_random_forest_cap150.joblib
│
├── notebooks/
│   ├── 01_fd001_data_exploration.ipynb
│   └── 02_reproducible_baseline.ipynb
│
├── reports/
│   ├── figures/
│   ├── fd001_baseline_metrics.json
│   ├── fd001_experiment_metrics_by_split.csv
│   ├── fd001_experiment_predictions.csv
│   ├── fd001_experiment_summary.csv
│   ├── fd001_near_failure_weight_metrics.csv
│   ├── fd001_near_failure_weight_predictions.csv
│   ├── fd001_near_failure_weight_summary.csv
│   ├── fd001_sensor_pattern_engine_comparison.csv
│   ├── fd001_sensor_pattern_group_summary.csv
│   ├── fd001_sensor_trend_metrics_by_split.csv
│   ├── fd001_sensor_trend_near_failure.csv
│   ├── fd001_sensor_trend_predictions.csv
│   ├── fd001_sensor_trend_summary.csv
│   └── fd001_validation_predictions.csv
│
├── src/
│   ├── agent/
│   ├── data/
│   ├── evaluation/
│   ├── compare_rul_experiments.py
│   ├── compare_sensor_patterns.py
│   ├── evaluate_near_failure_weights.py
│   └── evaluate_sensor_trends.py
│
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

Some directories are reserved for future components, and the generated reports may vary as experiments evolve.

---

## Getting Started

### Prerequisites

- Python 3.10 or later is recommended; use a version compatible with the installed dependencies.
- Git.
- A local copy of the NASA C-MAPSS dataset.

The baseline experiments can be run on a CPU; a GPU is not required for the current models.

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

If you encounter dependency issues, verify the package versions and Python version expected by the project.

### 4. Prepare the Dataset

Download the NASA C-MAPSS dataset from the official NASA Prognostics Center of Excellence data repository.

Place the FD001 training data, test data, and test RUL labels in the project’s expected directory structure.

For example:

```text
data/
└── raw/
    ├── train/
    ├── test/
    └── Rul/
```

The exact filenames and file extensions must match the loading paths in the notebooks or scripts.

**Dataset note:** The raw dataset is not required to be committed to GitHub. Download it separately and keep the local data directories excluded from version control where appropriate.

### 5. Run the Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Run the notebooks in this order:

1. `notebooks/01_fd001_data_exploration.ipynb`
2. `notebooks/02_reproducible_baseline.ipynb`

The first notebook explores the dataset and constructs the target. The second notebook runs the reproducible baseline workflow.

### 6. Run Experiment Scripts

The repository also contains scripts for additional experiments.

For example:

```bash
python src/compare_rul_experiments.py
python src/compare_sensor_patterns.py
python src/evaluate_near_failure_weights.py
python src/evaluate_sensor_trends.py
```

Run these scripts from the repository root. Ensure that their required input files and dependencies are available.

The generated reports are written under `reports/` according to each script's configuration.

---

## Reproducibility and Evaluation

Reproducibility is important because machine-learning performance can change depending on how engines are split into training and validation sets.

The project uses fixed random seeds in its recorded baseline experiments.

The evaluation process considers:

- Engine-level separation between training and validation data.
- MAE and RMSE.
- Prediction bias.
- Multiple random train/validation splits.
- Different observation cutoffs.
- Prediction errors for individual engines.
- Sensor trajectories associated with selected prediction errors.

The five-split experiment provides a more informative view of variability than relying on a single split.

Nevertheless, these results remain preliminary. Further work is needed to evaluate the final model on the official test set and establish a consistent, leakage-safe evaluation protocol for model selection and tuning.

---

## Current Limitations

The project is still under development.

### 1. Early RUL Prediction

Performance degrades when fewer engine observations are available. Improving early-life RUL prediction is an important research direction.

### 2. Positive Prediction Bias

The baseline models tend to overestimate RUL on average in the recorded five-split experiment.

This is important for maintenance applications because excessive RUL estimates could delay intervention.

### 3. Temporal Feature Engineering

The initial temporal-feature experiment did not improve the baseline metrics. Additional feature engineering should be guided by validation results rather than assumptions about sensor behavior.

### 4. Anomaly Detection

A dedicated anomaly detection model has not yet been established as part of the end-to-end pipeline.

### 5. Automated Maintenance Actions

The rule-based triage agent and automatic work-order generation are planned components, not completed features.

### 6. MLOps Integration

Experiment tracking, production monitoring, and automated retraining are part of the planned architecture. Their complete integration has not yet been demonstrated.

### 7. Application Interface

The backend and frontend directories are reserved for future application development. A complete operational dashboard has not yet been established.

### 8. Dataset Coverage

FD001 is the initial dataset. The remaining C-MAPSS subsets have not yet been established as part of the completed evaluation.

---

## Roadmap

The next development stages are:

### Phase 1: Strengthen the Baseline

- Consolidate experiment configurations and results.
- Recheck feature and target construction.
- Establish a consistent model-selection protocol.
- Evaluate the selected model on the official test set.
- Investigate large overestimates and near-failure prediction errors.
- Add automated tests for preprocessing and evaluation.

### Phase 2: Develop Anomaly Detection

- Select suitable sensor features.
- Implement an anomaly detection baseline.
- Evaluate anomaly scores and their usefulness for maintenance decisions.
- Combine anomaly information with RUL predictions.

### Phase 3: Build the Rule-Based Triage Agent

- Define maintenance severity rules.
- Use RUL predictions and anomaly scores as inputs.
- Generate prioritized maintenance actions.
- Escalate exceptional or uncertain cases for human review.

### Phase 4: Add MLOps

- Track model experiments and parameters using MLflow.
- Register and version candidate models.
- Monitor model performance and prediction behavior.
- Develop a controlled retraining workflow.

### Phase 5: Build the Application

- Develop an interface for sensor data and RUL predictions.
- Visualize anomaly scores and machine health.
- Display maintenance priorities and work-order status.
- Connect the model, triage agent, and maintenance workflow.

### Phase 6: Expand Evaluation

- Evaluate additional C-MAPSS subsets.
- Test the robustness of the approach under different operating conditions.
- Compare model performance across observation cutoffs.
- Document performance, limitations, and failure cases.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| scikit-learn | Baseline regression models and evaluation |
| Jupyter Notebook | Exploration and experimentation |
| Joblib | Serialization of trained model artifacts |
| Git and GitHub | Version control and source management |
| MLflow | Planned experiment and model tracking |
| Streamlit | Planned monitoring and visualization interface |

The current baseline work primarily uses Python, Pandas, NumPy, scikit-learn, and Jupyter Notebook. The remaining components will be integrated as development progresses.

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

This repository represents the initial research and experimentation stage of a predictive maintenance system.

The work completed so far establishes a reproducible starting point for RUL prediction and investigates the limitations of baseline models. The next goal is to develop a reliable, evaluated pipeline that connects sensor data, anomaly detection, RUL prediction, rule-based triage, and maintenance actions.

**The central objective is not only to predict when a machine may fail, but to turn those predictions into timely and actionable maintenance decisions.**
