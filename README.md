# DQN1 — Urban Air Quality & Health Risk Prediction (GBDT Baseline)

This repository implements a Gradient-Boosted Decision Trees (GBDT) baseline for two supervised tasks on the DQN1 dataset:
(1) forecast PM2.5 from weather and lagged pollutant features; (2) predict HealthRiskScore from weather, pollutant, and calendar features.

Foundational reference: Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
We follow the textbook’s guidance on model selection, bias–variance, and evaluation framing.

## Quick Start
1) Environment
- Create venv and install: `pip install -r requirements.txt`.
2) Run training & evaluation
- Put the Excel at repo root as `DQN1 Dataset.xlsx` (or pass a path) and run:
  `python src/train_and_eval.py --data "DQN1 Dataset.xlsx" --sheet "Data" --test_ratio 0.2 --outdir artifacts`
3) Artifacts
- `artifacts/metrics_pm25.json`, `artifacts/metrics_health.json`
- `artifacts/feature_importance_pm25.csv`, `feature_importance_health.csv`
- `artifacts/pm25_actual_vs_pred.png`, `health_actual_vs_pred.png`
- `artifacts/summary.json`

## GitLab (Rubric A)
- Initialize git, connect remote, and make rubric-aligned commits:
  C1/C2 for implementation & repo scaffolding; D2 for metrics artifacts; D3 for report; then export branch history.

## References
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

