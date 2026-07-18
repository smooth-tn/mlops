# mlops 
# Fraud Detection MLOps Pipeline

An end-to-end MLOps pipeline on **Azure ML** for real-time fraud detection, built on the [PaySim-style](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) synthetic mobile-money transactions dataset. Covers preprocessing, training, evaluation, champion/challenger model registration, and deployment to a managed online endpoint, wired together with a GitHub Actions CI/CD pipeline.

## Architecture

```
                 ┌────────────────┐
  raw_data ────▶ │  preprocess.py │ ── x_train/x_cv/x_test, encoder.pkl, scaler.pkl
                 └────────────────┘
                         │
                         ▼
                 ┌────────────────┐
                 │    train.py    │ ── trains XGBoost + Random Forest
                 │ (nested MLflow │    (nested runs under one parent
                 │      runs)     │     experiment run)
                 └────────────────┘
                         │  run_info.json (run IDs)
                         ▼
                 ┌────────────────┐
                 │  evaluate.py   │ ── scores both models on test set,
                 │                │    picks champion by F-beta (β=2),
                 │                │    registers champion + challenger
                 └────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │   Managed Online Endpoint         │
        │   fraud-detection-endpoint        │
        │   ├─ champion-endpoint  (80%)     │
        │   └─ challenger-endpoint (20%)    │
        └───────────────────────────────────┘
```

Each pipeline step is an Azure ML **command component**, wired together in `pipeline.yml` and submitted as a job. Deployment is a **Managed Online Endpoint** with two live deployments (champion/challenger) behind a traffic split, rather than a single fixed model — this is the "challenger" piece of the champion/challenger pattern, not a full offline A/B test with ground-truth comparison (see [Scope decisions](#scope-decisions)).

## Pipeline steps

- **`preprocess_data`** (`src/preprocess.py`) — loads raw transactions, engineers balance-discrepancy features (`transferred_amount_orig/dest`), target-encodes `type` (fit on train only), scales features, stratified 60/20/20 train/cv/test split
- **`train_model`** (`src/train.py`) — trains XGBoost and Random Forest in parallel as **nested MLflow runs** under a shared parent (the parent run is the Azure ML job's own experiment run), logs metrics/params/artifacts for each
- **`evaluate_model`** (`src/evaluate.py`) — scores both models on the held-out test set, requires min recall ≥ 0.8 and precision ≥ 0.25 to qualify, promotes the higher-F-beta model to `fraud-detection-champion` (Production stage) and the other to `fraud-detection-challenger` (Staging stage)

- ** `sweep_job` ** is done manually `src/train.py` is compatible with a sweep job the yaml default variables are the result of a manual sweep (they were hardcoded for simplicity, so the sweep_job flow is do a sweep_job in azure get the params and hardcode them in .yml)

Model selection uses **F-beta (β=2)**, weighting recall higher than precision — appropriate for fraud detection where missed fraud is costlier than false positives.

## Deployment

- `endpoint.yml` — defines the managed online endpoint
- `champion.yml` / `challenger.yml` — deployment configs, each pointing at its registered model, with its own tuned classification threshold set via environment variable
- `src/score.py` — shared scoring script for both deployments; supports a `{"request": "schema"}` introspection call so consumers can discover the expected input format without reading source code
- Both deployments enable Azure ML's **Data Collector** on inputs and outputs — this logs every request/response for auditability, but is intentionally *not* wired up to a drift-monitoring job (see below)

## CI/CD

`.github/workflows/main.yml` runs on pushes touching pipeline/component/deployment files:

1. **submit-pipeline** — authenticates via OIDC (federated credential, no stored secrets), submits `pipeline.yml` as an Azure ML job, injecting the raw data path from a GitHub secret at submission time (`--set inputs.raw_data.path=...`) rather than committing it
2. **deploy** — creates the endpoint if it doesn't exist, creates/updates both champion and challenger deployments, then sets the 80/20 traffic split

Environment: Azure ML workspace `amlmouhib`, resource group `mlopsmouhib`, West Europe. Auth via a dedicated App Registration with OIDC federated credentials (no long-lived secrets in CI).

## Scope decisions

A few things were deliberately left out, rather than left unfinished:

- **Full drift monitoring (Azure ML Model Monitoring on top of the Data Collector) was scoped out.** Detecting drift is only meaningful against real, time-varying production traffic; without that, standing up a monitor job would mean generating synthetic traffic just to trigger it, demonstrating that the wiring works, not that drift detection works. The Data Collector itself (input/output logging) is kept, since that's useful independent of monitoring. 
- **A/B testing with statistical significance was dropped.** The PaySim dataset has no ground-truth labels for live traffic, so there's no way to compute a real lift between champion and challenger post-deployment. The champion/challenger *deployment* pattern (traffic split, promotable via CI) is kept; the *evaluation* is done offline in `evaluate.py` against the held-out test set instead.

## Local setup

conda env create -f conda.yml
conda activate mlops_project


Requires an Azure ML workspace with a registered environment (`fraud-detection-env`) and compute target (you can rename it) matching the names in the `*.yml` files, plus GitHub secrets: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `DATA_PATH`.
