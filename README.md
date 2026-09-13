# Continuous Learning ML App

A Phase 6 full-stack application with a React + Vite frontend, FastAPI backend, SQLAlchemy persistence, registry-backed prediction, immutable ground-truth feedback, and safe threshold-based automatic full retraining. True online learning, distributed workers, authentication, and production deployment are intentionally not implemented.

## Prerequisites

- Python 3.11 or newer
- Node.js 20.19 or newer
- npm 10 or newer

## Configuration

From the repository root:

```bash
cp .env.example .env
```

The important development settings are:

```dotenv
DATABASE_URL=sqlite:///./app.db
MODEL_DIR=models
VITE_API_BASE_URL=http://localhost:8000
AUTO_RETRAIN_ENABLED=true
RETRAIN_MIN_NEW_SAMPLES=50
MIN_TRAINING_SAMPLES=100
PRIMARY_PROMOTION_METRIC=f1_score
MIN_PROMOTION_IMPROVEMENT=0.01
MIN_ACCEPTABLE_F1=0.60
```

The backend command below runs from `backend/`, so SQLite creates the ignored database at `backend/app.db`. `MODEL_DIR` is resolved from the repository root, placing ignored joblib artifacts in `models/`. FastAPI applies pending Alembic migrations at startup.

To apply migrations manually instead, run:

```bash
source .venv/bin/activate
cd backend
python -m alembic upgrade head
```

To use PostgreSQL later, install the relevant SQLAlchemy driver and replace `DATABASE_URL` with a PostgreSQL URL. Repository and service logic do not depend on SQLite-specific queries.

The frontend accesses FastAPI only through its shared API client and `VITE_API_BASE_URL`. Codespaces setup may replace local frontend/backend URLs automatically; FastAPI accepts the configured local origin and Codespaces port `5173` origin pattern.

## Install and run the backend

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-dev.txt
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation is available at `http://localhost:8000/docs`.

## Install and run the frontend

In a second terminal, from the repository root:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, or the forwarded port `5173` URL in Codespaces.

## Train the first model

First submit at least `MIN_TRAINING_SAMPLES` labelled records through the frontend or `POST /api/data`. Both binary labels (`0` and `1`) must be present. Then run training manually from the repository root:

```bash
source .venv/bin/activate
cd backend
python -m app.ml.train
```

Training saves the complete preprocessing-and-classifier `Pipeline` as a versioned joblib artifact and stores its path, SHA-256 checksum, metrics, status, and `model_version` in `training_runs`.

Phase 4 adds an `is_active` registry flag but does not automate promotion. Inference chooses the newest completed active row. If no completed row is active, it safely falls back to the latest completed training run. The loader validates registry metadata, constrains artifacts to `MODEL_DIR`, verifies the file and checksum, and caches the loaded Pipeline. Each request still checks the registry, so a different selected version is lazily loaded when the registry changes.

## Prediction API

Create and persist a prediction:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"feature_1":2.5,"feature_2":4.1,"feature_3":6.7}'
```

The response contains `prediction_id`, `prediction`, `predicted_class`, class-1 `probability` when supported, `model_version`, and `created_at`. `model_version` identifies the exact registered artifact used for inference.

Other endpoints:

- `GET /health`
- `POST /api/data`
- `GET /api/data?skip=0&limit=20`
- `GET /api/data/{id}`
- `GET /api/predictions?skip=0&limit=20`
- `GET /api/predictions/{prediction_id}`
- `PATCH /api/predictions/{prediction_id}/feedback`
- `GET /api/predictions/{prediction_id}/feedback`
- `GET /api/predictions/feedback/summary`
- `GET /api/training-data?labelled_only=true&unused_only=true&skip=0&limit=20`
- `POST /api/training/check`
- `GET /api/training/status`
- `GET /api/training/runs?skip=0&limit=20`
- `GET /api/model/status`

`GET /api/model/status` reports model availability, selected version, algorithm, creation time, and evaluation metrics without exposing its filesystem path.

Prediction history is stored separately in the `predictions` table. Each row records its input features, prediction, optional probability, and exact model version. `actual_label` starts as `NULL` because a prediction is not ground truth.

## Ground-truth feedback

A model prediction is an estimate, not a verified label. Phase 5 accepts an actual outcome only through an explicit feedback request:

```bash
curl -X PATCH http://localhost:8000/api/predictions/21/feedback \
  -H 'Content-Type: application/json' \
  -d '{"actual_label":0}'
```

Replace `21` with a real prediction ID. `actual_label` must be the integer `0` or `1`. A successful response reports the prediction, actual result, observed correctness, model version, and update time. Feedback can be read without changing it:

```bash
curl http://localhost:8000/api/predictions/21/feedback
```

Feedback is immutable through the public API. A second submission returns HTTP `409`; it cannot silently rewrite historical ground truth. Any future correction feature must be a dedicated audited workflow.

The feedback service handles the prediction update and verified-sample creation in one database transaction. It validates that all stored feature values are present and finite, stores `actual_label` on the prediction, and creates one row in `samples` with:

- the prediction's original feature values;
- `label` set from `actual_label`, never `predicted_class`;
- `source_prediction_id` pointing to the originating prediction;
- `used_for_training=false`; and
- no training batch or model version yet.

`samples.source_prediction_id` is a nullable foreign key with a unique constraint. Existing manually submitted samples may leave it null, while feedback-derived samples have one-to-one lineage and cannot be duplicated.

Inspect unused labelled records prepared for future training:

```bash
curl 'http://localhost:8000/api/training-data?labelled_only=true&unused_only=true&skip=0&limit=20'
```

View observed performance only on predictions with verified ground truth:

```bash
curl http://localhost:8000/api/predictions/feedback/summary
```

This summary is not full model evaluation. It reports feedback coverage and accuracy only for predictions whose actual outcomes were submitted.

Submitting feedback never fits a model inside the HTTP request. After committing feedback, Phase 6 performs a lightweight eligibility check and may reserve an automatic training run for development-oriented background execution.

### Manual feedback flow

With the backend running and a trained model available:

```bash
# 1. Create a prediction and note prediction_id from the response.
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"feature_1":2.5,"feature_2":4.1,"feature_3":6.7}'

# 2. Replace 21 with that prediction_id and submit the verified outcome.
curl -X PATCH http://localhost:8000/api/predictions/21/feedback \
  -H 'Content-Type: application/json' \
  -d '{"actual_label":0}'

# 3. Confirm the prediction feedback status.
curl http://localhost:8000/api/predictions/21/feedback

# 4. Confirm the lineage-linked unused training sample.
curl 'http://localhost:8000/api/training-data?labelled_only=true&unused_only=true&skip=0&limit=100'

# 5. Repeat step 2 to verify that duplicate feedback returns HTTP 409.
```

## Continuous training architecture

Phase 6 uses **threshold-based full retraining**. The current `LogisticRegression` estimator has no `partial_fit()` method, so the application does not pretend to perform incremental learning. When enough newly verified records accumulate, it fits a new LogisticRegression pipeline from scratch.

The workflow is:

1. Feedback commits an immutable verified sample and returns normally.
2. A lightweight eligibility check counts new verified trigger records.
3. Once both configured thresholds are satisfied, the system reserves one `queued` run.
4. Development background execution changes it to `running`, snapshots data, and closes database sessions before `model.fit()`.
5. The candidate and current serving model are evaluated on the same reproducible held-out rows.
6. A short final transaction either promotes or rejects the candidate and checkpoints the trigger batch.

### Trigger data versus the full dataset

Trigger records and training records have different meanings:

- A **new trigger record** has a valid `0`/`1` label, comes from verified prediction feedback, has `used_for_training=false`, and has no `last_triggered_training_run_id`.
- The **full training snapshot** contains every valid verified feedback sample available when the run is reserved, including historical records used by older models.

For example, 50 new records may trigger retraining, while the candidate fits on all 150 verified records. The reserved run stores both `trigger_sample_ids` and the complete `data_selection`, plus held-out `evaluation_sample_ids`, so these roles remain auditable.

`used_for_training` means that a sample has participated in a successfully fitted candidate, whether that candidate was promoted or rejected. `last_triggered_training_run_id` separately means that the record was part of the new-data signal consumed by a completed quality decision. New feedback arriving after reservation is not part of that snapshot and remains eligible for the next cycle.

### Candidate comparison and promotion

The current verified snapshot is deterministically split using `ML_TEST_SIZE` and `ML_RANDOM_STATE`. The candidate is fitted only on the training portion. Both the candidate and existing serving model are then evaluated on the exact same held-out portion; the active model is loaded as-is and is never retrained for comparison.

Promotion uses `PRIMARY_PROMOTION_METRIC`. A candidate must first satisfy `MIN_ACCEPTABLE_F1`. If there is a current model, the candidate primary metric must be at least the current model's same-split metric plus `MIN_PROMOTION_IMPROVEMENT`. An unavailable configured metric causes rejection. If no model is serving, a candidate meeting minimum quality may be promoted.

Promotion atomically deactivates every prior active row, marks the candidate `promoted`, sets `promoted_at`, and makes it active. The model loader cache is invalidated, so the next prediction loads the new version without restarting FastAPI.

Rejected candidates retain their artifact and registry identity for inspection. Their trigger batch is checkpointed so the same data does not cause an endless retry; another threshold of genuinely new feedback is required. The current model remains active.

Technical failures are different: the run becomes `failed`, its incomplete candidate artifact is removed, the current model remains active, and trigger records are not checkpointed or marked used. A later feedback event or manual eligibility check may retry them. There is no automatic retry loop.

### Training state and concurrency

Automatic runs move through `queued` → `running` → `promoted`, `rejected`, or `failed`. Older manual runs may use `training` or `completed`. A process lock protects the check/reservation section in the current single instance, while a database-unique `concurrency_slot` prevents two automatic queued/running reservations. Model versions and training batch IDs are also unique.

Run metadata records trigger type/count/IDs, complete training IDs, evaluation IDs, metrics, prior active model, same-split active metrics, decision reasons, timestamps, and active state. Artifact paths and checksums are not exposed by the run-history API.

Check eligibility without bypassing thresholds:

```bash
curl -X POST http://localhost:8000/api/training/check
```

Inspect status and paginated run history:

```bash
curl http://localhost:8000/api/training/status
curl 'http://localhost:8000/api/training/runs?skip=0&limit=20'
```

The React page contains a small status card with a manual Refresh button. It shows whether automatic retraining is enabled, new verified count, threshold, running state, active model, and last run.

### Development limitations

FastAPI `BackgroundTasks` is only the Phase 6 development scheduler. The orchestration and worker entry point are isolated from FastAPI, but an in-process job is not durable: a process restart can interrupt it, a crashed `queued`/`running` lease currently needs operator recovery, and multiple application instances need a real distributed lock. Phase 8 should use Celery, RQ, or another durable queue with Redis or equivalent coordination, retries, leases/timeouts, and worker monitoring.

If no completed model exists, prediction returns HTTP `503` with `No trained model available.` Missing, unreadable, checksum-mismatched, or corrupt artifacts also return a sanitized `503` response and are logged by the backend.

## Tests and builds

Run the full backend suite from the repository root:

```bash
source .venv/bin/activate
python -m pytest backend
```

Run the frontend production build:

```bash
cd frontend
npm run build
```

Backend tests use a temporary SQLite database and temporary model directory. They never modify `backend/app.db` or the real `models/` artifacts.
