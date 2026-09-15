# Continuous Learning ML App

A Phase 8 full-stack application with a React + Vite frontend, FastAPI backend, SQLAlchemy persistence, registry-backed prediction, immutable ground-truth feedback, Celery/Redis retraining workers, model monitoring, and safe manual rollback. Authentication and production deployment are intentionally not implemented.

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
DRIFT_WINDOW_SIZE=200
DRIFT_MIN_SAMPLES=50
DRIFT_PSI_WARNING=0.10
DRIFT_PSI_CRITICAL=0.25
PERFORMANCE_WINDOW_SIZE=100
PERFORMANCE_MIN_FEEDBACK_SAMPLES=30
PERFORMANCE_WARNING_DROP=0.05
PERFORMANCE_CRITICAL_DROP=0.10
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
REDIS_URL=redis://localhost:6379/2
CELERY_TASK_ALWAYS_EAGER=false
TRAINING_QUEUE_NAME=training
TRAINING_TASK_MAX_RETRIES=3
TRAINING_TASK_RETRY_DELAY_SECONDS=60
TRAINING_LOCK_TIMEOUT_SECONDS=3600
TRAINING_TASK_SOFT_TIME_LIMIT_SECONDS=3300
TRAINING_TASK_TIME_LIMIT_SECONDS=3600
TRAINING_STALE_TIMEOUT_SECONDS=3600
WORKER_HEALTH_TIMEOUT_SECONDS=1.0
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

### Phase 6 historical limitation

Phase 6 originally used FastAPI `BackgroundTasks`. Phase 8 removes that execution path: FastAPI now only commits a reserved job and publishes its identifier to Celery.

If no completed model exists, prediction returns HTTP `503` with `No trained model available.` Missing, unreadable, checksum-mismatched, or corrupt artifacts also return a sanitized `503` response and are logged by the backend.

## Model monitoring

Phase 7 deliberately reports two independent signals:

- **Data drift** asks whether recent input-feature distributions differ from the data used to fit the active model. It does not require labels and does not prove that model quality changed.
- **Performance drift** asks whether predictions with later, verified outcomes perform worse than the active model's promotion-time validation baseline. Predictions without both `feedback_received=true` and a non-null `actual_label` are excluded from accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix.

Each successfully trained candidate stores compact per-feature reference statistics and a 10-bin histogram in `model_data_profiles`; raw training rows are not duplicated in the registry. Drift compares those histograms with the latest `DRIFT_WINDOW_SIZE` predictions made by the active model using Population Stability Index (PSI). With the defaults, PSI below `0.10` is stable, PSI from `0.10` to below `0.25` is warning, and PSI at least `0.25` is critical. Empty histogram bins are smoothed safely. Fewer than `DRIFT_MIN_SAMPLES` observations returns `insufficient_data`. Overall drift is critical if any feature is critical, otherwise warning if any feature is warning, otherwise stable.

Performance metrics use the latest `PERFORMANCE_WINDOW_SIZE` verified-feedback predictions. Fewer than `PERFORMANCE_MIN_FEEDBACK_SAMPLES` produces `insufficient_data`. The service compares `PRIMARY_PROMOTION_METRIC`—without silently changing metrics—to the active model's stored validation value. A drop below `PERFORMANCE_WARNING_DROP` is stable, a drop up to the critical threshold is warning, and a drop at least `PERFORMANCE_CRITICAL_DROP` is critical. ROC-AUC is null unless probabilities and both actual classes are available.

Feedback coverage is the fraction of predictions in the recent relevant window that have verified outcomes. Low or biased feedback coverage can make observed performance unrepresentative even when enough labels exist.

Model health follows one policy: a missing/invalid active artifact or critical performance is unhealthy; warning/critical data drift or warning performance is warning; otherwise it is healthy. The response also includes the latest training status, automatic-retraining progress, new verified sample count, feedback coverage, and a non-automatic rollback recommendation for critical performance.

Monitoring reads do not mutate the database:

- `GET /api/monitoring/drift`
- `GET /api/monitoring/performance`
- `GET /api/monitoring/health`

Run and persist one explicit three-part monitoring snapshot:

```bash
curl -X POST http://localhost:8000/api/monitoring/check
```

This stores `data_drift`, `performance`, and `health` snapshots. Monitoring never trains, promotes, deactivates, or rolls back a model.

## Model history, comparison, and rollback

Registry inspection endpoints are paginated where applicable:

- `GET /api/models?skip=0&limit=20`
- `GET /api/models/{model_version}`
- `GET /api/models/compare?model_a=model_v1&model_b=model_v2`

Model detail includes metadata, metrics, lifecycle events, and compact data profiles, but never exposes artifact paths or checksums. Comparison reports numeric metric differences and explicitly warns that historical validation results may come from different held-out datasets, so they are not necessarily statistically comparable.

To switch back to a valid completed/promoted version:

```bash
curl -X POST http://localhost:8000/api/models/model_v1/rollback \
  -H 'Content-Type: application/json' \
  -d '{"reason":"Performance regression observed"}'
```

Before changing state, rollback verifies registry status, artifact location, SHA-256 checksum, joblib deserialization, scikit-learn Pipeline type, binary prediction behavior, and the current three-feature input contract. It then deactivates all current active rows, activates the target, and records a `rollback` event in one database transaction. Only after commit does it invalidate the inference cache, so the next prediction uses the selected version without a restart. It neither retrains nor deletes any artifact, changes samples, or consumes training triggers.

**The rollback endpoint is an unauthenticated development/admin operation in Phase 7 and must be protected by authentication and authorization before production use.** The UI therefore requires confirmation, but a browser dialog is not a security boundary.

### Phase 7 limitations

- PSI detects distribution change; it does not establish cause or prove performance degradation.
- Performance monitoring depends on delayed ground truth, and selective/biased feedback can bias every reported metric.
- Historical validation metrics may use different evaluation datasets.
- Monitoring executes inside the current FastAPI process; durable scheduling, alert delivery, Celery/Redis workers, retries, and distributed coordination are not implemented.
- There is no automatic rollback, canary deployment, A/B testing, or authentication.

## Phase 8 distributed training architecture

```text
React frontend
      ↓ HTTP
FastAPI ───────→ SQLAlchemy database (authoritative job/model state)
      │
      └── training_run_id → Celery training queue → Redis broker
                                                    ↓
                                             Celery worker (1 process)
                                                    ↓
                               existing RetrainingService / ML Pipeline
                                                    ↓
                                  evaluation → promotion or rejection
                                                    ↓
                              SQL model registry + shared model artifacts
```

FastAPI never calls `model.fit()` and never waits for training. Feedback is committed first; when the threshold is reached, the coordinator commits a `queued` training run and the dispatcher sends only its integer ID to the dedicated `training` queue. The Celery worker creates its own SQLAlchemy sessions and invokes the existing retraining, evaluation, promotion, and registry services.

Redis has three isolated logical uses configured by environment variables: database `0` is the Celery broker, database `1` is the concise task-result backend, and database `2` holds infrastructure health/locking keys. SQL remains the source of truth for training status, retries, models, and metrics; the browser never connects to Redis or Celery.

### Start the complete development system

Install dependencies once from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements-dev.txt
cd frontend
npm install
```

Terminal 1 — start locally bound Redis:

```bash
docker compose up redis
```

Terminal 2 — migrate and start FastAPI:

```bash
source .venv/bin/activate
cd backend
python -m alembic upgrade head
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 3 — start the training worker:

```bash
source .venv/bin/activate
cd backend
celery -A app.workers.celery_app:celery_app worker --loglevel=INFO -Q training --concurrency=1
```

Concurrency is deliberately `1` because fitting is CPU-intensive and the project permits only one candidate run at a time.

Terminal 4 — start React:

```bash
cd frontend
npm run dev
```

Optional local Flower monitoring:

```bash
source .venv/bin/activate
cd backend
celery -A app.workers.celery_app:celery_app flower --port=5555
```

Flower is diagnostic only. Do not expose it publicly without authentication.

### Queue reliability, locking, and retries

Tasks use JSON messages/results, UTC timestamps, started-state tracking, late acknowledgement, worker-lost rejection, and a prefetch multiplier of one. Late acknowledgement allows redelivery after a worker crash, so the task checks SQL state before execution and skips terminal runs. Promotion remains an atomic SQL transaction; a failed or duplicate task cannot deliberately promote twice.

An expiring Redis `ml:training:lock` prevents concurrent fitting across worker processes. Lock acquisition is an atomic `SET NX EX`, and release uses a compare-and-delete Lua script with a unique ownership token, so one worker cannot release another worker's lock. The timeout prevents a crashed worker from creating a permanent Redis deadlock. SQL's unique automatic-training slot remains an additional reservation guard.

Transient Redis/network, database-connectivity, and filesystem failures retry up to `TRAINING_TASK_MAX_RETRIES` with exponential backoff and jitter. Dataset validation, insufficient class diversity, and ordinary candidate rejection are not retried. `retry_count`, sanitized last error, `last_retry_at`, `started_at`, `heartbeat_at`, and high-level `progress_stage` values are persisted. Soft time limits are recorded as failures; a hard-killed job is exposed as stale after `TRAINING_STALE_TIMEOUT_SECONDS` for controlled investigation.

The queued SQL row is also a lightweight recoverable dispatch record. A known broker publish failure leaves feedback committed and the run queued with `dispatch_error`. Recover it with:

```bash
curl -X POST http://localhost:8000/api/training/reconcile
```

There remains a small publish/SQL-recording crash window because this is not a full transactional broker outbox. Reconciliation may redispatch a job, and task idempotency plus the distributed lock make that at-least-once behavior safe. Terminal and already-running jobs are not reconciled.

### Job and infrastructure status

- `POST /api/training/check` returns `202 Accepted`, `training_run_id`, and Celery `task_id` when dispatched.
- `GET /api/training/status` reports queued/running counts, current stage, active model, latest result, verified sample count, Redis health, and worker health.
- `GET /api/training/runs/{training_run_id}` reads durable SQL job state and flags stale running work without exposing tracebacks.
- `POST /api/training/reconcile` redispatches recoverable queued jobs only.
- `GET /api/system/worker-health` separately reports Redis and Celery-worker availability.
- `GET /health` remains the API-process health check; worker outages do not make prediction unavailable.

If Redis is unavailable, prediction continues normally and feedback remains stored. Dispatch failure is logged and recoverable from SQL. If a worker fails during training, the currently active model is not deactivated, trigger samples are not consumed before finalization, the lock expires, and the run is retryable or eventually marked failed.

FastAPI and the worker may be separate processes, so in-memory cache invalidation is not relied upon. Every prediction asks the SQL registry for the active version; `ModelLoader` reuses its cache only when version and checksum still match. A worker promotion is therefore detected by the next API prediction without an application restart.

### Shared storage and production notes

The documented host-based setup runs FastAPI and Celery from the same repository, so both see the same `MODEL_DIR` and `backend/app.db`. If backend/worker containers are added, they must mount one shared artifact volume and use the exact same database URL. Never let each container create its own SQLite file.

SQLite is suitable for this single-worker development setup but has limited write concurrency. PostgreSQL is strongly recommended for production and multi-instance deployments. Production also needs authentication/authorization around administrative endpoints, a real transactional outbox or equivalent dispatcher, managed shared/object artifact storage, secret management, alerting, and operational worker supervision.

### Troubleshooting

- `redis_available=false`: run `docker compose up redis`, confirm port 6379 is not occupied, and check `REDIS_URL`/broker URLs.
- `celery_worker_available=false`: start the worker with the exact command above and confirm it consumes `-Q training`.
- A queued job has `dispatch_error`: restore Redis and call `POST /api/training/reconcile`.
- A running job has `is_stale=true`: inspect worker logs using its run/task IDs; do not manually promote artifacts.
- Worker cannot see a model: ensure FastAPI and Celery share the same absolute `MODEL_DIR` and filesystem permissions.
- SQLite lock errors under concurrency: keep worker concurrency at one for development or move `DATABASE_URL` to PostgreSQL.

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
