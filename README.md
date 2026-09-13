# Continuous Learning ML App

A Phase 4 full-stack application with a React + Vite frontend, FastAPI backend, SQLAlchemy persistence, manual scikit-learn training, and registry-backed prediction. Continuous or automatic retraining, training on predictions, background workers, model promotion automation, authentication, and automatic label collection are intentionally not implemented.

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
- `GET /api/model/status`

`GET /api/model/status` reports model availability, selected version, algorithm, creation time, and evaluation metrics without exposing its filesystem path.

Prediction history is stored separately in the `predictions` table. Each row records its input features, prediction, optional probability, and exact model version. `actual_label` starts as `NULL` because a prediction is not ground truth; labels and feedback are not collected automatically in this phase.

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
