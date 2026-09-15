# Continuous Learning ML App

A full-stack machine-learning operations application with a production-style React dashboard, FastAPI API, SQLAlchemy persistence, Redis/Celery training jobs, model monitoring, registry-backed inference, and immutable ground-truth feedback.

## Technology stack

- Frontend: React 19, TypeScript, Vite, Tailwind CSS
- API: FastAPI, Pydantic
- Persistence: SQLAlchemy 2.x, Alembic, SQLite for development
- ML: scikit-learn Pipeline, joblib model artifacts
- Background work: Celery with Redis
- Tests: pytest and FastAPI TestClient

## Architecture

```text
React + Tailwind dashboard
          |
          | HTTP / JSON
          v
       FastAPI
          |
          +---- SQLAlchemy ---- SQLite / PostgreSQL
          |
          +---- Celery queue ---- Redis ---- Training worker
                                              |
                                              v
                                Candidate evaluation
                                              |
                                   promotion / rejection
                                              |
                                   model registry + artifact
```

FastAPI handles HTTP requests but never fits a model inside a request. SQL is the source of truth for data, predictions, feedback, job state, metrics, and the active model. Model binaries remain outside database rows under the ignored `models/` directory.

## Prerequisites

Install these before using the local setup:

- Python 3.11 or newer
- Node.js 20.19 or newer
- npm 10 or newer
- Docker with Docker Compose

GitHub Codespaces already provides Python, Node.js, npm, Docker, and automatic port forwarding.

## Local setup

All commands below start from the repository root.

### 1. Create the environment file

```bash
cp .env.example .env
```

The development defaults use:

```dotenv
DATABASE_URL=sqlite:///./app.db
VITE_API_BASE_URL=http://localhost:8000
FRONTEND_ORIGIN=http://localhost:5173
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
REDIS_URL=redis://localhost:6379/2
MODEL_DIR=models
```

Because backend commands run from `backend/`, SQLite is created at `backend/app.db`. The file is ignored by Git.

### 2. Install backend dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-dev.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Start Redis — terminal 1

```bash
docker compose up redis
```

Redis is bound to `127.0.0.1:6379` and stores its development data in the `redis_data` Docker volume.

### 5. Start FastAPI — terminal 2

```bash
source .venv/bin/activate
cd backend
python -m alembic upgrade head
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- API health: <http://localhost:8000/health>
- Swagger API documentation: <http://localhost:8000/docs>
- ReDoc documentation: <http://localhost:8000/redoc>

The application also applies pending Alembic migrations during startup. Running `alembic upgrade head` explicitly makes startup state clear and is safe to repeat.

### 6. Start the Celery worker — terminal 3

```bash
source .venv/bin/activate
cd backend
celery -A app.workers.celery_app:celery_app worker --loglevel=INFO -Q training --concurrency=1
```

Keep worker concurrency at `1` with SQLite. It avoids unnecessary write contention and the application permits only one automatic candidate training run at a time.

### 7. Start the React frontend — terminal 4

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open <http://localhost:5173>.

## GitHub Codespaces setup

The repository contains a dev-container definition and `.devcontainer/setup-codespaces.sh`. Every time the Codespace attaches, the script:

1. Detects the current Codespace name and forwarding domain.
2. Writes the current port `5173` URL to `FRONTEND_ORIGIN`.
3. Writes the current port `8000` URL to `VITE_API_BASE_URL`.
4. Keeps the localhost frontend origin as a CORS fallback.
5. Starts the Redis Compose service on the Codespace VM.
6. Attempts to make backend port `8000` public so browser API requests do not redirect to GitHub authentication.

You do not need to manually paste Codespaces URLs into `.env`. FastAPI and Celery run inside the Codespace VM, so their Redis URLs intentionally remain `redis://localhost:6379/...`. Only URLs used by the browser need the forwarded Codespaces HTTPS domain.

### 1. Verify the generated URLs

From the Codespaces terminal:

```bash
bash .devcontainer/setup-codespaces.sh
grep -E '^(FRONTEND_ORIGIN|VITE_API_BASE_URL)=' .env
```

Expected format:

```text
FRONTEND_ORIGIN=https://<codespace-name>-5173.app.github.dev
VITE_API_BASE_URL=https://<codespace-name>-8000.app.github.dev
```

If organization policy prevents the script from changing visibility, open the **Ports** tab, right-click port `8000`, choose **Port Visibility**, then select **Public**. Port `5173` can remain private because you open it through your signed-in Codespaces session.

### 2. Install dependencies once

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-dev.txt
cd frontend
npm install
cd ..
```

### 3. Start all services

Open four Codespaces terminals and run the same service commands used locally.

Terminal 1:

```bash
docker compose up redis
```

Terminal 2:

```bash
source .venv/bin/activate
cd backend
python -m alembic upgrade head
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 3:

```bash
source .venv/bin/activate
cd backend
celery -A app.workers.celery_app:celery_app worker --loglevel=INFO -Q training --concurrency=1
```

Terminal 4:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open the forwarded **React frontend** URL from the Codespaces **Ports** tab. Do not open `localhost:5173` in your own computer browser when the application runs inside Codespaces.

## Stop the application

Stop FastAPI, Celery, and Vite with `Ctrl+C` in their terminals. Stop Redis without deleting its volume:

```bash
docker compose down
```

To remove the local Redis volume too:

```bash
docker compose down --volumes
```

The second command deletes Redis development data and should only be used when a clean Redis state is wanted.

## Database configuration

Local development uses:

```dotenv
DATABASE_URL=sqlite:///./app.db
```

Schema changes are managed by Alembic:

```bash
source .venv/bin/activate
cd backend
python -m alembic current
python -m alembic upgrade head
```

The repository/service layers avoid SQLite-specific business logic. To move to PostgreSQL, install a supported driver and change only the environment URL, for example:

```dotenv
DATABASE_URL=postgresql+psycopg://app_user:strong_password@db-host:5432/ml_app
```

Do not commit database credentials. PostgreSQL is recommended before multi-instance or real production deployment because SQLite has limited concurrent-write capacity.

## Train the first model

Prediction requires a trained model. First add at least `MIN_TRAINING_SAMPLES` labelled samples containing both label `0` and label `1`. Samples can be submitted from the dashboard or through `POST /api/data`.

Then run manual training from the repository root:

```bash
source .venv/bin/activate
cd backend
python -m app.ml.train
```

The command stores a versioned scikit-learn Pipeline under `models/` and creates its model-registry record. After training, refresh the dashboard. The serving-model card should show the active version.

If the UI says `No trained model available`, it means no usable registry artifact exists yet; it is not a frontend error.

## Automatic retraining

Verified prediction feedback creates lineage-linked training samples. Once the configured threshold is reached:

1. FastAPI commits a queued training run.
2. The run ID is published to the Celery `training` queue.
3. The worker creates a reproducible training snapshot.
4. A candidate pipeline is trained outside the API process.
5. Candidate and active model are evaluated on the same held-out data.
6. The candidate is promoted only when it passes configured acceptance rules.

The active model remains available if training, evaluation, or artifact creation fails.

Useful status commands:

```bash
curl http://localhost:8000/api/training/status
curl http://localhost:8000/api/system/worker-health
curl -X POST http://localhost:8000/api/training/check
curl -X POST http://localhost:8000/api/training/reconcile
```

## Important environment variables

| Variable | Purpose | Development default |
| --- | --- | --- |
| `APP_NAME` | FastAPI application name | `Continuous Learning ML API` |
| `DATABASE_URL` | SQLAlchemy connection URL | `sqlite:///./app.db` |
| `MODEL_DIR` | Model artifact directory | `models` |
| `VITE_API_BASE_URL` | Browser-visible FastAPI URL | `http://localhost:8000` |
| `FRONTEND_ORIGIN` | Primary allowed CORS origin | `http://localhost:5173` |
| `AUTO_RETRAIN_ENABLED` | Enables threshold eligibility checks | `true` |
| `MIN_TRAINING_SAMPLES` | Minimum complete labelled dataset | `100` |
| `RETRAIN_MIN_NEW_SAMPLES` | New feedback needed to trigger retraining | `50` |
| `CELERY_BROKER_URL` | Celery Redis broker | `redis://localhost:6379/0` |
| `CELERY_RESULT_BACKEND` | Celery result storage | `redis://localhost:6379/1` |
| `REDIS_URL` | Health checks and distributed lock | `redis://localhost:6379/2` |
| `WORKER_HEALTH_CACHE_SECONDS` | Cache duration for Redis/worker probes | `5.0` |

See `.env.example` for every supported development setting.

Vite environment values are embedded when the frontend process starts. Restart `npm run dev` after changing `VITE_API_BASE_URL`.

## API overview

Core data and prediction endpoints:

- `GET /health`
- `POST /api/data`
- `GET /api/data?skip=0&limit=20`
- `GET /api/data/{id}`
- `POST /api/predict`
- `GET /api/predictions?skip=0&limit=20`
- `GET /api/predictions/{prediction_id}`
- `PATCH /api/predictions/{prediction_id}/feedback`
- `GET /api/predictions/feedback/summary`
- `GET /api/model/status`

Training and monitoring endpoints:

- `POST /api/training/check`
- `GET /api/training/status`
- `GET /api/training/runs?skip=0&limit=20`
- `GET /api/training/runs/{training_run_id}`
- `POST /api/training/reconcile`
- `GET /api/monitoring/drift`
- `GET /api/monitoring/performance`
- `GET /api/monitoring/health`
- `POST /api/monitoring/check`
- `GET /api/models?skip=0&limit=20`
- `GET /api/models/{model_version}`
- `GET /api/models/compare?model_a=model_v1&model_b=model_v2`
- `POST /api/models/{model_version}/rollback`
- `GET /api/system/worker-health`

Interactive request/response schemas are always available from `/docs` while FastAPI is running.

The dashboard overview separates three counts that have different meanings:

- **Total stored samples** includes manual entries and feedback-derived records.
- **Labelled samples** includes every stored record with a label and can be used by manual training.
- **Verified feedback samples** includes only real outcomes submitted for predictions. These records drive the automatic-retraining minimum and new-feedback threshold.

## Tests and production build

Run all backend tests. The suite uses temporary SQLite databases and temporary model directories; it does not modify `backend/app.db` or real artifacts.

```bash
source .venv/bin/activate
python -m pytest backend -q
```

Build the frontend exactly as it would be compiled for deployment:

```bash
cd frontend
npm run build
```

Preview the compiled frontend locally:

```bash
cd frontend
npm run preview -- --host 0.0.0.0 --port 4173
```

## Troubleshooting

### `No trained model available`

Add enough labelled samples and run `python -m app.ml.train`. Confirm the backend and worker use the same `MODEL_DIR`.

### `redis_available=false`

Run `docker compose up redis`, then verify `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, and `REDIS_URL` in `.env`.

### `celery_worker_available=false`

Run `docker compose up -d redis`, then start the worker with the exact `celery ... -Q training --concurrency=1` command above. The queue name must match `TRAINING_QUEUE_NAME`. In Codespaces, do not replace the Celery/Redis `localhost` URLs with `app.github.dev` URLs: Redis is consumed inside the VM, not by the browser.

### Codespaces frontend cannot call FastAPI

Run `bash .devcontainer/setup-codespaces.sh`, restart Vite, and confirm port `8000` is public in the **Ports** tab. Also confirm `VITE_API_BASE_URL` points to the current Codespace rather than an older Codespace URL.

### A training job remains queued

Restore Redis and the worker, inspect worker logs, then call:

```bash
curl -X POST http://localhost:8000/api/training/reconcile
```

### SQLite reports lock errors

Keep Celery concurrency at one. For higher write concurrency or multiple application instances, migrate to PostgreSQL.

## Security and deployment note

The frontend is structured and styled for a production-quality user experience, but the full system is not ready for public production exposure yet. Administrative rollback and monitoring actions do not have authentication or authorization. Before a real deployment, add authentication, role-based access, HTTPS at the edge, managed PostgreSQL/Redis, secret management, durable artifact storage, observability/alerts, backups, and worker supervision.

Never commit `.env`, database files, model binaries, Redis data, frontend `dist/`, or Python/Node dependency directories.
