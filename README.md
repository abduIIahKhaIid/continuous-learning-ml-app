# Continuous Learning ML App

A Phase 3 full-stack application with a React + Vite frontend, FastAPI backend, persistent SQLAlchemy storage, and a manually invoked baseline training pipeline. Labelled samples train a versioned scikit-learn pipeline. Prediction, automatic or incremental retraining, background workers, and authentication remain outside this phase.

## Prerequisites

- Python 3.11 or newer
- Node.js 20.19 or newer
- npm 10 or newer

## Configuration

From the repository root, create the local environment file:

```bash
cp .env.example .env
```

The checked-in defaults connect the Vite development server at `http://localhost:5173` to FastAPI at `http://localhost:8000`. The frontend reads the backend URL from `VITE_API_BASE_URL`; application components do not hardcode it.

`DATABASE_URL` is required by the backend. The development value is:

```dotenv
DATABASE_URL=sqlite:///./app.db
```

Because the documented backend command runs from `backend/`, SQLite creates the ignored database file at `backend/app.db`. FastAPI applies pending Alembic migrations during application startup. An existing Phase 2 database is safely stamped at its known baseline before the Phase 3 migration runs.

Initial training is configured through the same root `.env` file:

```dotenv
MIN_TRAINING_SAMPLES=20
ML_TEST_SIZE=0.2
ML_RANDOM_STATE=42
MODEL_DIR=models
```

Relative model paths are resolved from the repository root, not the shell's current directory.

## Start the backend

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements-dev.txt
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend exposes:

- `GET http://localhost:8000/health`
- `POST http://localhost:8000/api/data`
- `GET http://localhost:8000/api/data?skip=0&limit=20`
- `GET http://localhost:8000/api/data/{id}`
- API documentation at `http://localhost:8000/docs`

## Start the frontend

In a second terminal, from the repository root:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### GitHub Codespaces

The dev container automatically forwards ports 5173 and 8000 whenever you attach. It also runs `.devcontainer/setup-codespaces.sh`, which:

- builds the frontend and backend URLs from the Codespaces-provided environment variables;
- writes them to the ignored `.env` file; and
- makes backend port 8000 public through GitHub CLI.

After rebuilding the dev container once, start the frontend with:

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

No manual URL replacement or port-visibility change is normally needed. An organization-level Codespaces policy can still prohibit public ports; in that case the setup script prints a warning.

## Run checks

Run the backend tests from the repository root after activating the virtual environment:

```bash
cd backend
python -m pytest
```

Build the frontend from the repository root:

```bash
cd frontend
npm install
npm run build
```

## Phase 2 data API

`POST /api/data` accepts JSON in this shape:

```json
{
  "feature_1": 1.25,
  "feature_2": 2.5,
  "feature_3": -3.75,
  "label": 1
}
```

`feature_1`, `feature_2`, and `feature_3` are required numeric values. `label` is optional and, when supplied, must be `0` or `1`. The POST endpoint returns the persisted record, including its ID, UTC timestamps, and training metadata. Listing is ordered by ID and accepts `skip >= 0` plus `limit` from 1 through 100.

## Run initial model training

Add at least the configured number of labelled samples through `POST /api/data` or the frontend form, then run:

```bash
cd backend
python -m app.ml.train
```

The command loads labelled rows, validates binary labels, performs a reproducible train/test split, fits a single scikit-learn `Pipeline` containing median imputation, standard scaling, and logistic regression, evaluates the held-out set, and writes a concise summary.

To apply database migrations manually without starting FastAPI or training:

```bash
cd backend
python -m alembic upgrade head
```

Successful artifacts are stored as immutable files under the root `models/` directory:

```text
models/model_v1.joblib
models/model_v2.joblib
```

Model binaries are ignored by Git. Every attempt reserves a unique model version and training batch. Completed runs store metrics, confusion matrix, parameters, artifact checksum, exact sample IDs, and status in `training_runs`. Sample training metadata is updated in the same final transaction. Failed runs retain their error details and do not modify sample training flags.

## PostgreSQL readiness and migrations

Persistence is isolated behind a repository interface, so services and API routes do not change when the database changes. A later PostgreSQL configuration can use a SQLAlchemy URL such as:

```dotenv
DATABASE_URL=postgresql+psycopg://user:password@localhost/app
```

Install the appropriate PostgreSQL driver when making that switch. Alembic now owns schema evolution: `phase2_samples` records the original table and `phase3_training_runs` adds the training registry. Startup runs upgrades automatically, while the manual command above remains available for controlled environments.

Tests create fresh SQLite files and model directories under pytest's temporary directory. They never use `backend/app.db` or the real `models/` artifact directory.
