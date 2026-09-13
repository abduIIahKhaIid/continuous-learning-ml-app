# Continuous Learning ML App

A Phase 2 full-stack application with a React + Vite frontend, FastAPI backend, and persistent SQLAlchemy storage. Incoming samples are validated, saved, and returned to the browser. Machine learning, background training, prediction, and authentication remain outside this phase.

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

Because the documented backend command runs from `backend/`, SQLite creates the ignored database file at `backend/app.db`. FastAPI creates the current development tables during application startup.

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

`feature_1`, `feature_2`, and `feature_3` are required numeric values. `label` is an optional integer. The POST endpoint returns the persisted record, including its ID, UTC timestamps, and future-training metadata. Listing is ordered by ID and accepts `skip >= 0` plus `limit` from 1 through 100.

## PostgreSQL readiness and migrations

Persistence is isolated behind a repository interface, so services and API routes do not change when the database changes. A later PostgreSQL configuration can use a SQLAlchemy URL such as:

```dotenv
DATABASE_URL=postgresql+psycopg://user:password@localhost/app
```

Install the appropriate PostgreSQL driver when making that switch. Alembic is intentionally not included in Phase 2: there is only one initial development schema, and `Base.metadata.create_all()` is sufficient for clean local initialization. Add Alembic before the first deployed schema change or whenever existing databases must be upgraded without recreation.

Tests override FastAPI's database dependency and create a fresh SQLite file in pytest's temporary directory for every test. They never use `backend/app.db`.
