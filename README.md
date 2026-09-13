# Continuous Learning ML App

A Phase 1 full-stack scaffold with a React + Vite frontend and a FastAPI backend. The current application validates a temporary feature payload and echoes it to the browser. Database persistence, machine learning, background training, and authentication are intentionally outside this phase.

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

Both development servers must listen on all interfaces. Start the backend with the command above, then start the frontend with its public backend URL:

```bash
cd frontend
VITE_API_BASE_URL="https://${CODESPACE_NAME}-8000.app.github.dev" npm run dev -- --host 0.0.0.0
```

Set ports 5173 and 8000 to the visibility appropriate for your Codespace. Backend CORS accepts the local Vite origin and Codespaces origins on port 5173; both are configurable in `.env`.

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

## Phase 1 API payload

`POST /api/data` accepts JSON in this shape:

```json
{
  "feature_1": 1.25,
  "feature_2": 2.5,
  "feature_3": -3.75,
  "label": 1
}
```

`feature_1`, `feature_2`, and `feature_3` are required numeric values. `label` is an optional integer. The endpoint validates the request with Pydantic and returns the same fields; when the label is omitted, the response contains `"label": null`. It does not persist data or invoke model training.
