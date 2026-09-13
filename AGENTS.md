# Project development rules

These rules apply to the entire repository.

## Architecture and scope

- Keep the project as a monorepo with `frontend/` for React and `backend/` for Python.
- Keep HTTP handling, application services, persistence, and machine learning as separate layers.
- Prefer small, explicit modules over framework-heavy abstractions or premature microservices.
- Do not add functionality outside the task being requested. Record material architecture decisions in `docs/decisions/` when that directory is introduced.

## Frontend

- Use React with Vite. Prefer TypeScript for new frontend code.
- Organize product code by feature; keep shared UI, hooks, and infrastructure in clearly named shared directories.
- Access the backend through one API client layer. Do not scatter raw HTTP calls across components.
- Treat the FastAPI OpenAPI schema as the API contract and generate or derive frontend request and response types from it when practical.
- Keep domain and model-training decisions out of browser code.

## Backend

- Use FastAPI for HTTP delivery and Pydantic models at system boundaries.
- Keep route handlers thin: validate input, call an application service, and translate the result into an HTTP response.
- Do not place SQL queries, feature engineering, model fitting, model selection, or artifact handling in route modules.
- Put business workflows in services and persistence operations behind repository interfaces.
- Keep configuration in typed settings loaded from environment variables. Never commit secrets.
- Version public routes under `/api/v1` once endpoints are introduced.

## Persistence

- Use SQLite for local development while keeping schema and queries compatible with PostgreSQL.
- Use SQLAlchemy 2.x for persistence and Alembic for every schema change when the database layer is implemented.
- Avoid SQLite-specific SQL and behavior. Use portable column types, explicit constraints, UTC timestamps, and deliberate transaction boundaries.
- Store incoming examples, labels, predictions, training jobs/runs, model-version metadata, evaluation metrics, and the active-model reference in the database.
- Store serialized model binaries outside relational rows. Persist an artifact URI/path, checksum, format, and model metadata in the database.
- Preserve label history or revisions when labels can be corrected; do not silently overwrite training provenance.

## Machine learning and training

- Keep reusable ML code independent of FastAPI and database implementation details.
- Use a scikit-learn `Pipeline` so preprocessing and estimation are versioned and serialized together.
- Use the same feature transformation path for training and inference.
- Never fit or retrain a model directly inside an API request. An API request may persist data and create a durable training job or event.
- Run training through a separate service/worker entry point. The initial worker may poll the database; its interface must allow a real background queue to be added later.
- Make training jobs idempotent and safe against concurrent execution. Record the labelled-data watermark or exact training-data selection for every run.
- Support estimators with `partial_fit` where appropriate. For estimators without incremental support, schedule controlled batch retraining outside request handling.
- Treat every trained model as a candidate version. Record its parameters, code/data provenance, artifact location, and evaluation metrics.
- Evaluate candidates against explicit, configurable acceptance criteria and the current active model. Promote only candidates that pass.
- Change the active-model reference atomically. A failed training or evaluation run must leave the current model available.
- Load model artifacts through a dedicated inference service and record the model version used for each persisted prediction.

## Testing and quality

- Add focused unit tests for services, feature transformations, evaluation gates, and promotion decisions.
- Add integration tests for repositories against SQLite and keep a PostgreSQL test path available for database-specific validation.
- Test API behavior through FastAPI's test client without performing real training in request tests.
- Add frontend component tests for meaningful behavior and reserve root-level `tests/e2e/` for full-system flows.
- Use deterministic random seeds in ML tests and small synthetic fixtures rather than committed production data.
- Run the checks relevant to changed code before handing work back. Do not add tests that merely duplicate implementation details.

## Repository hygiene

- Do not commit secrets, local databases, uploaded data, production datasets, trained-model binaries, caches, virtual environments, or frontend build output.
- Keep only small documented sample data under `data/samples/`.
- Keep generated local model artifacts under an ignored `artifacts/` directory; do not use the same directory for ML source code.
- Update documentation and `.env.example` when configuration or developer commands change.
- Preserve unrelated user changes in the working tree.
