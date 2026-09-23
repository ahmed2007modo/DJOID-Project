# MusiQ — Music AI Quality Evaluation & Error Detection

**An independent, open-source prototype.** MusiQ does **not** connect to, embed, or reproduce any private or proprietary systems. No internal data, code, or tooling from any closed platform is used or shipped here; everything in this repository is built from public APIs, public datasets, and open-source libraries only.

MusiQ is a reliability-focused pipeline for music-genre audio classification: it trains an embedding-based classifier, evaluates how trustworthy its predictions actually are, flags predictions that are likely errors, and lets a human review them.

## Why this exists

Most music-classifier demos report one number (accuracy) and stop. MusiQ treats **trust** as the problem:

- **Calibration** — Expected Calibration Error (ECE) and Maximum Calibration Error (MCE): does 90% "confidence" really mean 90% correct?
- **Reliability diagrams** — visualize confidence vs. observed accuracy per bin.
- **Suspicious-prediction detection** — a transparent, weighted risk score combining low confidence, margin, entropy, distance from the training distribution, neighbor disagreement, and the historical error rate of the predicted class. Every risk reason is emitted as human-readable text.
- **Confirmed-error tracking** — when ground truth is known and the prediction is wrong, the failure is surfaced, not hidden.
- **Human review loop** — reviewers mark predictions correct/incorrect; the review syncs back into the system (the reviewer always has the final say — the ML never overrides a human judgment).

## Architecture

```
frontend/   React + TypeScript + Vite + Tailwind (SPA)
backend/    FastAPI + SQLAlchemy + Alembic + MLflow
├── app/api          REST endpoints (tracks, analyze, evaluation, reviews, training)
├── app/ml           embedding backends, metrics, calibration, risk, classifier, GTZAN pipeline
├── app/services     validation, preprocessing, embedding cache, registry, prediction, evaluation
├── app/models       ORM entities
└── alembic          schema migrations
scripts/    CLI: prepare_dataset.py, train.py, evaluate_test.py
docker-compose.yml   postgres + backend + nginx-frontend
.github/workflows/   CI (backend pytest + frontend build)
```

Pipeline: `audio upload → validate → preprocess → embed (cached) → classifier → risk → status (normal / suspicious / confirmed_error) → review`

## Quickstart (local, mock embeddings)

Requires Python 3.12, ffmpeg, Node 20+.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows; Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements-dev.txt

# run the test suite (no torch needed; mock embedding backend, temp sqlite)
$env:PYTHONPATH="backend"; python -m pytest backend/tests -q

# run the API
alembic -c backend/alembic.ini upgrade head   # env: MUSIQ_DATABASE_URL
uvicorn app.main:app --app-dir backend --reload
```

> On Windows PowerShell set env vars on one line, e.g.
> `$env:PYTHONPATH="E:\...\backend"; uvicorn app.main:app --app-dir backend --port 8000`
> Config is read from environment variables prefixed `MUSIQ_` (see `backend/app/core/config.py`).

The app **works out of the box with the deterministic `mock` embedding backend** so the whole flow (upload → train → analyze → review) runs without installing PyTorch. Set `MUSIQ_EMBEDDING_BACKEND=clap` (needs torch + transformers) to use the real CLAP embedding model — results then differ (and improve) from mock.

Frontend:

```bash
cd frontend
npm install
npm run dev        # serves on http://localhost:5173, proxies API to :8000
```

## Quickstart (Docker Compose)

```bash
docker compose up --build
# backend:  http://localhost:8000/docs
# frontend: http://localhost:5173
```

Compose runs PostgreSQL 16, applies Alembic migrations on start, and keeps models/media/MLflow in the `musiqdata` volume.

## Dataset (GTZAN)

By default the project uses the public **GTZAN** collection via Hugging Face
(`mtg-upf/gtzan-genre-recognition`, 10 genres, 100 × 30s tracks).

```bash
python scripts/prepare_dataset.py --download   # downloads & builds manifest + splits (needs ~1-2 GB)
python scripts/train.py                        # trains on train split, evaluates on val, registers model
python scripts/evaluate_test.py                # frozen-model evaluation on the untouched test split
```

Splits are **artist-aware** (tracks by the same artist never span train/val/test) to avoid optimistic leakage; known GTZAN duplicates/wrong-labels caveats are documented in `backend/app/ml/gtzan.py`. For quick local experimentation you can also build a tiny synthetic dataset via the API tests' seeding logic.

> The full GTZAN + CLAP run is a **manual, heavy step** by design. No results in this repo are fabricated: everything committed here was verified on the synthetic/mock path (114 tests).

## Configuration (env vars)

| Variable | Default | Purpose |
| --- | --- | --- |
| `MUSIQ_DATABASE_URL` | `sqlite:///./musiq.db` | SQLAlchemy URL (use `postgresql+psycopg2://` in prod) |
| `MUSIQ_EMBEDDING_BACKEND` | `clap` | `clap`, `mert`, or `mock` (falls back to mock without torch) |
| `MUSIQ_EMBEDDING_DIM` | `64` | Mock-backend dimension |
| `MUSIQ_AUTO_TRAIN_ON_START` | `true` | Auto-train a model at startup when data exists |
| `MUSIQ_RISK_THRESHOLD` | `0.6` | Risk above this ⇒ `suspicious` |
| `MUSIQ_MLFLOW_TRACKING_URI` | `./mlruns` | MLflow tracking location |
| `MUSIQ_MEDIA_DIR`, `MUSIQ_DATASET_DIR`, `MUSIQ_MODELS_DIR` | `data/…` | On-disk storage |

## API (summary)

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/tracks` | Upload & validate an audio file |
| `POST` | `/analyze` | Full predict + risk on a track |
| `GET` | `/tracks/{id}` · `/predictions/{id}` | Details |
| `GET` | `/evaluations` · `/metrics` · `/errors` · `/suspicious` | Evaluation & error surfaces |
| `POST` | `/reviews` | Human review of a prediction |
| `GET` | `/models` · `/models/compare` | Registry & comparison |
| `POST` | `/ml/train` | Train + evaluate + register a version |
| `GET` | `/dashboard/summary` | Overview for the UI |

Interactive docs are served at `/docs`.

## Tests

`backend/tests` (114 tests) covers: audio validation, preprocessing, mock embedding + cache, metrics (hand-computed), calibration (hand-computed ECE), risk scoring, GB of the dataset split logic, DB constraints, the full API flow (upload→train→analyze→review→suspect), and anomaly handling — all offline, no network, no torch.

## Roadmap / known limitations

- Mock embeddings are deterministic signal features — good for testing the *pipeline*, not for real accuracy. Real numbers require the CLAP run.
- `MUSIQ_EMBEDDING_BACKEND=clap` installs Hugging Face Transformers + audio dependencies; torch is intentionally not in the base requirements to keep CI fast.
- The review loop currently does not feed corrected labels back into training; that is a deliberate next-phase feature (`HumanReview` model already stores the data to enable it).
- No auth is included (local tooling); add reverse-proxy auth for shared deployments.