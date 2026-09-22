# InsightForge AI — Rebuilt

An agentic AutoML platform: upload any tabular CSV, and it auto-detects the
target column and task type (classification or regression), trains several
candidate models, picks the best one, explains it with SHAP, and generates a
plain-English summary with an LLM (with memory of past runs via RAG).

This is a rebuild of the original project, focused on fixing the issues that
caused projects to get stuck at `status: "training"` with no explanation,
and on making the pipeline dataset-agnostic instead of Titanic-shaped.

## What changed vs. the original

- **Generic target/task detection** (`ai-engine/app/preprocessing.py`): no
  hardcoded column names or dataset assumptions. Detects the target column by
  name hints, else the last non-ID-like column; detects classification vs.
  regression from the target's values. Tested against real churn data, a
  synthetic regression dataset, and small edge cases.
- **Task-aware AutoML** (`ai-engine/app/automl.py`): separate model registries
  for classification and regression, with metrics appropriate to each
  (ROC AUC / F1 for classification, R² / RMSE for regression).
- **Robust explainability** (`ai-engine/app/explainability.py`): SHAP first,
  automatic fallback to permutation importance if SHAP fails for any reason
  — training never crashes because of the explainability step.
- **LLM insights always return a string** (`ai-engine/app/insights.py`): the
  original bug where a failed LLM call sent back a Python list (which the
  frontend couldn't render) is fixed at the source. Falls back to a templated
  summary if no API key is configured or the call fails.
- **RAG memory degrades gracefully** (`ai-engine/app/rag/vector_store.py`):
  if FAISS/sentence-transformers can't load (no network, no local model
  cache), training continues without memory instead of crashing.
- **No more silent failures**: the AI engine now reports failures back to
  Django via a new `training-failed` endpoint, with the real error message
  attached. In the original, an exception mid-training was swallowed by a
  bare `except: print(...)`, leaving the project stuck at `training` forever.
- **Django hardening**: env-based settings, dataset validation (`.csv` only,
  size limit), a `failed` status with `error_message`, and `insights` is now
  a `TextField` (was `JSONField`, which is what let the list/string bug reach
  the database in the first place).
- **Frontend**: shows projects in every status (`created` / `training` /
  `completed` / `failed`), not just `completed`; polls automatically while
  anything is training; has an actual "create project" form (the original
  had no way to create a project from the UI).

## Architecture

```
Frontend (React/Vite)
      |
      v
Django (REST API, project state, dataset storage)
      |  POST /start-training/{id}
      v
FastAPI AI Engine
  -> preprocessing (auto target/task detection)
  -> AutoML (multiple models, best one selected)
  -> SHAP / permutation importance
  -> FAISS RAG memory (past experiments)
  -> LLM insights (OpenRouter)
      |  POST /training-result/{id}  or  /training-failed/{id}
      v
Django (stores result / error)
```

## Setup

### 1. Backend (Django)
```bash
cd core-backend/platform_core
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r ../requirements.txt   # see below, or pip install django djangorestframework django-cors-headers requests
python manage.py migrate
python manage.py runserver 8000
```

Environment variables (optional, sensible defaults for local dev):
- `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`
- `AI_ENGINE_URL` (default `http://127.0.0.1:8001`)

### 2. AI Engine (FastAPI)
```bash
cd ai-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENROUTER_API_KEY if you want real LLM insights
uvicorn app.main:app --reload --port 8001
```

Without `OPENROUTER_API_KEY` set, the engine still works end-to-end — it
just falls back to a templated, feature-based summary instead of an
LLM-written one. Same for RAG memory: if it can't reach HuggingFace to
download the embedding model, it silently skips memory rather than failing.

### 3. Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
By default it talks to `http://127.0.0.1:8000/api/projects`. Override with
a `.env` file containing `VITE_API_BASE_URL=...` if needed.

## Notes

- Only `.csv` datasets are accepted (validated server-side, 50MB max).
- The `OPENROUTER_API_KEY` in the *original* project's source code was
  hardcoded and exposed. If that repo was ever pushed publicly, treat the
  key as compromised and rotate it — this rebuild loads it from an
  environment variable instead.
- This still uses the same simple HTTP-callback architecture as the
  original (Django → FastAPI → Django), just hardened with retries and
  error propagation, per your call not to introduce a task queue.
