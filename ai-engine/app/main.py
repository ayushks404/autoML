import os
import traceback
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

import joblib
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.automl import run_automl
from app.explainability import get_top_features
from app.insights import generate_insights
from app.preprocessing import build_features, detect_target_column
from app.rag.vector_store import retrieve_similar, store_experiment

app = FastAPI(title="InsightForge AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DJANGO_BASE_URL = os.environ.get("DJANGO_BASE_URL", "http://127.0.0.1:8000/api/projects").rstrip("/")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")


class TrainingRequest(BaseModel):
    dataset_path: str
    target_column: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ai-engine running"}


def _post_with_retry(url, json_payload, retries=3, timeout=15):
    last_err = None
    for _ in range(retries):
        try:
            r = requests.post(url, json=json_payload, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
    raise last_err


def _report_failure(project_id: int, error_text: str):
    print(f"❌ TRAINING FAILED for project {project_id}: {error_text}")
    try:
        _post_with_retry(f"{DJANGO_BASE_URL}/training-failed/{project_id}/", {"error": error_text})
    except Exception as e:
        print("❌ Could not report failure to Django either:", e)


@app.post("/start-training/{project_id}")
def start_training(project_id: int, req: TrainingRequest):
    try:
        if not os.path.exists(req.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {req.dataset_path}")

        df = pd.read_csv(req.dataset_path)
        print(f"✅ Dataset loaded for project {project_id}: {df.shape[0]} rows, {df.shape[1]} cols")

        if df.shape[0] < 10:
            raise ValueError("Dataset has too few rows to train on (need at least 10).")

        target_col = req.target_column or detect_target_column(df)
        print("🎯 Target column:", target_col)

        X, y, task_type, label_map, dropped_cols = build_features(df, target_col)
        print(f"🧹 Preprocessed: task={task_type}, features={X.shape[1]}, dropped={dropped_cols}")

        if X.shape[1] == 0:
            raise ValueError("No usable feature columns remained after preprocessing.")
        if y.nunique(dropna=True) < 2:
            raise ValueError(f"Target column '{target_col}' has fewer than 2 distinct values; cannot train.")

        best, all_results, X_train, X_test, y_train, y_test = run_automl(X, y, task_type)
        if best["model"] is None:
            raise ValueError("No model could be trained successfully. See per-model errors: " + str(all_results))

        print(f"🏆 Best model: {best['name']} ({best['metric_name']}={best['score']})")

        for r in all_results:
            try:
                _post_with_retry(
                    f"{DJANGO_BASE_URL}/experiment-result/{project_id}/",
                    {"model_name": r["name"], "metric_name": r.get("metric_name") or "n/a", "metric_value": r.get("score")},
                    retries=1,
                )
            except Exception as e:
                print(f"⚠️ experiment log failed for {r['name']}:", e)

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"model_{project_id}.pkl")
        joblib.dump(best["model"], model_path)

        top_features, explain_method = get_top_features(best["model"], best["name"], X_test, task_type)
        print(f"🔍 Top features ({explain_method}):", top_features)

        dataset_info = f"{df.shape[0]} rows, {df.shape[1]} columns, target='{target_col}', task={task_type}"
        summary_text = (
            f"Dataset: {dataset_info}. Best model: {best['name']}. "
            f"{best['metric_name']}={round(best['score'], 4) if best['score'] is not None else 'n/a'}. "
            f"Top features: {top_features}."
        )
        store_experiment(summary_text)
        past_context = retrieve_similar(dataset_info)

        insights = generate_insights(top_features, dataset_info, task_type, target_col, past_context)

        payload = {
            "task_type": task_type,
            "target_column": target_col,
            "best_model": best["name"],
            "metric_name": best["metric_name"],
            "metric_value": best["score"],
            "accuracy": best["metrics"].get("accuracy") if task_type == "classification" else None,
            "all_metrics": best["metrics"],
            "model_path": model_path,
            "top_features": top_features,
            "insights": insights,
        }
        _post_with_retry(f"{DJANGO_BASE_URL}/training-result/{project_id}/", payload)

        return {"status": "training completed", "best_model": best["name"], "metric_name": best["metric_name"], "score": best["score"]}

    except Exception as e:
        error_text = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        _report_failure(project_id, error_text)
        return {"status": "training failed", "error": str(e)}
